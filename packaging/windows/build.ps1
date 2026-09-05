<#
.SYNOPSIS
    Build the Figura Obscura Windows installer end to end.

.DESCRIPTION
    Compiles the release binaries, stages them with a bundled LGPL FFmpeg, then
    compiles the Inno Setup installer.

    Run this on Windows. Cross-compiling is not supported: `ort` downloads a
    prebuilt ONNX Runtime for the *host* triple, and the bundled ffmpeg must be
    a Windows binary anyway.

.PARAMETER FfmpegDir
    Optional. Directory containing ffmpeg.exe and ffprobe.exe. Omit it to ship
    without a bundled ffmpeg: the app then finds the user's own on PATH, and the
    build carries no redistribution obligations at all. Images work either way;
    only video needs it. This mirrors packaging/stage.sh, where omitting
    --ffmpeg is likewise supported.

    When given, it must be an LGPL build —
    a GPL build would place all of Figura Obscura under the GPL. Get one from
    https://github.com/BtbN/FFmpeg-Builds (choose an asset with "lgpl" in the
    name, e.g. ffmpeg-n7.1-latest-win64-lgpl-shared-7.1.zip).

.PARAMETER Gpu
    none (default) or cuda. CUDA builds also require CUDA 12.x + cuDNN 9.x on
    the *user's* machine, so ship it as a separate optional download rather
    than as the main installer.

.EXAMPLE
    .\packaging\windows\build.ps1 -FfmpegDir C:\ffmpeg-lgpl\bin
#>
[CmdletBinding()]
param(
    [string]$FfmpegDir = '',
    [ValidateSet('none', 'cuda')][string]$Gpu = 'none',
    [string]$InnoSetup = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = 'Stop'
$repo  = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$stage = Join-Path $repo 'target\stage'

function Assert-Tool($path, $what, $hint) {
    if (-not (Test-Path $path)) {
        throw "$what not found at $path`n$hint"
    }
}

# --- 0. preflight -----------------------------------------------------------
Assert-Tool $InnoSetup 'Inno Setup (ISCC.exe)' `
    'Install it from https://jrsoftware.org/isdl.php, or pass -InnoSetup <path>.'
$bundleFfmpeg = -not [string]::IsNullOrWhiteSpace($FfmpegDir)
if ($bundleFfmpeg) {
    foreach ($tool in 'ffmpeg.exe', 'ffprobe.exe') {
        Assert-Tool (Join-Path $FfmpegDir $tool) $tool `
            'Point -FfmpegDir at the bin\ directory of an LGPL FFmpeg build.'
    }

    # Refuse a GPL/nonfree ffmpeg. Same rule the Unix staging script enforces; it
    # is reimplemented here because the shell script needs bash, which Windows
    # lacks.
    $banner = & (Join-Path $FfmpegDir 'ffmpeg.exe') -hide_banner -version 2>&1 | Out-String
    if ($banner -match '--enable-nonfree') {
        throw 'That FFmpeg is a --enable-nonfree build and cannot be redistributed at all.'
    }
    if ($banner -match '--enable-gpl') {
        throw @'
That FFmpeg is a --enable-gpl build. Bundling it would place all of Figura Obscura
under the GPL. Download an "lgpl" asset from https://github.com/BtbN/FFmpeg-Builds
instead.
'@
    }
    Write-Host "==> FFmpeg licence OK: $(($banner -split "`n")[0].Trim())"
} else {
    Write-Host '==> no -FfmpegDir given: shipping without a bundled ffmpeg.'
    Write-Host '    Video will require the user to have ffmpeg on PATH; images are unaffected.'
}

# --- 1. version -------------------------------------------------------------
# Read it from the workspace manifest so the installer filename, the wizard and
# the binaries can never disagree about what version this is.
$cargoToml = Get-Content (Join-Path $repo 'Cargo.toml') -Raw
if ($cargoToml -notmatch '(?ms)^\[workspace\.package\].*?^version\s*=\s*"([^"]+)"') {
    throw 'Could not read version from [workspace.package] in Cargo.toml'
}
$version = $Matches[1]
Write-Host "==> Figura Obscura $version"

# --- 2. build ---------------------------------------------------------------
Push-Location $repo
try {
    $features = @()
    if ($Gpu -eq 'cuda') { $features = @('--features', 'ob-detect/cuda') }

    Write-Host "==> cargo build --release (gpu=$Gpu)"
    & cargo build --release --workspace @features
    if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }

    # --- 3. stage -----------------------------------------------------------
    Write-Host "==> staging into $stage"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Path $stage, "$stage\bin", "$stage\licenses" | Out-Null

    Copy-Item "target\release\obscura.exe"     $stage
    Copy-Item "target\release\obscura-gui.exe" $stage
    if ($bundleFfmpeg) {
        Copy-Item (Join-Path $FfmpegDir 'ffmpeg.exe')  "$stage\bin"
        Copy-Item (Join-Path $FfmpegDir 'ffprobe.exe') "$stage\bin"

        # An LGPL shared build needs its DLLs beside the executables.
        Get-ChildItem $FfmpegDir -Filter *.dll -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName "$stage\bin" }
    }

    # GPU execution providers exist only in a CUDA build; the CPU runtime is
    # statically linked, so there is nothing to copy for -Gpu none.
    Get-ChildItem 'target\release' -Filter 'onnxruntime_providers_*' -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName $stage }

    Copy-Item 'packaging\common\THIRD-PARTY.md' $stage
    Copy-Item 'README.md' $stage
    if (Test-Path 'packaging\common\licenses') {
        Copy-Item 'packaging\common\licenses\*' "$stage\licenses" -ErrorAction SilentlyContinue
    }

    # --- 4. verify the stage before wrapping it -----------------------------
    Write-Host '==> verifying staged binaries'
    & "$stage\obscura.exe" models list | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'staged obscura.exe does not run' }
    if ($bundleFfmpeg) {
        & "$stage\bin\ffmpeg.exe" -hide_banner -version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'staged ffmpeg.exe does not run' }
    }

    # --- 5. installer -------------------------------------------------------
    Write-Host '==> compiling the installer'
    & $InnoSetup "/DStageDir=$stage" "/DAppVersion=$version" `
        'packaging\windows\figura-obscura.iss'
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed' }

    $out = Join-Path $repo "target\installer\FiguraObscura-$version-windows-x64-setup.exe"
    Write-Host ''
    Write-Host "==> installer: $out"
    Write-Host '    Sign it before release, or SmartScreen will warn every buyer:'
    Write-Host '    signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a "<path>"'
}
finally {
    Pop-Location
}
