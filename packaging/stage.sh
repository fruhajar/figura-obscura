#!/usr/bin/env bash
# Build Figura Obscura and assemble a distributable payload directory.
#
# Every platform packager (Inno Setup, AppImage, .app/dmg) wraps the *same*
# staged tree, so "what ships" is defined once, here, rather than three times in
# three formats that drift apart.
#
# Layout produced in $STAGE:
#     obscura[.exe]                 CLI
#     obscura-gui[.exe]             desktop app
#     bin/ffmpeg[.exe]         bundled tools (ob-media looks here first)
#     bin/ffprobe[.exe]
#     onnxruntime_providers_*  only for GPU builds (CPU ORT is static)
#     THIRD-PARTY.md, licenses/
#     assets/                  icons
#
# Usage:
#   packaging/stage.sh [--gpu cuda|webgpu|none] [--ffmpeg DIR] [--out DIR]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="none"
ffmpeg_dir=""
allow_gpl=false
stage="$repo_root/target/stage"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    gpu="$2"; shift 2 ;;
        --ffmpeg) ffmpeg_dir="$2"; shift 2 ;;
        # Deliberately bundle a GPL ffmpeg. Obscura's own code is unaffected (it
        # spawns ffmpeg, never links libav), but you must then publish the
        # corresponding ffmpeg source. See THIRD-PARTY.md.
        --allow-gpl) allow_gpl=true; shift ;;
        --out)    stage="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

exe_suffix=""
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) exe_suffix=".exe" ;;
esac

# --- 1. build ---------------------------------------------------------------
features=()
case "$gpu" in
    none)   ;;
    cuda)   features=(--features ob-detect/cuda) ;;
    # AMD: `rocm` has no prebuilt for x86_64 linux and silently yields a
    # CPU-only binary. See HOST-BUILD.md.
    webgpu) features=(--features ob-detect/webgpu) ;;
    *) echo "error: --gpu must be one of none, cuda, webgpu" >&2; exit 1 ;;
esac

echo "==> building (release, gpu=$gpu)"
( cd "$repo_root" && cargo build --release --workspace "${features[@]}" )

# --- 2. stage ---------------------------------------------------------------
echo "==> staging into $stage"
rm -rf "$stage"
mkdir -p "$stage/bin" "$stage/licenses" "$stage/assets"

release="$repo_root/target/release"
install -m 0755 "$release/obscura$exe_suffix"     "$stage/obscura$exe_suffix"
install -m 0755 "$release/obscura-gui$exe_suffix" "$stage/obscura-gui$exe_suffix"

# GPU execution providers are separate shared libraries. The CPU ONNX Runtime is
# linked *statically* into both binaries, so a --gpu none build ships nothing
# extra — and must not, because the CUDA and TensorRT providers alone are around
# a gigabyte and are dead weight without a CUDA build.
#
# Two traps here, both hit in practice:
#   * `ort` leaves these as symlinks into its own per-user download cache
#     (~/.cache/ort.pyke.io/...). `-f` follows the link, so a stale one left by
#     an earlier build on another machine is skipped rather than aborting the
#     build; `install` dereferences, so real files are what get staged.
#   * The symlinks are present even for a CPU build, so their existence is not
#     evidence that a GPU build was requested — the flag is.
if [[ "$gpu" != "none" ]]; then
    shopt -s nullglob
    staged_providers=0
    for lib in "$release"/*onnxruntime_providers_*; do
        if [[ -f "$lib" ]]; then
            install -m 0755 "$lib" "$stage/$(basename "$lib")"
            echo "    + $(basename "$lib") ($(du -h "$lib" | cut -f1))"
            staged_providers=$((staged_providers + 1))
        else
            echo "    ! skipping dangling $(basename "$lib")" >&2
        fi
    done
    shopt -u nullglob
    if [[ "$gpu" == "cuda" && "$staged_providers" -eq 0 ]]; then
        echo "error: --gpu cuda but no execution-provider libraries were produced." >&2
        echo "       This is the silent-CPU-fallback trap described in HOST-BUILD.md." >&2
        exit 1
    fi
fi

# --- 3. bundled ffmpeg ------------------------------------------------------
if [[ -n "$ffmpeg_dir" ]]; then
    for tool in ffmpeg ffprobe; do
        src="$ffmpeg_dir/$tool$exe_suffix"
        [[ -f "$src" ]] || { echo "error: $src not found" >&2; exit 1; }
        licence_args=()
        $allow_gpl && licence_args+=(--allow-gpl)
        "$repo_root/packaging/common/check-ffmpeg-licence.sh" \
            "${licence_args[@]+"${licence_args[@]}"}" "$src"
        install -m 0755 "$src" "$stage/bin/$tool$exe_suffix"
    done
    # The bundled binary's licence text must travel with it. A hard failure,
    # not a warning: the omission is invisible in the finished artifact, so
    # nothing downstream ever catches it — which is exactly how a release can
    # go out noncompliant.
    if $allow_gpl; then
        licence_file="GPL-2.0.txt"
        licence_url="https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt"
    else
        licence_file="LGPL-2.1.txt"
        licence_url="https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt"
    fi
    if [[ ! -f "$repo_root/packaging/common/licenses/$licence_file" ]]; then
        cat >&2 <<MSG
error: packaging/common/licenses/$licence_file is missing.

You are bundling an ffmpeg binary, and it may not be distributed without its
licence text. Fetch it, then build again:

    curl -fsSLO --output-dir packaging/common/licenses $licence_url

(If that GPL ffmpeg was configured --enable-version3, it needs GPL-3.0.txt
instead.) See packaging/common/licenses/README.md.
MSG
        exit 1
    fi
else
    # A supported configuration, not a broken one: Obscura finds a user-installed
    # ffmpeg on PATH and, when it cannot, tells the user how to install one.
    # Shipping no ffmpeg carries no redistribution obligations at all.
    echo "note: no --ffmpeg DIR given — the build will use the user's own ffmpeg." >&2
fi

# --- 4. docs and assets -----------------------------------------------------
cp "$repo_root/packaging/common/THIRD-PARTY.md" "$stage/"
cp -r "$repo_root/packaging/common/licenses/." "$stage/licenses/" 2>/dev/null || true
cp "$repo_root/packaging/assets/"*.png "$stage/assets/" 2>/dev/null || true
cp "$repo_root/README.md" "$stage/README.md"
# The product's own licence. THIRD-PARTY.md covers what we redistribute;
# this covers what we wrote, and shipping binaries without it grants the
# recipient nothing.
cp "$repo_root/LICENSE" "$stage/LICENSE"

# --- 5. verify --------------------------------------------------------------
# Catch a broken stage now rather than in a customer's installer.
echo "==> verifying"
"$stage/obscura$exe_suffix" models list >/dev/null
echo "    obscura runs"
if [[ -x "$stage/bin/ffmpeg$exe_suffix" ]]; then
    "$stage/bin/ffmpeg$exe_suffix" -hide_banner -version >/dev/null
    echo "    bundled ffmpeg runs"
fi

du -sh "$stage"
echo "==> staged: $stage"
