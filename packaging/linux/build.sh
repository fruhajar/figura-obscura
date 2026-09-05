#!/usr/bin/env bash
# Build the Linux release artifacts: a portable tarball and an AppImage.
#
#   packaging/linux/build.sh [--ffmpeg DIR] [--gpu none|cuda|webgpu] [--skip-appimage]
#
# The tarball is the primary artifact — it works everywhere, it is what itch.io's
# app installs, and its install.sh wires up the desktop entry and icons. The
# AppImage is a convenience for people who want a single file, and needs
# `appimagetool` (downloaded on demand if absent).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ffmpeg_dir=""
gpu="none"
skip_appimage=false
allow_gpl=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ffmpeg)        ffmpeg_dir="$2"; shift 2 ;;
        --gpu)           gpu="$2"; shift 2 ;;
        --skip-appimage) skip_appimage=true; shift ;;
        --allow-gpl)     allow_gpl=true; shift ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

version="$(awk '/^\[workspace\.package\]/{f=1} f && /^version/{gsub(/[",]/,"");print $3; exit}' "$repo_root/Cargo.toml")"
[[ -n "$version" ]] || { echo "error: could not read version from Cargo.toml" >&2; exit 1; }
echo "==> Figura Obscura $version (linux x86_64, gpu=$gpu)"

dist="$repo_root/target/dist"
stage="$repo_root/target/stage"
name="FiguraObscura-$version-linux-x86_64"
mkdir -p "$dist"

# --- 1. shared staging ------------------------------------------------------
stage_args=(--gpu "$gpu" --out "$stage")
[[ -n "$ffmpeg_dir" ]] && stage_args+=(--ffmpeg "$ffmpeg_dir")
$allow_gpl && stage_args+=(--allow-gpl)
"$repo_root/packaging/stage.sh" "${stage_args[@]}"

# --- 2. tarball -------------------------------------------------------------
echo "==> building the tarball"
tarroot="$repo_root/target/$name"
rm -rf "$tarroot"
cp -r "$stage" "$tarroot"
cp "$repo_root/packaging/linux/install.sh"        "$tarroot/"
cp "$repo_root/packaging/linux/figura-obscura.desktop" "$tarroot/"
chmod +x "$tarroot/install.sh"

tar -C "$repo_root/target" -czf "$dist/$name.tar.gz" "$name"
echo "    $dist/$name.tar.gz ($(du -h "$dist/$name.tar.gz" | cut -f1))"

# --- 3. AppImage ------------------------------------------------------------
if $skip_appimage; then
    echo "==> skipping the AppImage"
else
    appdir="$repo_root/target/FiguraObscura.AppDir"
    echo "==> building the AppDir"
    rm -rf "$appdir"
    mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/figura-obscura" \
             "$appdir/usr/share/applications" \
             "$appdir/usr/share/icons/hicolor/256x256/apps"

    install -m 0755 "$stage/obscura"     "$appdir/usr/bin/obscura"
    install -m 0755 "$stage/obscura-gui" "$appdir/usr/bin/obscura-gui"
    # ob-media searches <exe>/../lib/figura-obscura, which is exactly this path.
    if [[ -d "$stage/bin" ]]; then
        find "$stage/bin" -maxdepth 1 -type f -exec install -m 0755 {} "$appdir/usr/lib/figura-obscura/" \;
    fi
    shopt -s nullglob
    for lib in "$stage"/*onnxruntime_providers_*; do
        install -m 0755 "$lib" "$appdir/usr/lib/figura-obscura/"
    done
    shopt -u nullglob

    install -m 0644 "$repo_root/packaging/linux/figura-obscura.desktop" \
        "$appdir/usr/share/applications/figura-obscura.desktop"
    # appimagetool requires the .desktop and the icon at the AppDir root too.
    cp "$appdir/usr/share/applications/figura-obscura.desktop" "$appdir/figura-obscura.desktop"
    install -m 0644 "$repo_root/packaging/assets/icon-256.png" \
        "$appdir/usr/share/icons/hicolor/256x256/apps/figura-obscura.png"
    cp "$appdir/usr/share/icons/hicolor/256x256/apps/figura-obscura.png" "$appdir/figura-obscura.png"
    cp "$appdir/figura-obscura.png" "$appdir/.DirIcon"

    # AppRun: resolve the AppDir and exec the GUI from it. `exec` matters — the
    # process the desktop launched must *become* the app, or the window manager
    # loses track of it and the taskbar entry never settles.
    cat > "$appdir/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
exec "$HERE/usr/bin/obscura-gui" "$@"
APPRUN
    chmod +x "$appdir/AppRun"

    tool="$(command -v appimagetool || true)"
    if [[ -z "$tool" ]]; then
        cached="$repo_root/target/appimagetool-x86_64.AppImage"
        if [[ ! -x "$cached" ]]; then
            echo "==> fetching appimagetool"
            curl -fsSL -o "$cached" \
                https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage \
                || { echo "error: could not download appimagetool; re-run with --skip-appimage" >&2; exit 1; }
            chmod +x "$cached"
        fi
        tool="$cached"
    fi

    echo "==> building the AppImage"
    # ARCH is required by appimagetool and is not inferred.
    ARCH=x86_64 "$tool" "$appdir" "$dist/$name.AppImage"
    echo "    $dist/$name.AppImage ($(du -h "$dist/$name.AppImage" | cut -f1))"
fi

echo
echo "==> artifacts in $dist"
ls -lh "$dist"
