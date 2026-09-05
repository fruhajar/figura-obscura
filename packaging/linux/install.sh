#!/usr/bin/env bash
# Figura Obscura installer for the Linux tarball.
#
# Installs to ~/.local by default — no root, no package manager, and it lands in
# the XDG directories a desktop session already searches. Pass --prefix to put
# it somewhere else, or --uninstall to take it away again.
set -euo pipefail

prefix="${HOME}/.local"
do_uninstall=false
fetch_models=true

usage() {
    cat <<'USAGE'
Usage: ./install.sh [options]

  --prefix DIR    Install under DIR (default: ~/.local; use /usr/local for
                  a system-wide install, which needs sudo)
  --no-models     Skip downloading detection models
  --uninstall     Remove a previous installation
  -h, --help      This message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)    prefix="$2"; shift 2 ;;
        --no-models) fetch_models=false; shift ;;
        --uninstall) do_uninstall=true; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bindir="$prefix/bin"
libdir="$prefix/lib/figura-obscura"
appdir="$prefix/share/applications"
icondir="$prefix/share/icons/hicolor"
docdir="$prefix/share/doc/figura-obscura"

# --- uninstall --------------------------------------------------------------
if $do_uninstall; then
    echo "==> removing Figura Obscura from $prefix"
    rm -f "$bindir/obscura" "$bindir/obscura-gui"
    rm -rf "$libdir" "$docdir"
    rm -f "$appdir/figura-obscura.desktop"
    for size in 16 32 48 64 128 256 512; do
        rm -f "$icondir/${size}x${size}/apps/figura-obscura.png"
    done
    command -v update-desktop-database >/dev/null && \
        update-desktop-database "$appdir" 2>/dev/null || true
    cat <<MSG

Removed. Two things were deliberately left alone, because they are your data:
  models   ${XDG_CACHE_HOME:-$HOME/.cache}/figura-obscura
  settings ${XDG_CONFIG_HOME:-$HOME/.config}/figura-obscura
Delete those by hand if you want them gone.
MSG
    exit 0
fi

# --- install ----------------------------------------------------------------
[[ -f "$here/obscura" && -f "$here/obscura-gui" ]] || {
    echo "error: run this from inside the extracted Figura Obscura tarball." >&2
    exit 1
}

echo "==> installing Figura Obscura into $prefix"
mkdir -p "$bindir" "$libdir" "$appdir" "$docdir"

install -m 0755 "$here/obscura"     "$bindir/obscura"
install -m 0755 "$here/obscura-gui" "$bindir/obscura-gui"

# Bundled tools go in libdir: ob-media searches <exe>/../lib/figura-obscura, so
# they are found without putting a second ffmpeg on the user's PATH.
if [[ -d "$here/bin" ]]; then
    for tool in "$here/bin"/*; do
        [[ -f "$tool" ]] && install -m 0755 "$tool" "$libdir/$(basename "$tool")"
    done
fi
# GPU execution providers, when this is a GPU build.
for lib in "$here"/*onnxruntime_providers_*; do
    [[ -f "$lib" ]] && install -m 0755 "$lib" "$libdir/$(basename "$lib")"
done

install -m 0644 "$here/figura-obscura.desktop" "$appdir/figura-obscura.desktop"
for size in 16 32 48 64 128 256 512; do
    src="$here/assets/icon-${size}.png"
    if [[ -f "$src" ]]; then
        mkdir -p "$icondir/${size}x${size}/apps"
        install -m 0644 "$src" "$icondir/${size}x${size}/apps/figura-obscura.png"
    fi
done
for doc in THIRD-PARTY.md README.md; do
    [[ -f "$here/$doc" ]] && install -m 0644 "$here/$doc" "$docdir/$doc"
done
[[ -d "$here/licenses" ]] && cp -r "$here/licenses" "$docdir/"

command -v update-desktop-database >/dev/null && \
    update-desktop-database "$appdir" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -qtf "$icondir" 2>/dev/null || true

# --- models -----------------------------------------------------------------
if $fetch_models; then
    echo
    "$bindir/obscura" setup || {
        echo
        echo "Models could not be downloaded, but Figura Obscura is installed."
        echo "Run 'obscura setup' again when you are online, or use the Models page in the app."
    }
fi

# --- PATH hint --------------------------------------------------------------
case ":$PATH:" in
    *":$bindir:"*) ;;
    *)
        echo
        echo "Note: $bindir is not on your PATH. Add this to your shell profile:"
        echo "    export PATH=\"\$PATH:$bindir\""
        ;;
esac

cat <<MSG

Installed. Launch it from your applications menu, or run:
    obscura-gui        the desktop app
    obscura --help     the command-line tool
MSG
