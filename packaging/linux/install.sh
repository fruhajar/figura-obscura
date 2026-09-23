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
# GPU runtime libraries, when this is a GPU build. `libwebgpu_dawn.so` is not
# named like a provider but is a hard link-time dependency of a webgpu build --
# leave it behind and the installed binary does not start at all.
for lib in "$here"/*onnxruntime_providers_* "$here"/libwebgpu_dawn.so; do
    [[ -f "$lib" ]] && install -m 0755 "$lib" "$libdir/$(basename "$lib")"
done

# Exec is rewritten to an absolute path. The shipped entry says a bare
# `obscura-gui` because the AppImage needs it that way, but a bare name leaves
# the desktop session's PATH to decide which copy launches -- so any stray older
# binary (classically ~/.cargo/bin/obscura-gui) goes on opening from the
# applications menu long after this one is installed, and the menu silently
# reports the old version.
sed "s|^Exec=obscura-gui\\b|Exec=$bindir/obscura-gui|" \
    "$here/figura-obscura.desktop" > "$appdir/figura-obscura.desktop"
chmod 0644 "$appdir/figura-obscura.desktop"

for size in 16 32 48 64 128 256 512; do
    src="$here/assets/icon-${size}.png"
    if [[ -f "$src" ]]; then
        mkdir -p "$icondir/${size}x${size}/apps"
        install -m 0644 "$src" "$icondir/${size}x${size}/apps/figura-obscura.png"
    fi
done
for doc in LICENSE THIRD-PARTY.md README.md; do
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

# --- shadowed by another copy? ----------------------------------------------
# The menu entry is absolute, so it always opens the copy installed above, but
# another obscura-gui earlier on PATH still answers at the shell -- and it is
# the reason an old version appears to survive a reinstall. Worth naming both,
# with versions, rather than leaving it to be discovered.
# First line only: `--version` also reports the build's execution providers,
# which would bury this two-line note in a wall of text. A binary old enough to
# have no --version at all prints nothing, hence the fallback.
version_of() {
    local v
    v="$("$1" --version 2>/dev/null | head -n1)" || true
    if [[ -n "$v" ]]; then printf '%s\n' "$v"; else echo "version unknown"; fi
}
shadow="$(command -v obscura-gui 2>/dev/null || true)"
if [[ -n "$shadow" && "$shadow" != "$bindir/obscura-gui" ]]; then
    cat <<MSG

Note: 'obscura-gui' on your PATH is another copy:
    $shadow  ->  $(version_of "$shadow")
this install is:
    $bindir/obscura-gui  ->  $(version_of "$bindir/obscura-gui")
Remove the other one ('cargo uninstall ob-gui' if it came from cargo install),
or put $bindir earlier on your PATH.
MSG
fi

cat <<MSG

Installed. Launch it from your applications menu, or run:
    obscura-gui        the desktop app
    obscura --help     the command-line tool
MSG
