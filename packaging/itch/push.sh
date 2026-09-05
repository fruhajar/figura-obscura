#!/usr/bin/env bash
# Publish the built artifacts to itch.io with butler.
#
#   packaging/itch/push.sh --user YOURNAME --game figura-obscura [--dry-run]
#
# Channel names matter to itch: the app picks a build by the platform prefix in
# the channel name (`windows`, `linux`, `osx`), so they are not arbitrary
# labels. Version comes from Cargo.toml so the itch build number always matches
# what the binaries report.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
user=""
game="figura-obscura"
dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)    user="$2"; shift 2 ;;
        --game)    game="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$user" ]] || { echo "error: --user is required (your itch.io username)" >&2; exit 1; }
# Only a real push needs butler; --dry-run exists to check which artifacts would
# go to which channel, which is useful before installing anything.
if ! $dry_run && ! command -v butler >/dev/null; then
    echo "error: butler not found. Install it from https://itch.io/docs/butler/" >&2
    exit 1
fi

version="$(awk '/^\[workspace\.package\]/{f=1} f && /^version/{gsub(/[",]/,"");print $3; exit}' "$repo_root/Cargo.toml")"
dist="$repo_root/target/dist"
[[ -d "$dist" ]] || { echo "error: nothing built — run the platform build scripts first" >&2; exit 1; }

echo "==> pushing Figura Obscura $version to $user/$game"

push() {
    local artifact="$1" channel="$2"
    if [[ ! -e "$artifact" ]]; then
        echo "    skip $channel (no $(basename "$artifact"))"
        return
    fi
    echo "    $channel <- $(basename "$artifact")"
    if $dry_run; then
        return
    fi
    butler push "$artifact" "$user/$game:$channel" --userversion "$version"
}

# Each platform build directory gets the manifest so the itch app can launch it.
stage_dir_with_manifest() {
    local src="$1" out="$2"
    rm -rf "$out"
    cp -r "$src" "$out"
    cp "$repo_root/packaging/itch/itch.toml" "$out/.itch.toml"
    echo "$out"
}

# Windows: push the installer. itch will run it, and the app's own uninstaller
# handles removal.
push "$dist/FiguraObscura-$version-windows-x64-setup.exe" "windows"

# Linux: push the *extracted* tarball directory, not the .tar.gz — butler diffs
# file trees, so pushing an archive re-uploads the whole thing every release.
linux_src="$repo_root/target/FiguraObscura-$version-linux-x86_64"
if [[ -d "$linux_src" ]]; then
    push "$(stage_dir_with_manifest "$linux_src" "$repo_root/target/itch-linux")" "linux"
fi

# macOS: same reasoning — push the .app tree.
if [[ -d "$repo_root/target/Figura Obscura.app" ]]; then
    mac_out="$repo_root/target/itch-macos"
    rm -rf "$mac_out"
    mkdir -p "$mac_out"
    cp -r "$repo_root/target/Figura Obscura.app" "$mac_out/"
    cp "$repo_root/packaging/itch/itch.toml" "$mac_out/.itch.toml"
    push "$mac_out" "osx"
fi

if $dry_run; then
    echo "==> dry run; nothing uploaded"
else
    echo "==> done. Check status with: butler status $user/$game"
fi
