#!/usr/bin/env bash
# Build Figura Obscura.app and a .dmg for macOS.
#
#   packaging/macos/build.sh --ffmpeg DIR [--sign "Developer ID Application: You (TEAMID)"]
#                            [--notarize-profile NAME] [--universal]
#
# Must run on macOS: `ort` downloads a prebuilt ONNX Runtime for the host
# triple, and codesign/notarytool are Apple tools.
#
# Signing is optional here but effectively mandatory for release. An unsigned,
# un-notarised app downloaded from itch.io is quarantined by Gatekeeper and
# shows "Figura Obscura is damaged and can't be opened" — which reads to a buyer
# as a broken product, not as a security setting.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ffmpeg_dir=""
sign_id=""
notarize_profile=""
universal=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ffmpeg)           ffmpeg_dir="$2"; shift 2 ;;
        --sign)             sign_id="$2"; shift 2 ;;
        --notarize-profile) notarize_profile="$2"; shift 2 ;;
        --universal)        universal=true; shift ;;
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "error: this script must run on macOS." >&2; exit 1; }

version="$(awk '/^\[workspace\.package\]/{f=1} f && /^version/{gsub(/[",]/,"");print $3; exit}' "$repo_root/Cargo.toml")"
app="$repo_root/target/Figura Obscura.app"
dist="$repo_root/target/dist"
mkdir -p "$dist"
echo "==> Figura Obscura $version (macOS)"

# --- 1. build ---------------------------------------------------------------
if $universal; then
    # A universal binary is what a customer expects from a paid Mac app: one
    # download that is native on both Apple silicon and Intel.
    echo "==> building for both architectures"
    for target in aarch64-apple-darwin x86_64-apple-darwin; do
        rustup target add "$target" >/dev/null 2>&1 || true
        ( cd "$repo_root" && cargo build --release --workspace --target "$target" )
    done
    bin_dir="$repo_root/target/universal"
    mkdir -p "$bin_dir"
    for exe in obscura obscura-gui; do
        lipo -create -output "$bin_dir/$exe" \
            "$repo_root/target/aarch64-apple-darwin/release/$exe" \
            "$repo_root/target/x86_64-apple-darwin/release/$exe"
    done
else
    ( cd "$repo_root" && cargo build --release --workspace )
    bin_dir="$repo_root/target/release"
fi

# --- 2. bundle --------------------------------------------------------------
echo "==> assembling Figura Obscura.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/bin"

install -m 0755 "$bin_dir/obscura-gui" "$app/Contents/MacOS/obscura-gui"
install -m 0755 "$bin_dir/obscura"     "$app/Contents/MacOS/obscura"

sed "s/__VERSION__/$version/g" "$repo_root/packaging/macos/Info.plist" \
    > "$app/Contents/Info.plist"
install -m 0644 "$repo_root/packaging/assets/figura-obscura.icns" \
    "$app/Contents/Resources/figura-obscura.icns"
install -m 0644 "$repo_root/packaging/common/THIRD-PARTY.md" \
    "$app/Contents/Resources/THIRD-PARTY.md"
install -m 0644 "$repo_root/LICENSE" "$app/Contents/Resources/LICENSE"

# ob-media searches Contents/Resources/bin — see ob-media::tools::bundle_dirs.
if [[ -n "$ffmpeg_dir" ]]; then
    for tool in ffmpeg ffprobe; do
        src="$ffmpeg_dir/$tool"
        [[ -f "$src" ]] || { echo "error: $src not found" >&2; exit 1; }
        "$repo_root/packaging/common/check-ffmpeg-licence.sh" "$src"
        install -m 0755 "$src" "$app/Contents/Resources/bin/$tool"
    done
else
    echo "warning: no --ffmpeg DIR; video will need ffmpeg on the user's PATH" >&2
fi

# --- 3. sign ----------------------------------------------------------------
if [[ -n "$sign_id" ]]; then
    echo "==> signing"
    entitlements="$repo_root/packaging/macos/entitlements.plist"
    # Inside out: nested code must be signed before the bundle that contains it,
    # or the outer signature is invalidated the moment the inner one is applied.
    while IFS= read -r -d '' item; do
        codesign --force --options runtime --timestamp \
            --entitlements "$entitlements" --sign "$sign_id" "$item"
    done < <(find "$app/Contents/Resources/bin" "$app/Contents/MacOS" -type f -perm -u+x -print0)
    codesign --force --options runtime --timestamp \
        --entitlements "$entitlements" --sign "$sign_id" "$app"
    codesign --verify --deep --strict --verbose=2 "$app"
else
    echo "warning: not signed — Gatekeeper will refuse this on any other Mac." >&2
fi

# --- 4. dmg -----------------------------------------------------------------
echo "==> building the disk image"
dmg="$dist/FiguraObscura-$version-macos.dmg"
staging="$repo_root/target/dmg"
rm -rf "$staging" "$dmg"
mkdir -p "$staging"
cp -R "$app" "$staging/"
# The drag-to-install convention; without it users run the app from the image.
ln -s /Applications "$staging/Applications"
cp "$repo_root/packaging/common/THIRD-PARTY.md" "$staging/"
cp "$repo_root/LICENSE" "$staging/"
hdiutil create -volname "Figura Obscura $version" -srcfolder "$staging" \
    -ov -format UDZO "$dmg"

[[ -n "$sign_id" ]] && codesign --force --sign "$sign_id" "$dmg"

# --- 5. notarize ------------------------------------------------------------
if [[ -n "$notarize_profile" ]]; then
    echo "==> notarizing (this takes a few minutes)"
    xcrun notarytool submit "$dmg" --keychain-profile "$notarize_profile" --wait
    # Staple so the app opens on a machine that is offline at first launch.
    xcrun stapler staple "$dmg"
    xcrun stapler validate "$dmg"
else
    cat >&2 <<'MSG'

warning: not notarized. Set up a keychain profile once:
  xcrun notarytool store-credentials figura-obscura \
      --apple-id you@example.com --team-id TEAMID --password <app-specific-password>
then re-run with --notarize-profile figura-obscura.
MSG
fi

echo
echo "==> $dmg ($(du -h "$dmg" | cut -f1))"
