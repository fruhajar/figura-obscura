#!/usr/bin/env bash
# Check the licence of an FFmpeg build before bundling it.
#
# Figura Obscura spawns ffmpeg as a *child process* and never links libav, so the
# two are separate programs and the copyleft does not reach Obscura's own code. What
# bundling does create is an obligation for the ffmpeg binary itself: ship its
# licence text and offer the corresponding source for that exact build.
#
#   nonfree  -> refused. Cannot be redistributed at all, by anyone.
#   GPL      -> allowed with --allow-gpl, and only then, because it carries the
#               strongest source-offer obligation and is easy to pick up by
#               accident: nearly every convenient prebuilt binary is GPL.
#   LGPL     -> the default and the recommendation: lightest obligations.
#
# Not bundling at all is also fully supported — Obscura finds a user-installed ffmpeg
# on PATH — and carries no obligations whatsoever. See THIRD-PARTY.md.
set -euo pipefail

allow_gpl=false
args=()
for arg in "$@"; do
    case "$arg" in
        --allow-gpl) allow_gpl=true ;;
        *) args+=("$arg") ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"

ffmpeg_bin="${1:?usage: check-ffmpeg-licence.sh [--allow-gpl] /path/to/ffmpeg}"

if [[ ! -x "$ffmpeg_bin" ]]; then
    echo "error: $ffmpeg_bin is not an executable" >&2
    exit 1
fi

banner="$("$ffmpeg_bin" -hide_banner -version 2>&1 || true)"
config="$(printf '%s' "$banner" | grep -o -- '--enable-[a-z0-9-]*' || true)"

if printf '%s' "$config" | grep -qx -- '--enable-nonfree'; then
    echo "error: this FFmpeg is a --enable-nonfree build and cannot be redistributed." >&2
    exit 1
fi

if printf '%s' "$config" | grep -qx -- '--enable-gpl'; then
    if ! $allow_gpl; then
        cat >&2 <<'MSG'
error: this FFmpeg is a --enable-gpl build.

Figura Obscura spawns ffmpeg as a separate process, so this does NOT relicense
Figura Obscura's own code. It does mean that if you ship this binary you must
offer the corresponding FFmpeg source for this exact build, alongside the
release on itch.io.

Three ways forward:
  1. Use an LGPL build (recommended, lightest obligations):
       Windows  https://github.com/BtbN/FFmpeg-Builds  (pick an *lgpl* asset)
       macOS    ./configure --disable-gpl --disable-nonfree
       Linux    distro ffmpeg is usually GPL; build LGPL yourself
  2. Ship this GPL build deliberately: re-run with --allow-gpl and publish the
     matching source tarball.
  3. Do not bundle ffmpeg at all: omit --ffmpeg from the staging step. Obscura finds
     a user-installed ffmpeg on PATH and tells the user how to install one.
MSG
        exit 1
    fi
    version="$(printf '%s' "$banner" | head -n1)"
    cat >&2 <<MSG
warning: bundling a GPL FFmpeg ($version).

Figura Obscura's own code is unaffected (separate process, no libav linkage), but
you must publish the corresponding FFmpeg source for this exact build next to
the release. Record which upstream tag it came from now, while you know.
MSG
    exit 0
fi

version="$(printf '%s' "$banner" | head -n1)"
echo "ok: LGPL-compatible FFmpeg — $version"
