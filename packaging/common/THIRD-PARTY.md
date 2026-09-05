# Third-party components shipped with Figura Obscura

Figura Obscura is distributed with the components below. Each keeps its own
licence; this file is included in every installer and archive so those terms
travel with the binaries.

## Bundled binaries

### FFmpeg — <https://ffmpeg.org>

Used for video decoding and encoding (`ob-media`).

**The key fact: Figura Obscura spawns `ffmpeg`/`ffprobe` as child processes and
never links libav.** They are separate programs exchanging bytes over pipes, so
FFmpeg's copyleft does not reach Figura Obscura's own source. What it can create is
an obligation attached to the *FFmpeg binary you distribute*. Linking libav
directly (via `ffmpeg-sys`, `gstreamer`, and friends) would be a different
situation entirely and is deliberately avoided.

That leaves four supportable options:

| Option | Obligation | Trade-off |
|---|---|---|
| **Don't bundle** — use the user's ffmpeg | **None at all** | User must install ffmpeg for video; images work regardless |
| **Bundle LGPL-2.1** *(default)* | Ship the licence text; offer the corresponding source; keep it replaceable | Needs an LGPL build, which is less commonly prebuilt |
| **Bundle GPL** (`--allow-gpl`) | Ship the licence text; offer the corresponding source for that exact build | Easiest binaries to obtain; strongest source obligation |
| **Bundle `--enable-nonfree`** | **Not redistributable by anyone** | Never do this |

`check-ffmpeg-licence.sh` enforces this: nonfree is always refused, GPL requires
an explicit `--allow-gpl`, LGPL passes. Omitting `--ffmpeg` from the staging step
selects the first row.

Whichever is chosen, Obscura keeps FFmpeg **separately replaceable** — a user can
substitute their own build, and `OBSCURA_FFMPEG`/`OBSCURA_FFPROBE` exist to make that
explicit. That is an LGPL requirement and good practice besides.

If you bundle anything, put the matching FFmpeg source tarball (or a link to the
exact upstream tag) next to the release on itch.io, and record which tag it was
**at build time**, while you still know.

> Separately from copyright: H.264 encoding has patent-licensing considerations
> for commercial products, independent of which encoder you use. It is widely
> ignored at small scale, but it is a distinct question from the one above and
> worth a look if volume grows.

### ONNX Runtime — <https://onnxruntime.ai> (MIT)

Model inference. The CPU build is **statically linked** into `obscura` and `obscura-gui`,
so nothing extra ships for the default build. GPU builds additionally emit
`onnxruntime_providers_*` shared libraries, which the staging script copies when
present.

## Detector models (downloaded, not bundled)

Weights are fetched on first run, not shipped in the installer — they are large,
they update independently, and the licences below travel with them.

| Model | Licence | Source |
|---|---|---|
| NudeNet 320n / 640m | Apache-2.0 | <https://github.com/notAI-tech/NudeNet> |
| deepghs `anime_censor_detection` (v1.0_s, v1.0_n, v0.10_s) | MIT | <https://huggingface.co/deepghs/anime_censor_detection> |

Both licences permit commercial use and redistribution with attribution, which
is what the About page provides.

## Rust dependencies

The build's full dependency tree and its licences can be regenerated with:

```sh
cargo install cargo-about && cargo about generate about.hbs
```

egui/eframe (MIT OR Apache-2.0) and the wider crates.io tree are permissive; no
copyleft crate is in the dependency graph.
