# Figura Obscura

Offline, local batch tool (CLI + GUI) that selectively censors lewd regions in
images and videos. No network is used at processing time; detector models are
downloaded once, up front.

> **Status:** feature-complete, packaged, and building. Every crate is
> implemented, the desktop app has been rebuilt around a first-run setup flow,
> in-app model downloads and a proper theme, and installers exist for Windows,
> Linux and macOS. `cargo build --release`, `cargo test` (130 tests) and
> `cargo clippy --all-targets` are green on rustc 1.98.0.
>
> What is **not** yet validated is anything needing real hardware, a display, or
> real weights: GPU execution providers, inference against an actual `.onnx`,
> and the app rendered on screen (the build container has no display server, so
> the interface is exercised headlessly instead — every page is laid out under
> test, which catches panics and id clashes but not appearance). No model host
> is reachable from this container either, so model checksums stay unpinned —
> never pin a digest computed here.
>
> To ship: [`RELEASING.md`](RELEASING.md). To build on a host:
> [`HOST-BUILD.md`](HOST-BUILD.md).

## Architecture

A Cargo workspace of small crates. The invariant everywhere is a pure
`frame + settings → detections → composited frame`, so the planned real-time
screen tool (`ob-screen`) can reuse everything but the I/O ends.

| Crate | Responsibility |
|-------|----------------|
| `ob-core` | Canonical taxonomy, geometry/Frame types, model registry, declarative settings metadata (drives CLI+GUI+tooltips), filter rules, censor styles, profiles. No I/O, no inference. |
| `ob-detect` | `ort` inference, letterbox, NMS, native→canonical label mapping, execution-provider selection with CPU fallback. |
| `ob-censor` | Box renderers: solid fill, pixelate, blur, preset-image overlay; per-part overrides; padding/rounding. |
| `ob-track` | IoU tracker + hysteresis to stop censor flicker across frames. |
| `ob-media` | Image decode/encode; video demux/encode + audio passthrough (ffmpeg); `FrameSource`/`FrameSink`. |
| `ob-job` | Input expansion, worker pool, progress, per-file error isolation, dry-run, fail-closed, the batch pipeline. |
| `ob-models` | The **only** networked crate: one-time download, checksum verify, local cache. |
| `ob-cli` | `obscura` binary. |
| `obscura-gui` | `obscura-gui` desktop app (egui/eframe); tooltips sourced from `ob-core`. |
| `xtask` | Build-time asset generation (`cargo xtask icons`). Not shipped. |

## Quick start (on a host with Rust + ffmpeg)

```sh
cargo build --release
./target/release/obscura setup                               # download the default models
./target/release/obscura process ./photos -o ./censored
./target/release/obscura-gui                                 # the desktop app
```

`obscura setup` is the one command a fresh machine needs: it downloads the recommended
models, prints each file's SHA-256 (paste those into the registry to pin them),
and checks that ffmpeg is runnable. It is idempotent, and it is what the
installers run after copying files.

Other useful commands:

```sh
obscura models list                     # what is installed, and how big the rest are
obscura models show nudenet-320n        # every setting, with the same text the GUI tooltips use
obscura models path                     # where the cache lives
obscura process ./in -o ./out --no-auto-fetch   # fail rather than download in a script
```

`obscura process` downloads a missing model by default and says so first. Ctrl-C stops
it between files, so a cancelled run never leaves a truncated video behind.

## The desktop app

`obscura-gui` is the product most users will see.

- **First run** offers the recommended models (~56 MB) as a single action, with
  live progress, and can be skipped for an offline install.
- **Batch** takes dropped files and folders, shows what will be skipped, and
  gives a preview with an optional side-by-side of the uncensored source — the
  only way to judge whether the padding is large enough or a region was missed.
  It also says **how long the batch will take, before you start it** — see
  [Estimating a run](#estimating-a-run).
- **Tuning** is the model, filter tree and censor styles, each control carrying
  its `ob-core` tooltip, **with the preview beside them**. It re-renders shortly
  after you stop adjusting, so a threshold or a padding fraction is chosen by
  looking rather than by guessing. Changing a censor style, a category or the
  padding re-paints the frame that was already analysed — only the model and the
  detection settings cost another inference pass. Turn it off with the panel's
  **Live** toggle and drive it from **Refresh preview** instead. Profiles export
  to the same JSON `obscura process --profile` reads.
- The preview renders the first file in the batch, or any **Sample…** you pick —
  a representative frame can be tuned against without joining the queue. Videos
  preview their first frame.
- **Models** installs, re-downloads and removes weights, with per-model progress,
  cancel, and the licence and source of each.
- Settings persist to `~/.config/figura-obscura/settings.json`; models to
  `~/.cache/figura-obscura/models` (`SB_MODEL_DIR` overrides).

### Estimating a run

"Files done / files total" is a poor guide to how long a batch has left, because
files are not equal: one 4K video can outweigh a thousand thumbnails. Counting
them gives a bar that stalls, then leaps, and an ETA that only appears after the
first file — which on a batch of long video is exactly when you wanted it.

So the batch is measured first. Adding files kicks off a background scan that
reads each one's dimensions (an image header read; one `ffprobe` per video) and
costs it in **inference passes** — using `tiles_for_size`, the same planner the
detector itself uses, so a 4K frame that will tile into twelve passes is costed
as twelve and the estimate cannot drift from the detector's behaviour.

That gives *relative* weights. The absolute seconds-per-pass is the machine's,
and varies by two orders of magnitude between a CPU session and a discrete GPU,
so it is measured rather than assumed: seeded from a conservative default,
replaced by the real rate as soon as the run produces one, and saved to settings
so the next batch is estimated accurately before it starts. Until a run has been
timed on this machine the app says so rather than presenting the guess as a
measurement.

Probing and costing are deliberately separate. Reading headers is I/O over every
file; re-costing those measurements is arithmetic. So changing the tiling mode,
`--detect-every`, or the number of models updates the estimate instantly,
without touching the disk again — the same split the live preview makes between
detection and composition.

Videos whose length cannot be read are excluded from the total and reported
separately ("plus 2 video(s) whose length could not be read"), rather than
having a duration invented for them.

## Packaging

`packaging/` builds the shipping artifacts. One staging step defines what ships;
each platform packager wraps the same tree.

```sh
packaging/stage.sh --ffmpeg /path/to/lgpl-ffmpeg/bin   # shared payload
packaging/linux/build.sh --ffmpeg ...                  # tarball + AppImage
packaging/windows/build.ps1 -FfmpegDir ...             # Inno Setup .exe
packaging/macos/build.sh --ffmpeg ... --universal      # .app + .dmg
packaging/itch/push.sh --user YOU --dry-run            # publish plan
```

Bundled ffmpeg must be an **LGPL** build; the staging scripts refuse a GPL or
nonfree one, because bundling those would relicense or block the whole product.
See [`packaging/common/THIRD-PARTY.md`](packaging/common/THIRD-PARTY.md).

The app finds its bundled ffmpeg next to the executable (`bin/`,
`Contents/Resources/bin`, `../lib/figura-obscura`) before falling back to `PATH`, so
a customer never has to install it. `SB_FFMPEG`/`SB_FFPROBE` override.

## Detector coverage & limits

### Multi-pass detection

Detection is no longer a single downscaled pass. `ob-detect` runs the model over
the whole frame **and** over an overlapping grid of native-resolution tiles
(`ob-detect/src/tile.rs`), merging every pass through one NMS. This is what lets
a small region be seen at all: a 4K frame letterboxed into a 640px input is
scaled by ~0.167, turning a 30px feature into 5px — at or below YOLOv8's
smallest stride of 8px. A tile crops that region at native resolution instead.

`tiling` is `auto` by default: tiles are used only when the whole-frame pass
would downscale past 0.5 (i.e. the frame is more than twice the model input), so
small images cost exactly what they did before. Tiles start at the model's own
input size — no downscale at all — and are enlarged only if the grid would
exceed `tile_max` (12), which bounds the worst-case cost. **Inference time is
linear in tile count**: a tiled 4K frame is several passes, not one. For video,
raise `--detect-every` to compensate.

The whole-frame pass always runs, even when tiling: it is the only pass that
sees large regions in one piece. Where a tile clips a region at a seam, the
truncated box and the neighbouring tile's full box may both survive NMS — for
censoring, covering a region twice is harmless and covering it not at all is not.

| Setting | Default | Effect |
|---|---|---|
| `tiling` | `auto` | `off` restores the single-pass behaviour; `always` tiles regardless of size |
| `tile_overlap` | `0.25` | Shared strip between neighbours — stops a seam region being seen only in fragments |
| `tile_max` | `12` | Cap on tiles per frame; the grid coarsens rather than exceeding it |

### Downscale filtering

Whatever scaling remains is now done with a properly filtered, support-scaled
resampler (`ob-detect/src/preprocess.rs`) rather than nearest-neighbour point
sampling. When minifying, the filter's radius widens by `1/scale` so every
source pixel contributes to some output pixel — the difference between
anti-aliased downscaling and simply missing whatever falls between samples. The
unit test `small_feature_survives_heavy_downscale` pins the regression: a 4px
feature downscaled 8× disappears completely under nearest and survives under the
filtered path.

`resample` accepts `triangle` (default), `catmull-rom`, `lanczos3` and
`nearest`. **Triangle is the default deliberately.** With support scaling it
approaches a box/area average at large minification factors — the standard
choice for feeding a detector — while staying close in character to the bilinear
resize these models saw during *training*, which keeps inference preprocessing
consistent with training preprocessing. Lanczos3 holds slightly more acutance on
fine detail but rings around hard edges, and a ringing halo is a false edge the
detector can fire on. Try it if detail matters more than false positives; do not
use `nearest`, which exists for comparison only.

### `anime-censor-*` detects nipples, not breasts

The anime models' three native classes are `nipple_f`, `penis`, `pussy`. Obscura maps
`nipple_f` onto `FEMALE_BREAST_EXPOSED` because that is the closest canonical
category, but the box the model emits bounds the *areola*, not the breast — so
the censor rectangle is areola-sized regardless of the figure it came from. On a
small or flat chest that box is a fair share of the censorable area; on a large
one it covers a small fraction of it.

If the goal is to obscure the breast rather than the nipple, raise the region
padding (`censor.shape.padding`, default `0.10` = 10% per side) well above that
default — the GUI's padding field is an unbounded spin box for exactly this
reason. There is no `--padding` CLI flag: it is a profile field, so set it in the
GUI or a saved profile and pass that with `obscura process --profile`.

A missing box and a too-small box look similar in the output but need opposite
fixes — lower `conf_threshold` for the first, more padding for the second.

These models are body-shape agnostic in the sense that matters: they are not
looking for a breast silhouette at all, so an atypical or petite figure does not
put them off-distribution the way a shape-based detector would be. What hurt
them was scale, which tiling and filtered downscaling now address. Note this
reasoning is from the models' class list and architecture — **their actual
recall across styles and figures is untested here**, because no weights are
downloadable in this container.

### Cross-examining several models

Three anime detectors share the taxonomy above, so where they disagree is where
a single model is unsure:

| id | F1 | Published threshold | Notes |
|---|---|---|---|
| `anime-censor-v1-s` | 0.83 | 0.238 | yolov8s, 11.1M params — the default |
| `anime-censor-v1-n` | 0.80 | 0.278 | yolov8n, 3.01M — cheap enough to run on every frame |
| `anime-censor-v0.10-s` | 0.83 | 0.15 | Same architecture as v1.0_s but an earlier training run, so its mistakes are the least correlated — the most informative second opinion |

Each entry carries its **own** published F1-optimal threshold; they differ a lot
(0.15 vs 0.278) and using one model's threshold on another moves its operating
point. `--set` overrides are applied only where a companion declares the same
key, so unset thresholds stay per-model.

```sh
# Union (default): censor anything any model saw — maximises recall.
obscura process ./in -o ./out --model anime-censor-v1-s \
    --also-model anime-censor-v0.10-s

# Consensus: censor only what two models independently found.
obscura process ./in -o ./out --model anime-censor-v1-s \
    --also-model anime-censor-v0.10-s --also-model anime-censor-v1-n \
    --min-votes 2
```

Union is the default because it is the fail-safe direction for a censoring tool:
a false positive costs a needlessly obscured patch, a false negative costs the
thing the tool exists to prevent. `--min-votes 2` is the opposite trade — useful
for auditing how much of a model's output is corroborated, risky as a production
policy. One model firing twice on the same spot does not count as agreement.

Votes are counted **per category, among the models that could cast one**. The
anime entries are 3-class and NudeNet is 18-class, so mixing the two families
under `--min-votes 2` would otherwise delete every category outside the smaller
taxonomy — a model would veto regions it is structurally unable to see. Where
both models do cover a category, consensus applies normally.

The GUI exposes the same thing under **Tuning → Cross-examination**: tick the
models to run alongside the primary and set how many must agree. The preview
runs the same ensemble as the batch, so the effect of adding a model or
raising the threshold is visible before committing to a run.

### Seeing what fires

`obscura process --dry-run` over **stills** performs full detection and reports the
region count, writing nothing. For *videos* `--dry-run` only checks that the file
opens — it skips detection and always reports 0 regions — so pull representative
frames out first (`ffmpeg -i clip.mp4 -vf fps=1 f%04d.png`) and dry-run those.

## Licence

Figura Obscura is released under the **PolyForm Noncommercial License 1.0.0**
(`LICENSE`). The source may be read, forked and modified; use is limited to
noncommercial purposes, and the commercial rights stay with the copyright
holder. It is a source-available licence, not an open-source one — GitHub's
licence *picker* only offers OSI-approved licences, so the file is committed
directly rather than generated, which changes nothing about its effect.

Three things it does **not** cover, each licensed separately:

- **Detector weights** are downloaded at runtime and are not part of this
  repository. Each carries its own licence, recorded in the model registry and
  printed by `obscura models list`.
- **FFmpeg**, when bundled, stays under its own LGPL or GPL terms. Obscura
  spawns it as a child process and never links libav, so that copyleft does not
  reach this source — see `packaging/common/THIRD-PARTY.md`.
- **Rust dependencies** keep their own (permissive) licences.

Design decisions and rationale live in the approved plan
(`~/.claude/plans/twinkling-foraging-puppy.md`).
