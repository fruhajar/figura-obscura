# Figura Obscura

Figura Obscura finds explicit content in images and video and covers it up:
solid blocks, pixelation, blur or an overlay image, applied only to the regions
a detector actually found.

It runs entirely on your own machine. Nothing you process is uploaded, and
nothing is sent anywhere while it runs. The detection models are downloaded once,
before the first run, and after that the tool never opens a socket.

There are two front ends over the same engine: **`obscura-gui`**, a desktop app
with a live preview, and **`obscura`**, a command-line tool for batches and
scripts.

> **Status:** feature complete and packaged. Installers exist for Windows, Linux
> and macOS, and `cargo build --release`, `cargo test` (220 tests) and
> `cargo clippy --all-targets` are green on rustc 1.98.0.
>
> Two things are genuinely unverified, and both are worth knowing before you
> rely on the tool. **GPU builds have never run on real hardware**, so the
> releases are CPU builds. And **detection quality has only been observed with
> one model** (`nudenet-320n`), so how well the others do across art styles is
> an open question. Try it on your own material rather than taking the model
> cards' word for it.
>
> Where to read more: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the
> pieces fit together, [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for what a
> run costs and what has been measured, [`docs/HOST-BUILD.md`](docs/HOST-BUILD.md)
> for building it yourself, and [`docs/RELEASING.md`](docs/RELEASING.md) for
> cutting a release.

## Quickstart

Downloads are on the [Releases](../../releases) page. If it looks empty, the
current release is still a draft, so either [build an installer
yourself](#build-the-installer-yourself) or [build from
source](#build-from-source), which needs no installer at all.

### Install

**Windows.** Download `FiguraObscura-<version>-windows-x64-setup.exe` and run
it. It installs to Program Files, adds a Start menu entry, offers a tickbox to
put `obscura.exe` on your `PATH`, and downloads the detection models on its final
page. Nothing else to do.

The installer is unsigned, so SmartScreen will interrupt with "Windows protected
your PC". Choose *More info*, then *Run anyway*.

**Linux.** Download `FiguraObscura-<version>-linux-x86_64.tar.gz`, then:

```sh
tar -xzf FiguraObscura-*-linux-x86_64.tar.gz
cd FiguraObscura-*-linux-x86_64
./install.sh
```

That installs into `~/.local`, with no root and no package manager. It registers
the desktop entry and icons so the app appears in your menu, and downloads the
models. Useful variants:

```sh
./install.sh --prefix /usr/local   # system-wide (needs sudo)
./install.sh --no-models           # skip the download; fetch later with `obscura setup`
./install.sh --uninstall           # remove it again
```

Uninstalling leaves your models and settings alone on purpose, and tells you
where they are so you can delete them yourself if you want to.

If `~/.local/bin` is not on your `PATH`, the installer says so and prints the
line to add. There is also an `.AppImage` if you would rather have a single file:
`chmod +x` it and run it.

**macOS.** Open the `.dmg` and drag **Figura Obscura** to Applications. The app
is not notarised, so the first launch needs *right-click → Open*; a double-click
will be refused. The command-line tool lives inside the bundle at
`/Applications/Figura Obscura.app/Contents/MacOS/obscura`, so symlink it onto
your `PATH` if you want it.

### First run

The installers already do this. If you skipped it, or you built from source:

```sh
obscura setup
```

That is the one command a fresh machine needs. It downloads the recommended
models (about 56 MB), prints each file's SHA-256, and checks that ffmpeg can run.
Running it twice is harmless.

Video needs `ffmpeg` and `ffprobe` on your `PATH`. Images do not. Nothing is
bundled, so install them from your package manager or from ffmpeg.org; the app
tells you exactly where it looked when it cannot find them.

### Use it

```sh
obscura-gui                                # the desktop app
obscura process ./photos -o ./censored     # the same thing, batched
obscura --help
```

Other useful commands:

```sh
obscura models list                    # what is installed, and how big the rest are
obscura models show nudenet-320n       # every setting, with the text the GUI tooltips use
obscura models path                    # where the cache lives
obscura process ./in -o ./out --no-auto-fetch   # fail rather than download, in a script
```

`obscura process` downloads a missing model by default, and says so before it
does. Ctrl-C stops between files, so a cancelled run never leaves a half-written
video behind.

### Build from source

Needs Rust, plus ffmpeg on your `PATH` if you want video.

```sh
cargo build --release
./target/release/obscura setup
./target/release/obscura-gui
```

This is the shortest path to a working copy, but it leaves the binaries in
`target/`: nothing is on your `PATH` and there is no menu entry. Use an installer
for that.

### Build the installer yourself

One script per platform, each run on its own operating system, all wrapping the
same staged tree. See [Packaging](#packaging) for the commands. Windows also
needs [Inno Setup 6](https://jrsoftware.org/isinfo.php). The full procedure,
including how a tagged release builds all three platforms in CI, is in
[`docs/RELEASING.md`](docs/RELEASING.md).

## Architecture

A Cargo workspace of small crates. Everywhere, the shape of the work is the
same: `frame + settings → detections → composited frame`, with no I/O in the
middle. That is what lets the planned real-time screen tool (`ob-screen`) reuse
everything except the input and output ends.

| Crate | Responsibility |
|-------|----------------|
| `ob-core` | Canonical taxonomy, geometry and `Frame` types, model registry, declarative settings metadata (which drives the CLI, the GUI and its tooltips), filter rules, censor styles, profiles. No I/O, no inference. |
| `ob-detect` | `ort` inference, letterboxing, NMS, native to canonical label mapping, execution-provider selection with CPU fallback. |
| `ob-censor` | Region renderers: solid fill, pixelate, blur, overlay image; per-category overrides; padding and rounding. |
| `ob-track` | IoU tracker plus hysteresis, so coverage does not flicker between frames. |
| `ob-media` | Image decode and encode; video demux and encode with audio passthrough (ffmpeg); `FrameSource` and `FrameSink`. |
| `ob-job` | Input expansion, worker pool, progress, per-file error isolation, dry runs, fail-closed behaviour, the batch pipeline. |
| `ob-models` | The **only** crate that touches the network: one-time download, checksum verification, local cache. |
| `ob-cli` | The `obscura` binary. |
| `obscura-gui` | The desktop app (egui/eframe); tooltips come from `ob-core`. |
| `xtask` | Development tasks: `cargo xtask icons` regenerates the app icons, `cargo xtask bench` times the pipeline. Not shipped. |

## The desktop app

`obscura-gui` is what most people will use.

- **First run** offers the recommended models (about 56 MB) as a single action,
  with live progress. It can be skipped for an offline install.
- **Batch** takes dropped files and folders and shows what it will skip. Its
  preview can sit side by side with the untouched original, which is the only
  practical way to judge whether the padding is wide enough or a region was
  missed. It also tells you **how long the batch will take before you start it**,
  see [Estimating a run](#estimating-a-run).
- **Tuning** holds the model, the filter tree and the censor styles, each control
  carrying its tooltip from `ob-core`, **with the preview beside them**. It
  re-renders shortly after you stop adjusting, so a threshold or a padding
  fraction can be chosen by looking instead of by guessing. Changing a censor
  style, a category or the padding repaints the frame that was already analysed;
  only the model and the detection settings cost another inference pass. The
  panel's **Live** toggle turns that off in favour of **Refresh preview**.
  Profiles export to the same JSON that `obscura process --profile` reads.
- The preview shows the first file in the batch, or any **Sample…** you pick, so
  a representative frame can be tuned against without joining the queue. Videos
  preview their first frame.
- **Models** installs, re-downloads and removes model files, with per-model
  progress, a cancel button, and the licence and source of each.
- Settings are saved to `~/.config/figura-obscura/settings.json`, models to
  `~/.cache/figura-obscura/models` (`OBSCURA_MODEL_DIR` overrides that).

### Estimating a run

"Files done out of files total" is a poor guide to how much of a batch is left,
because files are not equal: one 4K video can outweigh a thousand thumbnails.
Counting files gives a bar that stalls and then leaps, and an ETA that only
appears after the first file has finished, which on a batch of long video is
exactly when you wanted it.

So the batch is measured first. Adding files starts a background scan that reads
each one's dimensions (an image header read, or one `ffprobe` per video) and
costs it in **inference passes**. The costing uses `tiles_for_size`, the same
planner the detector itself uses, so a 4K frame that will tile into twelve passes
is costed as twelve and the estimate cannot drift away from what the detector
really does.

That gives relative weights. The absolute seconds per pass belongs to your
machine, and it varies by two orders of magnitude between a CPU session and a
discrete GPU, so it is measured rather than assumed: seeded from a conservative
default, replaced by the real rate as soon as a run produces one, and saved to
settings so the next batch is estimated accurately before it starts. Until a run
has been timed on this machine, the app says so rather than presenting a guess as
a measurement.

Probing and costing are deliberately separate. Reading headers is I/O over every
file, while re-costing those measurements is arithmetic. So changing the tiling
mode, `--detect-every`, or the number of models updates the estimate instantly,
without touching the disk again. The live preview splits detection from
composition for the same reason.

Videos whose length cannot be read are excluded from the total and reported
separately ("plus 2 video(s) whose length could not be read") rather than having
a duration invented for them.

## Packaging

`packaging/` builds the artifacts that ship. One staging step defines what goes
in; each platform packager wraps that same tree, so the three formats cannot
drift apart.

```sh
packaging/stage.sh --ffmpeg /path/to/lgpl-ffmpeg/bin   # shared payload
packaging/linux/build.sh --ffmpeg ...                  # tarball + AppImage
packaging/windows/build.ps1 -FfmpegDir ...             # Inno Setup .exe
packaging/macos/build.sh --ffmpeg ... --universal      # .app + .dmg
```

Bundling ffmpeg is optional and the releases do not do it. If you do bundle one
it must be an **LGPL** build: the staging scripts refuse a GPL or a nonfree one,
because bundling those would either relicense the whole product or make it
undistributable. See
[`packaging/common/THIRD-PARTY.md`](packaging/common/THIRD-PARTY.md).

When ffmpeg is bundled, the app finds it next to the executable (`bin/`,
`Contents/Resources/bin`, `../lib/figura-obscura`) before falling back to `PATH`,
so the person running it never has to install anything. `OBSCURA_FFMPEG` and
`OBSCURA_FFPROBE` override the search.

## Detector coverage and limits

### Multi-pass detection

Detection is not a single downscaled pass. `ob-detect` runs the model over the
whole frame **and** over an overlapping grid of tiles at native resolution
(`ob-detect/src/tile.rs`), then merges every pass through one round of NMS.

This is what lets a small region be seen at all. A 4K frame letterboxed into a
640px input is scaled by about 0.167, which turns a 30px feature into 5px, at or
below YOLOv8's smallest stride of 8px. A tile crops that same region at native
resolution instead, so there is something left to detect.

`tiling` is `auto` by default: tiles are used only when the whole-frame pass
would downscale past 0.5, meaning the frame is more than twice the model input,
so small images cost exactly what they always did. Tiles start at the model's own
input size, with no downscaling at all, and are enlarged only if the grid would
otherwise exceed `tile_max` (12), which bounds the worst case. **Inference time
is linear in the number of tiles**: a tiled 4K frame is several passes, not one.
For video, raise `--detect-every` to compensate.

The whole-frame pass always runs, even when tiling, because it is the only pass
that sees a large region in one piece. Where a tile clips a region at a seam, the
truncated box and the neighbouring tile's complete box may both survive NMS.
For this tool that is the right way to fail: covering a region twice is
harmless, and not covering it is not.

| Setting | Default | Effect |
|---|---|---|
| `tiling` | `auto` | `off` restores the single-pass behaviour; `always` tiles regardless of size |
| `tile_overlap` | `0.25` | Shared strip between neighbours, so a region on a seam is not seen only in fragments |
| `tile_max` | `12` | Cap on tiles per frame; the grid coarsens rather than exceeding it |

### Downscale filtering

Whatever scaling remains is done with a properly filtered, support-scaled
resampler (`ob-detect/src/preprocess.rs`) rather than nearest-neighbour point
sampling. When shrinking an image, the filter's radius widens by `1/scale` so
that every source pixel contributes to some output pixel. That is the difference
between anti-aliased downscaling and simply missing whatever falls between
samples. The test `small_feature_survives_heavy_downscale` pins it down: a 4px
feature downscaled eight times disappears completely under nearest-neighbour and
survives under the filtered path.

`resample` accepts `triangle` (the default), `catmull-rom`, `lanczos3` and
`nearest`. **Triangle is the default deliberately.** With support scaling it
approaches a box average at large reductions, which is the standard choice for
feeding a detector, and it stays close in character to the bilinear resize these
models saw during *training*, which keeps inference preprocessing consistent with
how the model was taught. Lanczos3 holds slightly more bite on fine detail but
rings around hard edges, and a ringing halo is a false edge the detector can fire
on. Try it if detail matters more to you than false positives. Do not use
`nearest`; it exists for comparison only.

### The anime models box the areola, not the whole breast

The three `anime-censor-*` models share three native classes: `nipple_f`,
`penis`, `pussy`. Obscura maps `nipple_f` onto its own `FEMALE_BREAST_EXPOSED`
category, because that is the closest match in the canonical taxonomy, but the
box the model emits bounds the areola rather than the breast. The covering
rectangle is therefore areola-sized whatever figure it came from: on a small or
flat chest that is a fair share of the area you would want covered, and on a
large one it is a small fraction of it.

If the intent is to obscure the breast rather than the nipple, raise the region
padding (`censor.shape.padding`, default `0.10`, meaning 10% per side) well above
that default. The GUI's padding field is an unbounded spin box for exactly this
reason. There is no `--padding` flag on the command line, because padding is a
profile field: set it in the GUI or in a saved profile, and pass that with
`obscura process --profile`.

A missing box and a too-small box look similar in the output but need opposite
fixes: lower `conf_threshold` for the first, more padding for the second.

These models are indifferent to body shape in the way that matters here. They are
not looking for a breast silhouette at all, so an atypical or petite figure does
not push them off-distribution the way a shape-based detector would be pushed.
What did hurt them was scale, which tiling and filtered downscaling now address.
That reasoning comes from the models' class list and architecture: **their actual
recall across styles and figures has not been measured here.**

### Cross-examining several models

Three anime detectors share the taxonomy above, so where they disagree is where
any one of them is unsure:

| id | F1 | Published threshold | Notes |
|---|---|---|---|
| `anime-censor-v1-s` | 0.83 | 0.238 | yolov8s, 11.1M parameters, the default |
| `anime-censor-v1-n` | 0.80 | 0.278 | yolov8n, 3.01M, cheap enough to run on every frame |
| `anime-censor-v0.10-s` | 0.83 | 0.15 | Same architecture as v1.0_s but an earlier training run, so its mistakes are the least correlated with the others', which makes it the most informative second opinion |

Each entry carries its **own** published F1-optimal threshold, and they differ a
lot (0.15 against 0.278). Using one model's threshold on another moves its
operating point, so `--set` overrides apply only where a companion model declares
the same key, leaving unset thresholds per-model.

```sh
# Union (the default): cover anything any model saw, which maximises recall.
obscura process ./in -o ./out --model anime-censor-v1-s \
    --also-model anime-censor-v0.10-s

# Consensus: cover only what two models independently found.
obscura process ./in -o ./out --model anime-censor-v1-s \
    --also-model anime-censor-v0.10-s --also-model anime-censor-v1-n \
    --min-votes 2
```

Union is the default because it fails in the safer direction for this kind of
tool: a false positive costs a needlessly obscured patch, while a false negative
costs the very thing the tool exists to prevent. `--min-votes 2` is the opposite
trade, useful for auditing how much of a model's output is corroborated and risky
as a standing policy. One model firing twice on the same spot does not count as
agreement.

Votes are counted **per category, among the models that could cast one**. The
anime entries have 3 classes and NudeNet has 18, so mixing the two families under
`--min-votes 2` would otherwise delete every category outside the smaller
taxonomy: a model would veto regions it is structurally unable to see. Where both
models do cover a category, consensus applies normally.

The GUI offers the same thing under **Tuning → Cross-examination**: tick the
models to run alongside the primary one and set how many must agree. The preview
runs the same ensemble as the batch, so the effect of adding a model or raising a
threshold is visible before you commit to a run.

### Seeing what fires

`obscura process --dry-run` runs full detection and reports how many regions it
found per file, writing nothing.

**Stills** are detected whole. **Videos** are sampled: twelve frames by default,
spread evenly across the clip and taken from the midpoint of each slice, so the
sample avoids the fade in and fade out that bracket most footage. Each sample is
reached by seeking rather than by decoding everything up to it, so dry-running a
folder of long clips takes seconds. `--dry-run-frames N` buys a more thorough
audit for proportionally more time.

```sh
obscura process ./clips -o /tmp/x --dry-run --report audit.csv
obscura process ./clips -o /tmp/x --dry-run --dry-run-frames 40   # more thorough
```

The per-file number for a video is the total across its sampled frames, so read
it as "did this fire, and roughly how much", not as a frame-accurate count. Zero
still means what it should: the detector saw nothing anywhere it looked.

## Licence

Figura Obscura is released under the **PolyForm Noncommercial License 1.0.0**
(`LICENSE`). The source may be read, forked and modified; use is limited to
noncommercial purposes, and the commercial rights stay with the copyright holder.
It is a source-available licence rather than an open-source one. GitHub's licence
picker only offers OSI-approved licences, so the file is committed directly
instead of generated, which changes nothing about its effect.

Three things it does **not** cover, each licensed separately:

- **Detector models** are downloaded at runtime and are not part of this
  repository. Each carries its own licence, recorded in the model registry and
  printed by `obscura models list`.
- **FFmpeg**, when bundled, stays under its own LGPL or GPL terms. Obscura spawns
  it as a child process and never links libav, so that copyleft does not reach
  this source. See `packaging/common/THIRD-PARTY.md`.
- **Rust dependencies** keep their own (permissive) licences.
