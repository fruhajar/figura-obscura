# Architecture

How Figura Obscura is put together and why. For building it see
[`HOST-BUILD.md`](HOST-BUILD.md), for shipping it [`RELEASING.md`](RELEASING.md),
and for what it costs to run [`PERFORMANCE.md`](PERFORMANCE.md).

## The shape of it

Nine crates in one workspace, two binaries (`obscura`, `obscura-gui`), and a
build-task crate that ships with neither.

```
              ┌──────────┐        ┌──────────┐
              │  ob-cli  │        │  ob-gui  │      binaries
              └────┬─────┘        └────┬─────┘
                   └────────┬──────────┘
                            ▼
                      ┌──────────┐
                      │  ob-job  │     batch engine, progress, cancellation
                      └────┬─────┘
          ┌──────────┬─────┴────┬──────────┐
          ▼          ▼          ▼          ▼
    ┌──────────┐┌─────────┐┌─────────┐┌──────────┐
    │ob-detect ││ob-censor││ ob-media││ ob-track │
    └────┬─────┘└────┬────┘└────┬────┘└────┬─────┘
         └───────────┴──────┬───┴──────────┘
                            ▼
                      ┌──────────┐      taxonomy, geometry, registry,
                      │ ob-core  │      settings, filters, profiles
                      └──────────┘

    ob-models ─── used by both binaries and by ob-detect's tests only
```

The graph has that shape because of four rules, each of which exists to keep one
capability from leaking everywhere:

- **`ob-core` performs no I/O and no inference.** It is types and policy —
  the taxonomy, `Frame`/`BBox`/`Detection`/`Region`, the model registry, filter
  rules, censor configuration. Everything else can depend on it freely because
  there is nothing in it to be careful about.
- **`ob-models` is the only crate that touches the network** (requirement R8).
  Models are fetched once, up front, verified by SHA-256, and cached. No
  processing path can reach a socket, because no processing crate depends on
  anything that has one.
- **`ob-detect` is the only crate that touches `ort`.** ONNX Runtime 2.0 is a
  release candidate with an unfrozen API; confining it to one crate is what
  makes that an acceptable risk.
- **`ob-media` is the only crate that shells out to ffmpeg.** v1 spawns the
  system `ffmpeg`/`ffprobe` rather than linking libav: it keeps the binary small
  and the licensing clean, and a child process is an easy seam to reason about.

## The pipeline

One path, used by everything:

```
frame ──▶ detect ──▶ filter ──▶ censor ──▶ frame
```

`ob_job::censor_frame` is that path. The batch run calls it, `--dry-run` calls
it, and the GUI's live preview calls it. That sharing is a safety property
rather than tidiness: a preview that disagreed with the batch would be worse
than no preview, because someone would tune against it and ship the difference.

The stages:

- **detect** — `ob_detect::Detector` turns a `Frame` into `Vec<Detection>` in
  original-image pixel coordinates, already mapped into the canonical taxonomy.
- **filter** — `ob_core::filter::FilterSet` decides which detections get
  censored, expressed purely in the taxonomy so it is model-independent. Rules
  are wildcard-capable over three axes (sex, part, state) with per-rule score
  floors.
- **censor** — `ob_censor::apply` renders solid fill, pixelate, blur or an image
  overlay over each selected box, with padding and corner rounding.

### Detection failure is a policy decision, not an error

When the detector fails, `OnDetectFailure` decides what happens, and the answer
is the same in the batch, the dry run and the preview: pass the frame through,
skip the file, or blank the frame entirely. `Blank` is the fail-closed choice —
obliterate rather than risk a leak. This is centralised in one function
specifically so the three callers cannot drift apart.

## Detection

### Letterboxing

YOLO-family models want a square input, so a frame is scaled to fit and padded.
`ob_detect::preprocess` records the scale and padding as a `Letterbox` so boxes
can be mapped back afterwards.

The resampler matters more than it looks. A 4K frame reaching a 320px input is
scaled by about 0.083, so a 40px feature becomes 3px — below the model's
smallest stride. Point sampling reads one source pixel per output pixel and
simply misses everything in between, which is how small subjects in
high-resolution sources used to vanish before the model ever saw them. The
default `Triangle` kernel is **support-scaled**: on minification the filter
radius widens by `1/scale`, so every source pixel contributes to some output
pixel. That property matters far more than the choice of kernel.

### Two wrappers, both `Detector`s

`Detector` is the only thing the rest of the program knows about. Two
implementations wrap others, so they compose — an ensemble of tiled detectors is
just the two nested:

- **`TiledDetector`** runs the model over the whole frame *and* over an
  overlapping grid of tiles cropped at native resolution, merging everything
  through one global NMS. Anti-aliased resampling stops small features
  disappearing; it cannot give the model back resolution it never had, and
  tiling is how that is recovered. In `Auto` mode it engages only when the
  whole-frame pass would downscale past `min_scale`.
- **`EnsembleDetector`** runs several models and combines their verdicts. With
  `min_votes == 1` (the default) it is a union — anything any model saw gets
  censored, which is the fail-safe direction. Above that it demands corroboration.
  **Votes are counted per category among the members that could actually cast
  one**, because members do not share a taxonomy: without that clamp, adding a
  nipple specialist to a general model would *delete* every category the
  specialist cannot see.

`Detector::detect_regions` takes several sub-windows of one frame and returns
their detections in frame coordinates. It exists so a tile grid is one call
rather than a crop-call-offset loop repeated at each call site, and so an
implementation can answer a whole grid in one pass where that helps. The default
implementation is the per-window loop, so every detector gets correct behaviour
for free.

### Which execution provider actually ran

A CUDA build on a machine with no usable driver silently runs on CPU, and the
only symptom is that the job is slow. So the distinction between *compiled in*
and *available* is first-class: `EpStatus` reports both, `--version` prints a
line per provider, and `Detector::execution_provider` reports what ONNX Runtime
actually returned rather than what was requested. Providers are tried one at a
time with `error_on_failure`, because the bulk registration call returns `Ok`
whether or not any GPU provider took.

## Media I/O

`FrameSource` and `FrameSink` are the seam — `next_frame` and
`put_frame`/`finish`. The real-time screen tool reimplements them (R10).

Video is a pair of ffmpeg child processes: one decoding to packed `rgb24` on
stdout, one taking censored `rgb24` on stdin. The original file is muxed in a
second time purely for its audio stream. Policy is re-encode video, stream-copy
audio.

Three things in here are not obvious and all three were bugs first:

- **The container picks the codec, not the user.** The output keeps the input's
  extension, and a WebM muxer rejects h.264 while writing the header — before
  reading a single frame — so every `.webm` failed with `Broken pipe` from a
  dead encoder. `Container` decides.
- **GIF cannot stream.** `palettegen` only emits its palette at end of input, so
  a single-pass graph holds the whole clip in memory; a long GIF walked the
  machine into swap. Frames go to a lossless `ffv1`/`bgr0` intermediate beside
  the output, and two palette passes read it back.
- **A `.gif` is two formats wearing one extension.** `classify_resolved` opens
  the file in that one case: an animated GIF is a video, a single-frame one stays
  an image and keeps working with no ffmpeg installed.

Child stderr is drained on a thread. This is not optional: a pipe nobody reads
fills at 64 KiB and blocks the writer, and an ffmpeg blocked writing stderr never
reads its stdin again — the encode would hang instead of failing.

## Threading and cancellation

```
CLI / GUI ──spawn──▶ worker thread ──▶ ob_job::run
                          │
                          ├── images:  rayon par_iter, one file per task
                          │             └── detect() leases an ONNX session
                          │                 from a small pool
                          └── videos:  sequential, one frame at a time
```

Images run in parallel because each is independent and small. Videos run one at
a time because a single clip already saturates the machine between ffmpeg and
inference, and because the tracker is stateful across frames.

Inference holds a **pool** of ONNX sessions rather than one. `Session::run` takes
`&mut self` and the engine shares one `&dyn Detector` across workers, so a
session must be guarded — and a single guarded session meant twelve cores
decoding twelve ways and inferring one way. The pool's size and each session's
intra-op thread count are chosen together, and deliberately do not consume the
whole machine: decode and encode are happening on the other workers at the same
time. [`PERFORMANCE.md`](PERFORMANCE.md) has the measurements, including the two
ways of getting this wrong that are both slower than the single mutex.

**Cancellation** is a `CancelToken` polled between files and between video
frames — never mid-file. That is what makes it safe: a file is either fully
processed or never begun. A cancelled or failed video kills the encoder rather
than letting it see a clean EOF, because on EOF ffmpeg would finalise and write
out a truncated file, and a half-censored clip that looks playable is the one
thing an abandoned job must not leave on disk. A cancelled run is not a failed
one: it is reported separately so the CLI exits zero and the GUI shows no error.

The GUI never blocks its UI thread. Batch runs, previews and input scans each
spawn a worker reporting over an `mpsc` channel that the app drains once per
frame. Previews are split in two — `preview_detect` (decode plus inference,
hundreds of milliseconds to seconds) and `preview_compose` (repaint boxes, a few
milliseconds) — so dragging a censor-style slider re-composes from cached
detections instead of re-running the model. Only a change that invalidates the
detections re-runs inference.

## Estimating a run

The naive estimate — elapsed ÷ files done × files remaining — says nothing until
the first file lands, which on a batch of 4K video is exactly when someone wants
to know, and treats a thumbnail and a feature film as equal.

So the batch is measured first, and measurement is split from costing for the
same reason the preview is: `probe` walks every file (a header read for images,
one `ffprobe` per video) and `cost` re-answers "how long?" instantly, as often
as a slider moves.

Two properties make it work. The pass count comes from `tiles_for_size` — the
planner the detector itself uses — so a 4K frame that tiles into twelve passes is
costed as twelve, and the estimate cannot drift from the detector's behaviour
because there is only one implementation. And the scan produces *relative*
weights; the absolute seconds-per-unit is calibrated from real runs and
persisted, so an error in the constants stretches every item equally and cancels
out of the ratio.

## Invariants worth not breaking

- **Fail-closed is a user-visible policy**, answered identically by the batch,
  the dry run and the preview.
- **Output keeps the input's extension**, which is what makes the container
  constrain the codec rather than the other way round.
- **No network at processing time.** Only `ob-models` has an HTTP client, and
  only `fetch` uses it. Models are verified by checksum before use.
- **The taxonomy is the vocabulary.** Filters, profiles and the UI are expressed
  in canonical categories, never in a model's native class indices. Label maps
  live in the registry, and `can_emit` is how a model declares what it is capable
  of reporting at all.
- **A partially written output file is never left behind.**
- **Pixel output is a contract.** Optimisation work is held to byte-for-byte
  identical output, checked against transcribed reference implementations and
  end to end against real source material. See [`PERFORMANCE.md`](PERFORMANCE.md).

## What dominates a run

Inference passes, and it is not close. The levers, roughly in order of effect:
ensemble size (every member sees every frame), tiling mode and `max_tiles` (a 4K
frame is one pass or thirteen), the model's input size, and for video
`detect_every`. Decode and encode matter more than expected on image batches of
compressed stills — enough that the release profile's optimisation level was
worth 2× on its own.
