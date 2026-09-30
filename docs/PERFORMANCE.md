# Performance

What was measured, what changed, and what was tried and rejected.

Everything here is **CPU execution provider only**. This repository has never
been able to validate a GPU build — see the caveat in [`../README.md`](../README.md) —
and nothing in this document is evidence about one. Where a change is expected
to behave differently on a GPU that is said explicitly and marked unverified.

## How to reproduce

```sh
cargo build --release -p ob-cli -p xtask
cargo xtask bench                 # or: cargo xtask bench <path-to-obscura>
```

`xtask bench` has two halves.

**`pipeline`** drives the real `obscura` binary over a generated corpus
(`target/bench-corpus/`, created on first run and then reused, so both sides of
a comparison see identical bytes). This is the number that matters: it includes
decode, inference, censoring and encode.

**`stages`** calls a few library functions directly. It exists because the
pipeline half cannot reach all of them — a detector finds nothing in
procedurally generated images, so the censor renderers never run during a
pipeline pass however the corpus is built. The stage half deliberately uses only
functions that existed *before* this work, so the same file can be dropped into
an older checkout and produce comparable numbers.

Machine for every figure below: Intel i7-10750H, **6 physical cores / 12
logical**, Linux, ONNX Runtime CPU provider, model `nudenet-320n` (YOLOv8n,
320px input, 18 classes).

## Results

End to end, against `b6a8a3a`. Best of five runs each.

| case | before | after | change |
| --- | ---: | ---: | ---: |
| 20 images, tiling off (one pass each) | 1.37 s | 0.62 s | **−55%** |
| 20 images, tiling auto (4K frames tile into a grid) | 3.77 s | 2.43 s | **−36%** |
| 10 s 1080p video, detect every 3rd frame | 6.88 s | 4.46 s | **−35%** |

Stage level, same two builds:

| stage | before | after | change |
| --- | ---: | ---: | ---: |
| letterbox 4K → 320, triangle | 128.99 ms | 5.86 ms | **22×** |
| letterbox 1080p → 320, triangle | 34.30 ms | 1.81 ms | **19×** |
| letterbox 320 tile → 320 (identity resize) | 5.37 ms | 0.30 ms | **18×** |
| censor solid fill, 1200×800 | 3.91 ms | 0.54 ms | 7.2× |
| censor pixelate, block 16 | 10.24 ms | 2.05 ms | 5.0× |
| censor pixelate, block 4 | 11.18 ms | 2.71 ms | 4.1× |
| censor solid fill + rounding 0.4 | 7.24 ms | 2.05 ms | 3.5× |
| censor blur, sigma 12 | 255.94 ms | 89.33 ms | 2.9× |

The stage multiples are much larger than the end-to-end ones because a run is
dominated by inference and by codec work, neither of which these touch. A 22×
improvement to something that was 4% of the run is worth about 4%.

### Output is unchanged

Every change here preserves pixel output exactly. That is checked three ways:

- `the_rewritten_letterbox_is_bit_identical_to_the_old_one` compares the new
  letterbox against a literal transcription of the old one, with `assert_eq!` on
  raw `f32`, across four kernels and five size regimes.
- `the_rewritten_fills_are_byte_for_byte_the_old_ones` and
  `restricting_rounding_to_the_corners_restores_exactly_the_same_pixels` do the
  same for the censor renderers over forty randomised rectangles.
- End to end: both binaries run over real source material (`demo/`), where the
  detector reports 2–12 regions per image and 1807 across the clip, and the
  written PNGs compare **byte for byte identical** under both `tiling=off` and
  `tiling=auto`.

The one output that does differ between runs is WebM, and it differs when the
*same* binary is run twice: libvpx with `-row-mt 1` is not reproducible. That is
the encoder, not this work.

## What actually made the difference

### The release profile was optimising for size (biggest single win)

`[profile.release]` set `opt-level = "z"` across the workspace. For a program
whose inner loops are pixel arithmetic and whose input is compressed, that is
the wrong default by a wide margin.

| configuration | images, tiling off | binary |
| --- | ---: | ---: |
| `opt-level = "z"` (as shipped) | 1.34 s | 27.6 MB |
| per-package `opt-level = 3` on our crates + `image` | 1.34 s | 29.2 MB |
| `opt-level = 3` workspace-wide | **0.67 s** | 30.9 MB |

The middle row is the interesting one. It recovered the *stage* numbers almost
fully (letterbox 4K: 16.3 ms → 6.7 ms) and moved the end-to-end figure not at
all, because the remaining cost was PNG decode — `png` and `miniz_oxide`, which
a per-package list of our own crates does not cover. Optimising the whole
workspace is what halved it.

Cost: 3.3 MB, a little under 12%, on a binary that already carries ONNX Runtime.
The rest of R9 is untouched: `strip` and `panic = "abort"` are what make the
binary opaque, and fat LTO with one codegen unit still discards the unreachable.

A note for anyone repeating this: fat LTO does **not** defeat per-package
`opt-level` overrides, which was the concern that made this a three-way
measurement rather than a one-line change.

### The letterbox allocated per output pixel

`contributions()` built a `Vec` of per-destination-pixel structs, each owning its
own weight `Vec` — roughly 640 allocations per axis, rebuilt on every pass — and
the result was resampled into an interleaved buffer that was then copied pixel
by pixel into the planar tensor the model actually wants.

It now builds flattened taps (three vectors per axis, once), writes straight
into the planar tensor at the letterbox offset, and takes a fast path when the
scale is 1 — which is the common case, because a tile is cropped at exactly the
model's input size and so was running the entire separable-filter machinery to
perform an identity resize.

The fast path deliberately excludes Lanczos3. At scale 1 its taps are not
exactly `(0, 1, 0)`: `sinc(1)` evaluates to about `-2.8e-8` in `f32` rather than
zero, so the filtered path there is very slightly not the identity, and taking
the shortcut would change its output. The other three kernels are exact.

Reading a sub-rectangle in place also removed `Frame::crop` from the tiled path
— one full tile copy per tile per frame.

### The detector held one session behind a mutex

`ob_job::run` processes images with `rayon::par_iter`, but every `detect()` went
through a single `Mutex<Session>`. A twelve-core machine decoded and censored
twelve ways and inferred one way.

Getting this right turned out to be almost entirely about thread budgeting, and
the first two attempts were **slower than the mutex they replaced**:

| attempt | images, tiling off | images, tiling auto |
| --- | ---: | ---: |
| baseline: 1 session, mutex | 1.37 s | 3.74 s |
| 4 sessions, ORT default threads each | 2.18 s | 5.14 s |
| 4 sessions, threads = *logical* cores ÷ sessions | 1.62 s | 4.02 s |
| 2 sessions, 2 intra-op threads each | **1.39 s** | **3.25 s** |

Two separate mistakes, both worth naming:

1. ONNX Runtime sizes **each** session's intra-op pool to the machine. Four
   sessions therefore asked for forty-eight compute threads on twelve logical
   cores, on top of the batch engine's own workers.
2. Budgeting from `rayon::current_num_threads()` — twelve — handed every session
   twice the threads it wanted, because this machine has six physical cores and
   hyperthreading. Inference is compute bound, so the second hyperthread on a
   core contends for the same execution units rather than adding throughput.
   Hence the `num_cpus` dependency: the physical count is the one that matters.

The final split leaves about a third of the machine *unbudgeted for inference*
on purpose. Sessions × intra-op threads = 4 on a 6-core box, because the other
two cores are decoding and encoding the next images at the same time, and
starving that half of the pipeline was what made the untiled case — which is
mostly decode — slower rather than faster.

`OBSCURA_SESSIONS` and `OBSCURA_INTRA_THREADS` override both. They exist because
the right split depends on the machine and on what else is running on it, and
the defaults were tuned on exactly one machine.

### The ETA lookup was quadratic

`Workload::work_for` was a linear scan over every item in the batch, called once
per finished file to move the progress bar. On the fifty-thousand-image batch
the activity log's capacity was chosen for, that is on the order of a billion
path comparisons over the run, on the thread drawing the window. It is indexed
now, and the list and its index are only reachable through one method that
maintains both.

### Smaller things

- `probe()` spawned one `ffprobe` per video, sequentially. It is parallel now;
  the win is latency to first estimate on a folder of clips.
- `run()` copied the input tensor before handing it to ORT — 1.2 MB per pass at
  320px, 4.9 MB at 640px, for a buffer the caller had just built and had no
  further use for.
- The censor renderers indexed the frame three times per pixel. They use row
  slices now, and `round_corners` visits only the four corner squares instead of
  scanning the whole rectangle to discover that the interior is unaffected.
- `save_image` cloned the frame's buffer. The batch path is finished with the
  frame at that point and now hands the encoder its buffer directly.
- Filtering detections allocated twice per frame (`select_all(..).copied().collect()`).

## Tried and rejected

### Batched tile inference

This was expected to be the single biggest win and **it is not a win at all on
the CPU provider**. A tiled 4K frame is thirteen passes; the graph declares a
symbolic batch axis, so all thirteen can go in one run instead of thirteen. The
machinery works — `a_batched_grid_reports_exactly_what_window_by_window_reports`
checks it against the real model — but it is consistently slower, monotonically
worse as the batch grows, at every session and thread count tried:

| batch size | 1 session | 2 sessions | 3 sessions |
| ---: | ---: | ---: | ---: |
| 1 | 3.75 s | 3.47 s | 3.38 s |
| 2 | 3.82 s | 3.56 s | 3.40 s |
| 4 | 3.84 s | 3.66 s | 3.65 s |
| 8 | 4.04 s | 3.77 s | 3.83 s |
| 13 | 4.09 s | 3.99 s | 3.76 s |

The reasoning that motivated it was GPU reasoning applied to a CPU. ORT's CPU
kernels already spread a single inference across the intra-op threads, so a
batched tensor buys no extra parallelism while costing cache locality and a much
larger transient allocation.

The code is kept, defaulted **off** (`MAX_BATCH = 1`), because the argument
still holds for a GPU, where per-launch overhead dominates and wide batches are
the normal way to feed one. That is **unverified** — there is no GPU here — so
it is not a default anyone gets by accident. `OBSCURA_MAX_BATCH=8` turns it on
for someone in a position to measure it.

The refactor it required was worth keeping regardless: `Detector::detect_regions`
is what removed the per-tile `Frame::crop`, and it is a better seam than
"crop, call, offset" repeated at each call site.

### Nested parallelism in the resampler

Splitting the filter passes across rayon looked free and was not: when the call
comes from a batch worker the pool is already fully committed, so the split adds
scheduling overhead and cache pressure to every other worker for no gain. It is
now conditional on `rayon::current_thread_index().is_none()` — that is, on the
caller *not* being a rayon worker — so the video pipeline, GUI previews and
single-file CLI runs get it and the batch engine does not.

## Not attempted

Costed, plausible, and deliberately left alone.

- **Overlapping video decode / inference / encode.** The video path is strictly
  serial: while inference runs, both ffmpeg children idle. Bounded channels
  between the three stages would overlap them — an estimated 1.3–2× on video, by
  far the largest remaining win. The cost is a few frames of buffering (24 MB
  per frame at 4K), new cancellation paths, and the guarantee that a cancelled
  encode leaves nothing playable behind gets harder to hold.
- **Frame buffer recycling.** `FfmpegSource::next_frame` allocates and zeroes a
  fresh buffer per frame — 24 MB at 4K. Only pays off with a pool that gets
  buffers back, which is the pipelining work above, so it is listed here rather
  than done half.
- **A faster blur.** `image`'s `fast_blur` (three box passes) is several times
  quicker than the true Gaussian and visually near-identical. Blur is by far the
  most expensive censor style — 89 ms for one region where a solid fill is 0.5 ms
  — so this is the biggest remaining censor win. It changes output, so it would
  ship as an opt-in `blur_quality` setting with `gaussian` as the default.
- **Quantised (INT8) model variants.** Typically 2–4× on CPU inference, which is
  what dominates a tiled run. It changes detections, so it belongs in the
  registry as its own entry with its own accuracy note, not as a switch on an
  existing one.
- **IO binding on GPU providers**, to stop round-tripping tensors through host
  memory per pass. Unevaluable here.
- **Adaptive `detect_every`** driven by inter-frame difference, rather than a
  fixed stride. A real win on static footage, but it changes which frames are
  inferred on, so it is an accuracy question before it is a speed one.
- **A detection cache** keyed by file content, model and settings, so re-running
  a folder after a censor-style change skips inference entirely. The largest win
  available for the actual tuning workflow; needs a cache format, an eviction
  policy, and a correctness story about stale entries.
- **SIMD in the resampler.** Worth revisiting now that `opt-level = 3` is doing
  the auto-vectorisation the old profile forbade — but from a profile, not from
  a guess.
