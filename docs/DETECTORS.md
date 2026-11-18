# Detectors

Which models Obscura ships, what each one is good for, and the per-model
behaviour you need to know before trusting its output. The engineering that
feeds these models (multi-pass tiling, filtered downscaling, how several models
are combined) is in the main [README](../README.md); this file is about the
models themselves.

All five are third-party weights, downloaded on first run and never
redistributed here. `obscura models list` prints the same lineup with the
licence and the download state of each, and `obscura models show <id>` prints
every setting it exposes.

## The lineup

Two families, because photographic and drawn input need different detectors.
The default download set covers one of each, so the tool works on either kind of
input straight after `obscura setup`.

### Photographic (`Domain::RealLife`)

| id | Input | Size | Licence | Notes |
|---|---|---|---|---|
| `nudenet-320n` | 320px | 12 MB | Apache-2.0 | yolov8n. The default: fast, fine at ordinary photo resolutions, pair it with tiling for 4K. 18 categories. |
| `nudenet-640m` | 640px | 104 MB | Apache-2.0 | yolov8m. Roughly eight times the download and several times the inference cost, for better recall on small regions. |

`nudenet-640m` is the one registry entry whose **checksum is not yet pinned**
(`sha256: ""`), so it is verified by size and source only. Pin it before relying
on it in anything automated.

### Drawn (`Domain::Anime`)

| id | F1 | Published threshold | Notes |
|---|---|---|---|
| `anime-censor-v1-s` | 0.83 | 0.238 | yolov8s, 11.1M parameters, the default |
| `anime-censor-v1-n` | 0.80 | 0.278 | yolov8n, 3.01M, cheap enough to run on every frame |
| `anime-censor-v0.10-s` | 0.83 | 0.15 | Same architecture as v1.0_s but an earlier training run, so its mistakes are the least correlated with the others', which makes it the most informative second opinion |

All three are MIT, from
[`deepghs/anime_censor_detection`](https://huggingface.co/deepghs/anime_censor_detection),
and all run at a 640px input.

Each entry carries its **own** published F1-optimal threshold, and they differ a
lot (0.15 against 0.278). Using one model's threshold on another moves its
operating point, so `--set` overrides apply only where a companion model
declares the same key, leaving unset thresholds per-model. The same applies to
the GUI's cross-examination panel.

## Region size: the drawn models box less than you expect

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
fixes: lower `conf_threshold` for the first, more padding for the second. The
GUI's **Review** pass is the direct way to tell them apart, since it shows the
geometry before anything is written.

These models are indifferent to body shape in the way that matters here. They
are not looking for a breast silhouette at all, so an atypical or petite figure
does not push them off-distribution the way a shape-based detector would be
pushed. What did hurt them was scale, which tiling and filtered downscaling now
address. That reasoning comes from the models' class list and architecture:
**their actual recall across styles and figures has not been measured here.**

## What has not been measured

Detection quality has only been observed with `nudenet-320n`. The numbers in the
tables above are the publishers' own, carried over from their model cards, and
nothing in this repository reproduces them. How the other four behave across
styles, resolutions and subject matter is an open question, and the honest
advice is the same one the README gives: run `obscura process --dry-run --report`
over your own material and look at the files that came back with zero regions.
