# Host build & finish-out guide

**Status: the workspace builds and its tests pass.** On rustc 1.98.0 in a
Debian container, `cargo build`, `cargo build --release` and `cargo test` (330
tests) are green, and `cargo clippy --all-targets` reports no errors (six style
warnings, no errors). What is still unvalidated is anything that needs real
hardware: the GPU execution providers have never run on a real card. See §3 for
how to tell whether a GPU build is actually using one.

## 1. Prerequisites (CachyOS / Arch)

```sh
sudo pacman -S --needed rustup ffmpeg
rustup default stable          # 1.98.0 or newer is known-good
```

`ffmpeg` and `ffprobe` have to be on `PATH` at **runtime**. `ob-media` shells
out to them for video I/O, so they aren't a compile-time dependency.

### GPU packages

Pick the stack that matches the card:

```sh
# NVIDIA  (ONNX Runtime 1.22 needs CUDA 12.x + cuDNN 9.x; see §3)
sudo pacman -S --needed cuda cudnn
# AMD     (the Vulkan ICD backs the webgpu provider recommended in §3)
sudo pacman -S --needed vulkan-radeon vulkan-icd-loader
```

## 2. First compile and test (CPU, and do this first)

Always confirm the CPU path before adding a GPU provider, so a GPU failure is
unambiguous:

```sh
cargo build
cargo test
cargo clippy --all-targets
```

The build downloads a prebuilt ONNX Runtime **1.22.0** (the `ort` crate's
default `download-binaries` feature), so the *build* needs network access even
though the finished tool runs fully offline.

## 3. Build with GPU support

GPU execution providers are opt-in features on `ob-detect`, so the default
binary stays lean. **Which flag to use depends on your card**, and the choice
matters more than it looks. See the ROCm warning below.

### NVIDIA: use `cuda`

```sh
cargo build --release --features ob-detect/cuda
```

Verified: this pulls ONNX Runtime's real `cu12` prebuilt binary. The workspace-root
form above and the package-scoped form (`cargo build -p ob-cli --features
ob-detect/cuda`) both propagate correctly through `ort` to `ort-sys`.

Because the dist is `cu12`, the host needs **CUDA 12.x + cuDNN 9.x**. Arch tracks
CUDA closely and may already be on 13.x, which these binaries will not load. If
so, install the CUDA 12 compat packages or point `ort` at your own ONNX Runtime
build:

```sh
export ORT_LIB_LOCATION=/path/to/onnxruntime/build/Linux/Release
```

### AMD: do NOT use `rocm`, use `webgpu`

```sh
cargo build --release --features ob-detect/webgpu
```

**`--features ob-detect/rocm` is a trap on this stack.** ONNX Runtime publishes
no ROCm prebuilt for `x86_64-unknown-linux-gnu` (the available feature sets are
`none`, `cu12`, `train`, `train,cu12`, `wgpu`). When the requested set is
missing, `ort-sys` prints one line and **downloads the CPU-only build
instead**, and the compile still succeeds with exit 0. The result is a binary
that looks like a GPU build, registers no ROCm provider at runtime, and quietly
runs on CPU.
This was confirmed here: a `rocm` build completed cleanly while fetching no ROCm
binary at all.

`webgpu` is the vendor-neutral path and *does* have a real prebuilt (`wgpu`), so
it works without compiling ONNX Runtime yourself. If you specifically need the
ROCm EP, build ONNX Runtime from source with ROCm and set `ORT_LIB_LOCATION`
before using `--features ob-detect/rocm`.

### Confirm the GPU is actually being used

Do this rather than assuming. CPU is always registered **last** as a fallback
(see `execution_provider_dispatches` and the `cpu_is_always_last_ep` test), so a
misconfiguration degrades quietly instead of erroring.

**Ask the binary first.** Both binaries report their execution providers, so
this needs no model, no job and no GPU monitor:

```sh
obscura --version             # or: obscura-gui --version, or the GUI's About page
```

```
obscura 1.0.0 (644dd82)

execution providers (in preference order):
  CUDAExecutionProvider (ready)
  CPUExecutionProvider (ready)
```

Three outcomes, and they mean different things:

| Report | Meaning |
| --- | --- |
| Only `CPUExecutionProvider`, plus a "CPU-only build" note | Built without a GPU feature. Rebuild with one. |
| `… (compiled in, missing from the linked ONNX Runtime)` | The feature was enabled but ORT has no such provider. This is the `rocm` trap above, and the build can never use a GPU. |
| `… (ready)` | The provider is present. It can still fail to *load* on this machine (no driver, no device); the run reports that. |

**Then watch a run.** A job that loads on a GPU prints `running on
CUDAExecutionProvider` before it starts. If the build has GPU support but the
model loaded on CPU anyway, it says so as a warning instead of proceeding
silently. That's the case where the provider exists but the driver or the device
doesn't.

```sh
nvidia-smi dmon -s u          # or: rocm-smi --showuse / radeontop
# in another shell, run a batch and watch for non-zero GPU utilisation
```

A useful A/B: time the same batch with and without the feature flag. If the
timings match, the GPU is not engaged.

### Shipping a GPU build

`--gpu webgpu` produces no `onnxruntime_providers_*` library. It links
`libwebgpu_dawn.so` instead, and links it *directly*, so a package missing that
file doesn't fall back to CPU. The binary fails to start at all, with
`libwebgpu_dawn.so: cannot open shared object file`. `packaging/stage.sh` stages
it and refuses any GPU build that produced no runtime library at all.

Staging the libraries is not enough on its own: they sit beside the binaries in
the tarball and under `<prefix>/lib/figura-obscura` once installed, and the
dynamic loader searches neither. So `stage.sh` links GPU builds with

```
-C link-arg=-Wl,-rpath,$ORIGIN:$ORIGIN/../lib/figura-obscura
```

which covers both layouts, because `$ORIGIN` resolves against the binary's own
location. Without it a webgpu build doesn't start at all, and a cuda build
starts but can't `dlopen` its providers, which is the silent CPU fallback
again.
Confirm with `readelf -d <binary> | grep RUNPATH`.

## 4. Models

`obscura setup` is the one command a fresh host needs. It downloads the
recommended models, prints each digest, and checks that ffmpeg runs. `obscura models fetch
--model nudenet-320n` fetches a single model. Both write to
`~/.cache/figura-obscura/models/` (override with `OBSCURA_MODEL_DIR`).

The GUI does the same work from its first-run screen and its Models page, and
the installers run `obscura setup --quiet` after copying files, so all three paths go
through the same downloader.

The NudeNet URLs are pinned to the `notAI-tech/NudeNet` `v3.4-weights` GitHub
release. Four of the five registry entries now carry a real `sha256`; only
`nudenet-640m` is still empty, because the release-asset host is unreachable
from the build container. It answers **200 with a GitHub sign-in HTML page**
instead of the file. `fetch` detects that (it sniffs the payload for an ONNX
ModelProto header and rejects web pages), fails loudly, and leaves nothing in
the cache, so a bad download can't masquerade as a model.

The three anime digests came from HuggingFace's own metadata, which reports the
SHA-256 of every LFS file without serving any of it, so they could be pinned
from inside the container. `nudenet-640m` can't be done that way: it's a GitHub
release asset and the API reports no digest for it. On the host, once a fetch
succeeds it prints the file's SHA-256, and pasting that into the entry turns on
verification for every later download. [`RELEASING.md`](RELEASING.md) has the
exact commands.

### Which model for which content

| id | domain | notes |
|---|---|---|
| `nudenet-320n` | real-life | default; 320px, fast |
| `nudenet-640m` | real-life | 640px, more accurate |
| `anime-censor-v1-s` | **anime** | deepghs `censor_detect_v1.0_s`; yolov8s, F1 0.83, thr 0.238 |
| `anime-censor-v1-n` | **anime** | deepghs `censor_detect_v1.0_n`; yolov8n, F1 0.80, thr 0.278 |
| `anime-censor-v0.10-s` | **anime** | deepghs `censor_detect_v0.10_s`; earlier training run, F1 0.83, thr 0.15 |

For anime or illustrated content use `anime-censor-v1-s`. Be aware it detects
only **3 classes** (`nipple_f`, `penis`, `pussy`), mapped onto
female-breast-exposed, male-genitalia-exposed and female-genitalia-exposed. It cannot see buttocks,
anus, feet, belly, armpits or face, and never reports `Covered` states, so the
filter tree is far coarser than with NudeNet. Its confidence default is 0.238
(the repo's published F1-optimal threshold) rather than the usual 0.20.

Also note `nipple_f` is a **nipple** class, not a breast class: the box bounds
the areola, so the censor rectangle is areola-sized no matter the figure. To
cover the breast, raise `censor.shape.padding` (default 0.10) in a profile or
with the GUI's padding spin box. There is no CLI flag for it.
[`DETECTORS.md`](DETECTORS.md) covers this per model, and the README covers the
tiling and resampling settings.

The three anime entries share one taxonomy but not their weights, so they can
cross-examine each other in a single run:

```sh
obscura process ./in -o ./out --model anime-censor-v1-s \
    --also-model anime-censor-v0.10-s --min-votes 2
```

`--min-votes 1` (the default) is a union: anything any model saw. Higher values
demand consensus. Each model keeps its own published threshold unless `--set`
overrides it, and those thresholds differ substantially (0.15 to 0.278).

**Candidates that were checked and rejected.** `Anzhc/Anzhcs_YOLOs` is
reachable and does contain relevant detectors (`Anzhc Breasts Seg v1 1024{n,s,m}`, which would solve the
nipple-vs-breast geometry problem by detecting the breast itself). It is
unusable as-is on three counts: the repo ships **`.pt` only**, no ONNX, so it
would need an ultralytics export step; the breast models are **segmentation**
nets whose output tensor layout Obscura's YOLOv8 box decoder cannot read; and the
repo is **AGPL-3.0**, unlike the MIT/Apache weights Obscura otherwise ships. Worth
revisiting if a mask-capable censor path is ever added. Searches for other
anime-domain ONNX *detectors* turned up only whole-image NSFW *classifiers*
(`Falconsai/nsfw_image_detection` and its ONNX mirrors), which produce a score,
not boxes, and so cannot drive a censor.

**A usable NudeNet mirror exists.** `deepghs/nudenet_onnx` (Apache-2.0) carries
`320n.onnx`, the same weights as the blocked GitHub release asset. If the GitHub
asset host stays unreachable on the host, repoint `nudenet-320n`'s URL at
`https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx` and verify
the SHA-256 matches the GitHub copy before trusting it. It has no `640m.onnx`.

Its weights are on HuggingFace's LFS CDN (`us.aws.cdn.hf.co`), which is
unreachable from the build container just like GitHub's asset host, so a digest
taken from a mirror has to be checked against the original. `ob-models` honours
`OBSCURA_MODEL_DIR` and a per-model URL, so a mirror works.

## 4b. Running the GUI

The build produces two binaries; the GUI takes no arguments:

```sh
./target/release/obscura-gui          # or: cargo run -p obscura-gui --release
```

**Models come down on first run.** The GUI offers the recommended set on its
first-run screen, and its Models page installs, re-downloads and removes
individual files with per-model progress. Building a detector calls
`ob_models::require`, which errors rather than fetching, so if you skip the
offer then preview and Run fail with "model `X` is not downloaded" until you
fetch one from that page or from the CLI:

```sh
./target/release/obscura models fetch nudenet-320n
```

Both binaries read the same cache (`~/.cache/figura-obscura/models`, or
`OBSCURA_MODEL_DIR`), so one fetch serves both.

**GPU applies to the GUI too, but only if compiled in.** The execution provider
is a compile-time feature, not a runtime switch, and `obscura-gui` depends on
`ob-detect` the same way the CLI does. Build the workspace with the flag and
both binaries get it:

```sh
cargo build --release --features ob-detect/cuda      # or .../webgpu
```

To put them on `PATH`:

```sh
cargo install --path crates/ob-cli --features ob-detect/cuda
cargo install --path crates/ob-gui --features ob-detect/cuda
```

Both binaries take `--version`, and print the commit they were built from as
well as `1.0.0`:

```sh
obscura --version          # obscura 1.0.0 (3e0e567)
obscura-gui --version      # obscura-gui 1.0.0 (3e0e567)
```

Check it before reporting a bug: an installed binary does not update when the
checkout does, so a crash that the working tree has already fixed usually means
the install is older than the fix. The GUI shows the same string on its About
page, which is the way to read it on Windows (no console is attached there).

The GUI is eframe/winit on OpenGL and picks up Wayland or X11 automatically from
`WAYLAND_DISPLAY`/`DISPLAY`. With neither set it exits immediately with
`neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set`. That's the
expected headless behaviour, and it's how the binary was smoke-tested here, not
a build fault. Over SSH, use `ssh -X` or run it on the desktop session.

## 5. Verify end-to-end

What is worth checking on a real machine, in rough order:

- **A detect smoke test**, then an image batch, then video. For video, confirm
  the audio is bit-identical to the source, that coverage doesn't flicker
  between frames, and that fail-closed does what it claims.
- **Offline proof.** Run a batch under `unshare -n`. Nothing on the processing
  path should notice.
- **Odd frame dimensions.** The encoder uses `yuv420p`, which needs even width
  and height. Add a scale or pad filter if sources can be odd.
- **Video throughput.** Frames stream through a pipe, so memory stays flat, but
  confirm the rate on a long file.
- **Stripped binaries.** `file target/release/obscura` should say "stripped".

## 6. Release (opaque native binary, R9)

```sh
cargo build --release        # profile: opt-level=z, LTO, panic=abort, strip
file target/release/obscura       # confirm "stripped"
```
The release profile works and produced a ~27 MB `obscura` and a ~33 MB
`obscura-gui`. GPU providers stay feature-gated so the default artifact stays
lean.

## 7. What changed during the first successful build

Three defects surfaced only once the code actually ran:

- `ob-job`. The `censor_frame_censors_selected_region` test asserted on a
  **flat** 8×8 frame, but the default style is `Pixelate`, and averaging a
  uniform region returns that same value, so the assertion could never hold.
  It now uses a checkerboard, matching `ob-censor`'s own pixelate test.
- `ob-detect`. A decode test indexed with `0 * n`, which clippy denies as a
  correctness lint (`erasing_op`). Replaced with a `set(c, i, v)` helper that
  mirrors `decode`'s own accessor.
- `ob-models`. `fetch` committed **any** downloaded bytes as a `.onnx` and
  reported `ok`. With an unpinned checksum, an intercepted download (login page,
  captive portal, proxy) was cached as a model and only failed much later at
  inference. It now validates the payload first. See §4.

One feature was added: `ob-detect/webgpu`, registering ORT's WebGPU execution
provider. It exists because ROCm has no prebuilt binary (§3), leaving AMD cards
with no working GPU path that didn't require compiling ONNX Runtime by hand.

`ob-media`'s fps fallback is written as `!(fps > 0.0)` **deliberately**. The
negation is what catches `NaN`, since `fps <= 0.0` is false for it. It carries an
`#[allow]` and a comment so the lint suggestion doesn't turn it into a bug.
