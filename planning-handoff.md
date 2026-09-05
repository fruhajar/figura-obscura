# Planning Handoff (historical)

> **RENAMED 2026-09-05.** This project is now **Figura Obscura**; it was called
> "SodomyBatch" while it was being planned and built. The old name is left
> standing everywhere below **on purpose**: this file is a record of what was
> asked for and what was researched, and the section-1 request is quoted
> verbatim. Rewriting a quotation to match a later decision would make it
> useless as a record. Nothing else in the repository still uses the old name.
>
> Crate names in the §5 sketch are likewise as-drafted (`sb-*`); the shipped
> crates are `ob-*`. See `README.md` for the current architecture.


> **STATUS: RESOLVED 2026-08-22.** Planning is complete and the final plan was
> approved. This file is kept only for its research record; it is superseded by:
> - **Approved plan:** `~/.claude/plans/twinkling-foraging-puppy.md`
> - **Code scaffold:** the Cargo workspace in this directory (`crates/*`), plus
>   `README.md` and `HOST-BUILD.md`.
>
> Locked decisions: all-Rust (`ort` + `egui`), cross-vendor GPU with CPU
> fallback, Linux-only v1, bounding boxes only. Remaining research item —
> verifying `Anzhc/Anzhcs_YOLOs` — was dropped for v1 (HuggingFace egress
> blocked; the better-licensed deepghs anime detectors cover the gap).

---

## 1. Original request (verbatim)

> Plan out a batch-image and video processing cli+gui tool, which would allow selectively censoring lewd parts. This tool should allow setting filters for male/female with subcategories such as primarily breasts, buttocks, eyes, genitalia. The censorship part should allow either using a solid fill color, pixelation, or a preset image. The tool should allow processing of both real life or anime style images and videos, inputting of multiple files/directories, based on a selection of predefined model, with options for the model settings if available, with tooltips for each setting in the gui. This tool should be entirely local, no external connectivity should be needed, as such, I know this requires downloading the models, but iirc that isn't that difficult nor processing intensive for something simple like this. Plan out the architecture of the software, the deployment should be deployable as a non-human-readable binary. The structure of the tool should allow utilizing some parts of the code for other tools as well, with a known next tool being a real-time screen censorship tool with similar settings. The name of this tool will be SodomyBatch, or SB for short.

Follow-up instruction: *"save the prompt and initial plan stages into a file, I'll have to rerun the container with new egress filtering."*

## 2. Requirements extracted

| # | Requirement | Notes / implications |
|---|---|---|
| R1 | Batch processing of **images and videos** | Video needs decode → per-frame detect → composite → re-encode, with audio passthrough |
| R2 | **CLI + GUI**, same feature set | Shared core lib; GUI is a thin front-end over the same job engine |
| R3 | **Selective** censoring by category | Filter tree: sex (male/female/unknown) × part (breasts, buttocks, genitalia, anus, eyes/face, feet, belly, armpits…) × exposed/covered |
| R4 | Censor styles: **solid fill color, pixelation, preset image (overlay/sticker)** | Also worth planning: blur/gaussian, mosaic block-size, per-category style overrides |
| R5 | Handles **real-life AND anime/illustrated** content | Different detector models; needs a model-per-domain selector + possibly auto style classification |
| R6 | Input: **multiple files and directories** | Recursive globbing, include/exclude patterns, output dir mirroring |
| R7 | **Predefined model selection** with exposed model settings + **GUI tooltips per setting** | Model registry with declarative setting metadata (name, type, range, default, tooltip) so CLI flags, GUI widgets and tooltips are all generated from one source |
| R8 | **Fully local / offline**; models downloaded once | One-time model fetch (separate opt-in step), integrity-checked; zero network at processing time |
| R9 | Deployable as a **non-human-readable binary** | Rules out plain PyInstaller (trivially unpackable). Favors a compiled language, or Nuitka |
| R10 | **Reusable core** — next tool is a real-time screen censor | Core must be a library with a frame-in/frame-out API, low-latency path, tracking/temporal smoothing |

## 3. Environment observed

- `/work` is **empty** — greenfield project, not a git repo.
- Container: Debian 12 (bookworm), Python 3.11.2, `pip` present.
- **No** `uv`, `ffmpeg`, `nvidia-smi` on PATH at planning time (host is CachyOS / Linux 7.2.0 — likely has a GPU, unverified).

## 4. Research findings so far (verified via web)

### NudeNet v3 (real-life photos) — primary candidate detector
- Source: `notAI-tech/NudeNet`, PyPI `nudenet` (3.4.x). ONNX Runtime based (no TensorFlow).
- Models: `320n.onnx` (yolov8n @ 320×320, default, bundled in the pip package) and `640m.onnx` (yolov8m @ 640×640, more accurate).
- Preprocess: RGBA→BGR, scale 1/255.0, letterbox to model input.
- Defaults: conf ≥ 0.2, NMS score 0.25, NMS IoU 0.45.
- **18 classes (confirmed exact list):**
  `FEMALE_GENITALIA_COVERED, FACE_FEMALE, BUTTOCKS_EXPOSED, FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED, ANUS_EXPOSED, FEET_EXPOSED, BELLY_COVERED, FEET_COVERED, ARMPITS_COVERED, ARMPITS_EXPOSED, FACE_MALE, BELLY_EXPOSED, MALE_GENITALIA_EXPOSED, ANUS_COVERED, FEMALE_BREAST_COVERED, BUTTOCKS_COVERED`
- ⚠️ This maps almost 1:1 onto R3 (sex × part × exposed/covered) — **use this taxonomy as the canonical internal category enum** and map other models onto it.
- ⚠️ Gap vs R3: no **eyes** class (only FACE_MALE/FACE_FEMALE). Eye censoring needs either a separate face-landmark/eye detector or deriving an eye-strip from the face box.

### Anime / illustrated candidates (need verification)
- `Anzhc/Anzhcs_YOLOs` (HF) — YOLOv8/v11 **detection and segmentation** models trained on art/anime, incl. NSFW-relevant and face/eye models. **File list + license NOT yet verified** (fetch was interrupted).
- `deepghs/anime_object_detection` (HF Space) — anime face/head/body/eyes/**censor-point**/nudity detectors; also contains a `detection/nudenet.py`. deepghs publishes many per-task anime YOLO repos.
- `luxdelux7/ForbiddenVision_Models` — YOLOv11-S @640, face detect+segment across realistic/anime/NSFW.
- `ZygoteCode/NsfwSharp` — YOLOv11 ONNX NSFW model, C#/ONNX (model weights may be reusable).
- Segmentation (mask) models are valuable: they allow **contour-following censoring** instead of only rectangles.

### Runtime / language findings
- Rust `ort` crate (pykeio) 2.0-rc: safe wrapper over ONNX Runtime 1.28; CUDA / TensorRT / QNN execution providers behind cargo features; used for YOLO inference in production. Viable for an all-Rust implementation → satisfies R9 (native binary) and R10 (real-time screen tool) cleanly.
- Python path would need **Nuitka** (true C compilation) rather than PyInstaller to satisfy R9.

### Sources
- [nudenet · PyPI](https://pypi.org/project/nudenet/)
- [notAI-tech/NudeNet Python API (DeepWiki)](https://deepwiki.com/notAI-tech/NudeNet/5.1-python-api)
- [notAI-tech/NudeNet source `nudenet/nudenet.py`](https://raw.githubusercontent.com/notAI-tech/NudeNet/v3/nudenet/nudenet.py)
- [Anzhc/Anzhcs_YOLOs · Hugging Face](https://huggingface.co/Anzhc/Anzhcs_YOLOs)
- [deepghs/anime_object_detection · Hugging Face Space](https://huggingface.co/spaces/deepghs/anime_object_detection)
- [luxdelux7/ForbiddenVision_Models · Hugging Face](https://huggingface.co/luxdelux7/ForbiddenVision_Models/blob/main/README.md)
- [ZygoteCode/NsfwSharp](https://github.com/ZygoteCode/NsfwSharp)
- [pykeio/ort](https://github.com/pykeio/ort) · [ort on crates.io](https://crates.io/crates/ort)

## 5. Working architecture sketch (draft — not final)

Workspace of independent crates/packages so the real-time screen tool (R10) reuses everything but the I/O ends:

```
sb-core        detection taxonomy, Detection/Region types, model registry + settings
               metadata (type, range, default, tooltip text), config schema, profiles
sb-detect      ONNX Runtime inference backends; per-model adapters that map native
               class labels -> canonical taxonomy; letterbox/NMS; batching; EP selection
sb-censor      render pipeline: solid fill / pixelate / blur / preset-image overlay;
               box vs mask (segmentation) shapes; padding, rounding, feathering
sb-track       temporal smoothing: IoU tracker + hysteresis so video/real-time output
               doesn't flicker; detect every Nth frame, interpolate between
sb-media       image decode/encode; video demux/decode/encode + audio passthrough
               (ffmpeg), frame iterator abstraction
sb-job         job graph: input expansion (files/dirs/globs), worker pool, progress
               events, cancellation, resume, per-file error isolation, dry-run
sb-cli         `sb` binary — argument surface generated from the same settings metadata
sb-gui         desktop GUI — queue view, filter tree, style editor, live preview,
               tooltips sourced from sb-core metadata
sb-models      model manifest, one-time downloader, checksum verification, local cache
(future) sb-screen  screen capture source + overlay sink reusing detect/censor/track
```

Key design commitments to carry into the final plan:
1. **One canonical category taxonomy** (NudeNet-derived) + per-model label mapping tables. Filters, profiles and CLI flags speak the canonical taxonomy only.
2. **Settings metadata is declarative and single-source** — CLI flags, GUI widgets, and GUI tooltips are all generated from it (R7).
3. **Frame-in → detections → composited-frame-out** is a pure function; batch and real-time differ only in source/sink and latency budget (R10).
4. **Fail-closed option**: if a frame fails detection, optionally blank/skip rather than emit uncensored output.
5. **Offline by construction**: network code exists only in `sb-models`, gated behind an explicit `sb models fetch` command; the processing path has no network dependency at all.

## 6. Open questions for the user (ask on resume)

1. **Language/stack**: all-Rust (`ort` + `egui`/Tauri — best fit for R9 + R10) vs Python + Nuitka (fastest to build, weaker binary opacity) vs C++/Qt.
2. **GPU support**: CUDA/TensorRT/DirectML/ROCm, or CPU-only baseline first? (Affects binary size and packaging enormously.)
3. **Target OS(es)**: Linux-only, or Windows too (the screen-capture tool later differs a lot per-OS).
4. **Censor granularity**: rectangles only, or segmentation masks (contour-following) where the model supports it?
5. **Model licensing tolerance** — some anime NSFW YOLO weights have unclear/non-commercial licenses; is this personal-use only?
6. **Video output policy**: re-encode everything (quality loss, slow) vs stream-copy audio + re-encode video only; acceptable codecs/quality knobs.
7. **"Eyes" category**: derive from face box, or add a dedicated eye/face-landmark model?

## 7. Remaining research to do after container restart

- Verify `Anzhc/Anzhcs_YOLOs` file list, classes, and **license** (fetch was interrupted).
- Enumerate `deepghs` anime detection repos (face/eyes/censor-point/nudity) and their licenses.
- Confirm `ort` 2.0 release status and its ONNX Runtime version/EP feature matrix.
- Check GUI options: `egui`/`eframe` vs Tauri vs Slint — tooltip support, image preview performance, packaging into one binary.
- Check ffmpeg integration route: bundled static `ffmpeg` binary (GPL implications) vs `ffmpeg-next` bindings vs `symphonia`/`re_mp4`.
- Confirm host GPU (`nvidia-smi` absent in container; host is CachyOS).
