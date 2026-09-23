//! ONNX Runtime session wiring, isolated so the `ort` API surface Obscura depends on
//! stays small (ort 2.0 is rc — API not yet frozen; see plan Risks).
//!
//! The full detect flow — letterbox → run → decode YOLO output → NMS → invert
//! coordinates → map labels — is assembled here. The only calls into `ort` are
//! [`build_session`] and [`OnnxDetector::run`]; everything else is pure and unit
//! tested without a real model.

use crate::postprocess::{map_label, nms};
use crate::preprocess::{letterbox_chw_with, Letterbox, Resampler};
use crate::{DetectError, Detector, ExecProvider};
use ob_core::geometry::{BBox, Detection, Frame};
use ob_core::registry::{LabelMap, ModelEntry};
use ob_core::settings::SettingValues;
use ob_core::taxonomy::Category;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::Mutex;

/// A loaded ONNX detector bound to one model entry and its resolved settings.
pub struct OnnxDetector {
    #[allow(dead_code)]
    model_path: PathBuf,
    input_size: u32,
    label_map: LabelMap,
    conf_threshold: f32,
    nms_iou: f32,
    /// The provider the session actually loaded on — see
    /// [`Detector::execution_provider`]. Previously this held the *requested*
    /// list and was never read, which is why nothing noticed that a GPU build
    /// was running on CPU.
    execution_provider: ExecProvider,
    /// Why each preferred provider ahead of the active one was passed over.
    /// Empty on a clean GPU load, and the whole diagnosis when it is not.
    ep_notes: Vec<String>,
    /// The name of the model's single image input tensor (e.g. `"images"`).
    input_name: String,
    /// Kernel used to scale frames into model input space. See
    /// [`Resampler`] — the default is deliberately not nearest-neighbour.
    resampler: Resampler,
    /// The ONNX Runtime session. `Session::run` takes `&mut self`, but the
    /// batch engine shares one `Detector` across rayon worker threads via
    /// `&dyn Detector`, so the session is guarded by a `Mutex`. ORT serializes
    /// concurrent `Run` calls on one session internally regardless, so this
    /// costs nothing real while keeping `OnnxDetector: Send + Sync`.
    session: Mutex<Session>,
}

/// Map Obscura's ordered EP preference to `ort` execution-provider dispatches.
///
/// Only providers whose Cargo feature is enabled are compiled in; CPU is always
/// available and always last. Kept in the same order as
/// [`crate::preferred_execution_providers`], which names these for reporting.
// See `preferred_execution_providers` — `#[cfg]`-gated elements, so this cannot
// be a `vec![]` literal however it looks on a CPU-only build.
#[allow(clippy::vec_init_then_push)]
fn execution_provider_dispatches() -> Vec<(ExecProvider, ExecutionProviderDispatch)> {
    #[allow(unused_mut)]
    let mut eps: Vec<(ExecProvider, ExecutionProviderDispatch)> = Vec::new();
    #[cfg(feature = "cuda")]
    eps.push((
        ExecProvider::Cuda,
        ort::execution_providers::CUDAExecutionProvider::default().build(),
    ));
    #[cfg(feature = "rocm")]
    eps.push((
        ExecProvider::Rocm,
        ort::execution_providers::ROCmExecutionProvider::default().build(),
    ));
    #[cfg(feature = "webgpu")]
    eps.push((
        ExecProvider::WebGpu,
        ort::execution_providers::WebGPUExecutionProvider::default().build(),
    ));
    #[cfg(feature = "directml")]
    eps.push((
        ExecProvider::DirectMl,
        ort::execution_providers::DirectMLExecutionProvider::default().build(),
    ));
    #[cfg(feature = "coreml")]
    eps.push((
        ExecProvider::CoreMl,
        ort::execution_providers::CoreMLExecutionProvider::default().build(),
    ));
    eps.push((
        ExecProvider::Cpu,
        ort::execution_providers::CPUExecutionProvider::default().build(),
    ));
    eps
}

/// What one execution provider can do in *this* binary, on *this* machine.
///
/// The two flags answer two different questions that look alike until they
/// disagree, which is exactly when a GPU run turns out to be a CPU run:
/// `compiled_in` is about the build, `available` is about the ONNX Runtime it
/// links. The `rocm` feature makes them disagree by construction — there is no
/// ROCm prebuilt for linux-x86_64, so `--features rocm` builds happily against
/// the CPU runtime and produces a "GPU" binary that cannot ever use a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpStatus {
    pub provider: ExecProvider,
    /// This binary was compiled with the provider's Cargo feature.
    pub compiled_in: bool,
    /// The linked ONNX Runtime actually ships this provider.
    pub available: bool,
}

/// Report every execution provider this build asks for, and whether the linked
/// ONNX Runtime can actually supply it.
///
/// Cheap and model-free — it only asks ORT what it was built with — so both
/// binaries can answer "is this a GPU build?" from `--version`, without a model
/// file or a run.
pub fn probe_execution_providers() -> Vec<EpStatus> {
    #[allow(unused_imports)]
    use ort::execution_providers::ExecutionProvider as _;

    // `is_available` asks the linked runtime for its provider list and looks
    // for this provider's own name in it, so it has to be called on the
    // concrete type — which only exists when the feature compiled it in. The
    // arms therefore mirror `execution_provider_dispatches` rather than mapping
    // over `preferred_execution_providers`.
    #[allow(unused_mut)]
    let mut out: Vec<EpStatus> = Vec::new();

    #[allow(unused_macros)]
    macro_rules! probe {
        ($provider:expr, $ty:ty) => {
            out.push(EpStatus {
                provider: $provider,
                compiled_in: true,
                available: <$ty>::default().is_available().unwrap_or(false),
            });
        };
    }

    #[cfg(feature = "cuda")]
    probe!(
        ExecProvider::Cuda,
        ort::execution_providers::CUDAExecutionProvider
    );
    #[cfg(feature = "rocm")]
    probe!(
        ExecProvider::Rocm,
        ort::execution_providers::ROCmExecutionProvider
    );
    #[cfg(feature = "webgpu")]
    probe!(
        ExecProvider::WebGpu,
        ort::execution_providers::WebGPUExecutionProvider
    );
    #[cfg(feature = "directml")]
    probe!(
        ExecProvider::DirectMl,
        ort::execution_providers::DirectMLExecutionProvider
    );
    #[cfg(feature = "coreml")]
    probe!(
        ExecProvider::CoreMl,
        ort::execution_providers::CoreMLExecutionProvider
    );
    probe!(
        ExecProvider::Cpu,
        ort::execution_providers::CPUExecutionProvider
    );

    out
}

/// One line per execution provider, for `--version` and bug reports.
///
/// Reads as `CUDAExecutionProvider (compiled in, runtime missing)` — the case
/// that explains a GPU build pegged at a few percent GPU load.
pub fn execution_provider_report() -> Vec<String> {
    probe_execution_providers()
        .into_iter()
        .map(|s| {
            let state = match (s.compiled_in, s.available) {
                (true, true) => "ready",
                (true, false) => "compiled in, missing from the linked ONNX Runtime",
                (false, _) => "not compiled in",
            };
            format!("{} ({})", s.provider, state)
        })
        .collect()
}

/// Build an `ort::Session` for a model file, returning it alongside the
/// execution provider that actually took.
///
/// Each GPU provider is tried on its own with `error_on_failure`, newest
/// builder each time, and a failure moves to the next candidate. The bulk
/// `with_execution_providers` call this replaces returned `Ok` whether or not
/// any GPU provider registered, so a driverless CUDA build reported success and
/// then ran the whole job on CPU. Trying one at a time costs a discarded
/// session build per failed provider — once per detector load — and buys a
/// truthful answer about what is running the model.
fn build_session(
    entry: &ModelEntry,
    model_path: &PathBuf,
) -> Result<(Session, ExecProvider, Vec<String>), DetectError> {
    let mut notes = Vec::new();
    let candidates = execution_provider_dispatches();
    let last = candidates.len().saturating_sub(1);

    for (i, (provider, dispatch)) in candidates.into_iter().enumerate() {
        // CPU is the floor: it is never "tried and rejected", so let its
        // failure be the error the caller sees.
        let is_fallback = i == last;
        let dispatch = if is_fallback {
            dispatch
        } else {
            dispatch.error_on_failure()
        };

        let built = Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_execution_providers([dispatch]))
            .and_then(|b| b.commit_from_file(model_path));

        match built {
            Ok(session) => return Ok((session, provider, notes)),
            Err(e) if !is_fallback => {
                notes.push(format!("{provider} unavailable: {e}"));
            }
            Err(e) => return Err(DetectError::Load(entry.id.to_string(), e.to_string())),
        }
    }

    Err(DetectError::Load(
        entry.id.to_string(),
        "no execution provider could load the model".into(),
    ))
}

/// Read the model's real square input side from its graph, if it declares one.
///
/// A YOLOv8 export usually bakes a fixed `[1, 3, H, W]` input shape in, and a
/// registry entry stating a different `input_size` would either fail the run
/// outright or feed the model a resolution it was never trained for. Dynamic
/// axes come back as `-1`, in which case there is nothing to learn and the
/// registry's value stands.
fn declared_input_size(session: &Session) -> Option<u32> {
    let input = session.inputs.first()?;
    let ort::value::ValueType::Tensor { shape, .. } = &input.input_type else {
        return None;
    };
    if shape.len() != 4 {
        return None;
    }
    let (h, w) = (shape[2], shape[3]);
    if h > 0 && w > 0 && h == w {
        Some(h as u32)
    } else {
        None
    }
}

impl OnnxDetector {
    /// Build a detector for `entry` using `settings`, loading the model from
    /// `model_path` (produced by `ob-models`).
    pub fn load(
        entry: &ModelEntry,
        settings: &SettingValues,
        model_path: PathBuf,
    ) -> Result<Self, DetectError> {
        if !model_path.exists() {
            return Err(DetectError::ModelMissing(entry.id.to_string()));
        }
        let get = |k: &str, d: f64| settings.get(k).and_then(|v| v.as_f64()).unwrap_or(d) as f32;

        let (session, execution_provider, ep_notes) = build_session(entry, &model_path)?;
        // YOLOv8 exports use a single image input; read its real name so we bind
        // by name rather than assuming "images".
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "images".to_string());

        // Trust the model file over the registry: a mis-stated input_size would
        // otherwise surface as an opaque ORT shape error at the first run.
        let input_size = declared_input_size(&session).unwrap_or(entry.input_size);

        let resampler = settings
            .get("resample")
            .and_then(|v| v.as_str())
            .and_then(Resampler::parse)
            .unwrap_or_default();

        Ok(Self {
            model_path,
            input_size,
            label_map: entry.label_map(),
            conf_threshold: get("conf_threshold", 0.2),
            nms_iou: get("nms_iou", 0.45),
            execution_provider,
            ep_notes,
            input_name,
            resampler,
            session: Mutex::new(session),
        })
    }

    /// The input side actually in use — the model file's own, where it declares
    /// one. Tiling needs this to size its grid.
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// The NMS IoU this detector was built with, so a wrapper merging several
    /// passes can use the same value.
    pub fn nms_iou(&self) -> f32 {
        self.nms_iou
    }

    /// Why a preferred provider was passed over, in preference order. Empty
    /// when the first choice loaded.
    pub fn ep_notes(&self) -> &[String] {
        &self.ep_notes
    }

    /// Run the raw model on a CHW tensor, returning the flat output tensor and
    /// its shape. This is the only function that feeds data through `ort`.
    fn run(&self, input_chw: &[f32]) -> Result<(Vec<f32>, Vec<usize>), DetectError> {
        let size = self.input_size as usize;
        let shape = [1_usize, 3, size, size];
        let tensor = Tensor::from_array((shape, input_chw.to_vec()))
            .map_err(|e| DetectError::Inference(e.to_string()))?;

        // `Session::run` needs `&mut`; take the lock for the duration of the
        // call and copy the output out before releasing it (the borrowed output
        // view is tied to the session).
        let mut session = self
            .session
            .lock()
            .map_err(|_| DetectError::Inference("detector session mutex poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| DetectError::Inference(e.to_string()))?;

        // A YOLOv8 detect graph has a single output; take the first.
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectError::Inference(e.to_string()))?;
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        Ok((data.to_vec(), dims))
    }

    /// Decode a YOLOv8 detect output `[1, 4+C, N]` into detections in *model*
    /// coordinates, before NMS and letterbox inversion.
    fn decode_yolov8(&self, output: &[f32], shape: &[usize]) -> Vec<Detection> {
        // Expect [1, 4+C, N].
        if shape.len() != 3 {
            return Vec::new();
        }
        let channels = shape[1];
        let n = shape[2];
        let num_classes = channels.saturating_sub(4);
        let mut dets = Vec::new();
        // Output is channel-major: value(c, i) = output[c*n + i].
        let at = |c: usize, i: usize| output[c * n + i];
        for i in 0..n {
            // Best class for box i.
            let mut best_c = 0usize;
            let mut best_s = 0.0f32;
            for c in 0..num_classes {
                let s = at(4 + c, i);
                if s > best_s {
                    best_s = s;
                    best_c = c;
                }
            }
            if best_s < self.conf_threshold {
                continue;
            }
            let Some(category) = map_label(&self.label_map, best_c) else {
                continue;
            };
            // xywh (center) in model pixels.
            let cx = at(0, i);
            let cy = at(1, i);
            let w = at(2, i);
            let h = at(3, i);
            dets.push(Detection {
                bbox: BBox::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
                category,
                score: best_s,
            });
        }
        dets
    }
}

impl Detector for OnnxDetector {
    /// A model can emit exactly the categories its label map names — the
    /// three-class anime detectors can never report buttocks, however
    /// confidently they are asked.
    fn can_emit(&self, category: Category) -> bool {
        self.label_map.by_index.contains(&category)
    }

    fn execution_provider(&self) -> Option<ExecProvider> {
        Some(self.execution_provider)
    }

    fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, DetectError> {
        let (input, lb): (Vec<f32>, Letterbox) =
            letterbox_chw_with(frame, self.input_size, self.resampler);
        let (output, shape) = self.run(&input)?;
        let mut dets = self.decode_yolov8(&output, &shape);
        dets = nms(dets, self.nms_iou);
        // Map boxes back to original-image coordinates and clamp.
        for d in &mut dets {
            d.bbox = lb.invert(&d.bbox);
            d.bbox.x1 = d.bbox.x1.clamp(0.0, frame.width_f());
            d.bbox.y1 = d.bbox.y1.clamp(0.0, frame.height_f());
            d.bbox.x2 = d.bbox.x2.clamp(0.0, frame.width_f());
            d.bbox.y2 = d.bbox.y2.clamp(0.0, frame.height_f());
        }
        Ok(dets)
    }
}

#[cfg(test)]
mod tests {
    // NOTE: `decode_yolov8` is pure but reads `self.conf_threshold` and
    // `self.label_map`; a full `OnnxDetector` now owns a real `ort::Session`,
    // which can't be constructed without a model file. The decode/NMS math is
    // therefore exercised through a free helper mirroring the method, keeping
    // this file's logic testable on the host without a model download.
    use super::*;
    use ob_core::registry::nudenet_label_map;

    fn decode(
        label_map: &LabelMap,
        conf_threshold: f32,
        output: &[f32],
        shape: &[usize],
    ) -> Vec<Detection> {
        if shape.len() != 3 {
            return Vec::new();
        }
        let channels = shape[1];
        let n = shape[2];
        let num_classes = channels.saturating_sub(4);
        let mut dets = Vec::new();
        let at = |c: usize, i: usize| output[c * n + i];
        for i in 0..n {
            let mut best_c = 0usize;
            let mut best_s = 0.0f32;
            for c in 0..num_classes {
                let s = at(4 + c, i);
                if s > best_s {
                    best_s = s;
                    best_c = c;
                }
            }
            if best_s < conf_threshold {
                continue;
            }
            let Some(category) = map_label(label_map, best_c) else {
                continue;
            };
            let cx = at(0, i);
            let cy = at(1, i);
            let w = at(2, i);
            let h = at(3, i);
            dets.push(Detection {
                bbox: BBox::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
                category,
                score: best_s,
            });
        }
        dets
    }

    #[test]
    fn decode_yolov8_extracts_confident_box() {
        let lm = nudenet_label_map();
        // One box, 18 classes -> channels = 22, n = 1.
        let n = 1;
        let channels = 22;
        let mut out = vec![0.0f32; channels * n];
        // Mirrors decode's accessor: channel `c` of box `i` lives at c * n + i.
        let mut set = |c: usize, i: usize, v: f32| out[c * n + i] = v;
        set(0, 0, 100.0); // cx
        set(1, 0, 100.0); // cy
        set(2, 0, 20.0); // w
        set(3, 0, 20.0); // h
        set(4 + 3, 0, 0.9); // class 3 (FEMALE_BREAST_EXPOSED)
        let dets = decode(&lm, 0.2, &out, &[1, channels, n]);
        assert_eq!(dets.len(), 1);
        assert_eq!(
            dets[0].category,
            ob_core::taxonomy::cat::FEMALE_BREAST_EXPOSED
        );
        assert!((dets[0].bbox.x1 - 90.0).abs() < 1e-3);
    }

    #[test]
    fn decode_drops_low_confidence() {
        let lm = nudenet_label_map();
        let n = 1;
        let channels = 22;
        let mut out = vec![0.0f32; channels * n];
        out[(4 + 3) * n] = 0.05; // below 0.2
        assert!(decode(&lm, 0.2, &out, &[1, channels, n]).is_empty());
    }
}
