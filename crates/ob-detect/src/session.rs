//! ONNX Runtime session wiring, isolated so the `ort` API surface Obscura depends on
//! stays small (ort 2.0 is rc — API not yet frozen; see plan Risks).
//!
//! The full detect flow — letterbox → run → decode YOLO output → NMS → invert
//! coordinates → map labels — is assembled here. The only calls into `ort` are
//! [`build_session`] and [`OnnxDetector::run`]; everything else is pure and unit
//! tested without a real model.

use crate::postprocess::{map_label, nms};
use crate::preprocess::{letterbox_batch, Resampler};
use crate::{DetectError, Detector, ExecProvider};
use ob_core::geometry::{BBox, Detection, Frame, Region};
use ob_core::registry::{LabelMap, ModelEntry};
use ob_core::settings::SettingValues;
use ob_core::taxonomy::Category;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::{Condvar, Mutex};

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
    /// Most windows this detector will feed through the graph in one run.
    ///
    /// Zero or one means the model's export fixed its batch axis, so every
    /// window has to go on its own and [`Detector::detect_regions`] falls back
    /// to the default per-window loop.
    batch_limit: usize,
    /// The ONNX Runtime sessions. See [`SessionPool`] for why there is more
    /// than one.
    pool: SessionPool,
}

/// How many windows are fed through the graph in a single run, by default.
///
/// **One**, which is to say: off. This is a measured result and it went against
/// expectation. Feeding a tile grid to the graph as a batch is the obvious
/// optimisation — one run instead of thirteen, each paying the per-run overhead
/// once — and on the ONNX Runtime CPU provider it is consistently *slower*,
/// monotonically worse as the batch grows, at every session and thread count
/// tried. The CPU kernels already spread one inference across the intra-op
/// threads, and a batched tensor buys no extra parallelism while costing cache
/// locality and a much larger transient allocation. The numbers are in
/// `docs/PERFORMANCE.md`.
///
/// The machinery is kept because the reasoning that motivated it still holds
/// for a GPU, where per-launch overhead dominates and wide batches are the
/// normal way to feed one. That is unverified here — this environment has no
/// GPU — so it is not a default anybody gets by accident. A GPU user can try it
/// with `OBSCURA_MAX_BATCH=8` and measure.
const MAX_BATCH: usize = 1;

/// The ceiling on `OBSCURA_MAX_BATCH`, so a typo cannot ask for a tensor that
/// will not fit. Sixteen windows of a 640px model is 78 MB.
const BATCH_CEILING: usize = 16;

/// A small pool of identical ONNX Runtime sessions.
///
/// `Session::run` takes `&mut self`, and the batch engine shares one `Detector`
/// across rayon workers via `&dyn Detector`, so a session has to be guarded.
/// A single `Mutex<Session>` was the obvious guard — and the reason
/// `images.par_iter()` never actually ran inference in parallel: every worker
/// queued on the same lock, so a twelve-core machine did image decode and
/// censoring twelve ways and inference one way. ONNX Runtime does serialise
/// concurrent `Run` calls on one session internally, which is precisely why the
/// fix is to hold more than one session rather than a cleverer lock.
struct SessionPool {
    free: Mutex<Vec<Session>>,
    returned: Condvar,
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        Self {
            free: Mutex::new(sessions),
            returned: Condvar::new(),
        }
    }

    /// Take a session, waiting if every one is busy. The lease returns it on
    /// drop, including when the run between them panics or errors out.
    fn checkout(&self) -> Result<Lease<'_>, DetectError> {
        let mut free = self
            .free
            .lock()
            .map_err(|_| DetectError::Inference("detector session pool poisoned".into()))?;
        loop {
            if let Some(session) = free.pop() {
                return Ok(Lease {
                    pool: self,
                    session: Some(session),
                });
            }
            free = self
                .returned
                .wait(free)
                .map_err(|_| DetectError::Inference("detector session pool poisoned".into()))?;
        }
    }
}

/// One checked-out session, returned to the pool when dropped.
struct Lease<'a> {
    pool: &'a SessionPool,
    session: Option<Session>,
}

impl std::ops::Deref for Lease<'_> {
    type Target = Session;
    fn deref(&self) -> &Session {
        self.session.as_ref().expect("lease holds a session")
    }
}

impl std::ops::DerefMut for Lease<'_> {
    fn deref_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("lease holds a session")
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            // A poisoned pool still has to take the session back, or the pool
            // bleeds capacity and eventually every worker blocks forever.
            let mut free = self.pool.free.lock().unwrap_or_else(|e| e.into_inner());
            free.push(session);
            self.pool.returned.notify_one();
        }
    }
}

/// How a detector's CPU budget is split between concurrent sessions.
///
/// The two numbers cannot be chosen separately, which is the trap this exists
/// to avoid. ONNX Runtime gives **each** session an intra-op thread pool sized
/// to the machine by default, so simply holding four sessions on a twelve-core
/// box asks for forty-eight compute threads plus the batch engine's own
/// workers. Measured, that was slower than the single serialised session it
/// replaced — the image cases regressed by a third — because the cores spent
/// their time changing their minds about what to run.
///
/// So the budget is divided rather than multiplied: `sessions × intra_threads`
/// stays at roughly the core count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreadPlan {
    /// Sessions held in the pool, so that many detections can be in flight.
    sessions: usize,
    /// Intra-op threads each session gets. `None` leaves ONNX Runtime's own
    /// default alone, which is what a GPU provider wants.
    intra_threads: Option<usize>,
}

fn thread_plan(provider: ExecProvider) -> ThreadPlan {
    // One session for a GPU: the work is already parallel on the device, a
    // second would duplicate the model in VRAM for nothing, and the host-side
    // thread count is not what limits it.
    if provider.is_gpu() {
        return ThreadPlan {
            sessions: 1,
            intra_threads: None,
        };
    }
    // Physical cores, deliberately not logical ones. Inference is compute
    // bound, so the two hyperthreads sharing a core contend for the same
    // execution units rather than adding throughput: budgeting from the
    // logical count handed every session twice the threads it wanted and was
    // measurably slower than doing nothing at all.
    let cores = num_cpus::get_physical().max(1);
    // Overridable because "how much of this machine may it have" is a question
    // about the machine, not about the model — someone sharing a build box or
    // pinning a container needs to answer it themselves.
    let sessions = env_usize("OBSCURA_SESSIONS")
        .unwrap_or(DEFAULT_CPU_SESSIONS)
        .clamp(1, cores);
    // Deliberately less than the whole machine. Inference is not all a run
    // does: every image is also decoded and re-encoded, and those happen on the
    // batch engine's own workers at the same time. Budgeting every core to the
    // model starved that half of the pipeline — the untiled case, which is
    // mostly decode, got *slower* — so about a third of the cores are left for
    // it. Measured on a six-core machine; the overrides exist because the right
    // split depends on the machine and on what else is running.
    let intra = env_usize("OBSCURA_INTRA_THREADS").unwrap_or((cores / 3).max(1));
    ThreadPlan {
        sessions,
        intra_threads: Some(intra),
    }
}

/// A positive `usize` from the environment, or `None`.
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Concurrent CPU sessions by default.
///
/// Two, by measurement rather than by argument — see `docs/PERFORMANCE.md`. A
/// detection model this small parallelises poorly across many intra-op threads,
/// so a second narrow session beats one wide one; a third and fourth did not
/// help and eventually hurt, because a worker waiting for a session is a worker
/// not decoding the next image. Clamped to the physical core count so a
/// single-core machine gets one.
const DEFAULT_CPU_SESSIONS: usize = 2;

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

/// Build one more session on a provider already known to work.
///
/// Used to fill the pool after [`build_session`] has settled which provider
/// takes. Repeating the probing for each would mean rebuilding and discarding a
/// session per failed candidate per pool slot, for an answer already known.
fn build_session_on(
    model_path: &PathBuf,
    provider: ExecProvider,
    intra_threads: Option<usize>,
) -> Result<Session, String> {
    let dispatch = execution_provider_dispatches()
        .into_iter()
        .find(|(p, _)| *p == provider)
        .map(|(_, d)| d)
        .ok_or_else(|| format!("{provider} is not compiled into this build"))?;
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| match intra_threads {
            Some(n) => b.with_intra_threads(n),
            None => Ok(b),
        })
        .and_then(|b| b.with_execution_providers([dispatch]))
        .and_then(|b| b.commit_from_file(model_path))
        .map_err(|e| e.to_string())
}

/// Read what the model's graph declares about its input: the square side, and
/// whether its batch axis is dynamic.
///
/// A YOLOv8 export usually bakes a fixed `[1, 3, H, W]` input shape in, and a
/// registry entry stating a different `input_size` would either fail the run
/// outright or feed the model a resolution it was never trained for. Dynamic
/// axes come back as `-1`, in which case there is nothing to learn and the
/// registry's value stands.
fn declared_input_shape(session: &Session) -> DeclaredInput {
    let mut out = DeclaredInput {
        side: None,
        batchable: false,
    };
    let Some(input) = session.inputs.first() else {
        return out;
    };
    let ort::value::ValueType::Tensor { shape, .. } = &input.input_type else {
        return out;
    };
    if shape.len() != 4 {
        return out;
    }
    let (h, w) = (shape[2], shape[3]);
    if h > 0 && w > 0 && h == w {
        out.side = Some(h as u32);
    }
    // A symbolic or unspecified axis comes back non-positive, and that is the
    // only case where feeding the graph more than one window at a time is
    // sound. An export that pinned `batch` to a number — common enough in older
    // YOLO exports — would fail the run outright, so it takes the per-window
    // path instead.
    out.batchable = shape[0] <= 0;
    out
}

/// What the model's graph says about its own input tensor.
struct DeclaredInput {
    /// The square input side, when the graph fixes one.
    side: Option<u32>,
    /// Whether the batch axis is dynamic, so several windows can go in one run.
    batchable: bool,
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

        let (session, execution_provider, mut ep_notes) = build_session(entry, &model_path)?;
        // YOLOv8 exports use a single image input; read its real name so we bind
        // by name rather than assuming "images".
        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "images".to_string());

        // Trust the model file over the registry: a mis-stated input_size would
        // otherwise surface as an opaque ORT shape error at the first run.
        let declared = declared_input_shape(&session);
        let input_size = declared.side.unwrap_or(entry.input_size);
        let batch_limit = if declared.batchable {
            env_usize("OBSCURA_MAX_BATCH")
                .unwrap_or(MAX_BATCH)
                .min(BATCH_CEILING)
        } else {
            1
        };

        let resampler = settings
            .get("resample")
            .and_then(|v| v.as_str())
            .and_then(Resampler::parse)
            .unwrap_or_default();

        // Fill the pool on the provider that already took.
        //
        // The probing session is kept only when the plan asks for ONNX
        // Runtime's default threading, because that is what it was built with.
        // Where the plan divides the budget it has to be rebuilt: a session's
        // thread count is fixed at construction, and keeping one wide session
        // alongside three narrow ones is the oversubscription this is here to
        // avoid, in miniature.
        let plan = thread_plan(execution_provider);
        let mut sessions = Vec::with_capacity(plan.sessions);
        if plan.intra_threads.is_none() {
            sessions.push(session);
        } else {
            drop(session);
        }
        while sessions.len() < plan.sessions {
            match build_session_on(&model_path, execution_provider, plan.intra_threads) {
                Ok(extra) => sessions.push(extra),
                Err(e) => {
                    // Not fatal: a machine short of memory simply gets a
                    // smaller pool. Worth saying, though — otherwise the only
                    // symptom is a run quietly less parallel than the same
                    // build on another machine.
                    ep_notes.push(format!(
                        "only {} concurrent {execution_provider} session(s): {e}",
                        sessions.len()
                    ));
                    break;
                }
            }
        }
        if sessions.is_empty() {
            return Err(DetectError::Load(
                entry.id.to_string(),
                "no usable session could be built".into(),
            ));
        }

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
            batch_limit,
            pool: SessionPool::new(sessions),
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

    /// Run the model on `n` letterboxed windows packed into one CHW tensor and
    /// decode each one into detections in *model* coordinates.
    ///
    /// This is the only function that feeds data through `ort`. It takes the
    /// input by value because the caller built it and has no further use for
    /// it — the previous signature borrowed a slice and immediately copied it,
    /// which at 640px is a five-megabyte memcpy per pass for nothing.
    ///
    /// Decoding happens while the session is still checked out, so the output
    /// tensor is read in place rather than copied out first. Decode is a few
    /// tens of thousands of reads; with a pool of sessions, holding one a
    /// fraction longer no longer blocks the other workers.
    fn run_batch(&self, input: Vec<f32>, n: usize) -> Result<Vec<Vec<Detection>>, DetectError> {
        let size = self.input_size as usize;
        let tensor = Tensor::from_array(([n, 3, size, size], input))
            .map_err(|e| DetectError::Inference(e.to_string()))?;

        // `Session::run` needs `&mut`, hence the lease; it goes back to the
        // pool when this scope ends, error paths included.
        let mut session = self.pool.checkout()?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| DetectError::Inference(e.to_string()))?;

        // A YOLOv8 detect graph has a single output; take the first.
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectError::Inference(e.to_string()))?;
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        Ok((0..n).map(|b| self.decode_yolov8(data, &dims, b)).collect())
    }

    /// Decode batch element `b` of a YOLOv8 detect output `[B, 4+C, N]` into
    /// detections in *model* coordinates, before NMS and letterbox inversion.
    fn decode_yolov8(&self, output: &[f32], shape: &[usize], b: usize) -> Vec<Detection> {
        // Expect [B, 4+C, N].
        if shape.len() != 3 || b >= shape[0] {
            return Vec::new();
        }
        let channels = shape[1];
        let n = shape[2];
        let num_classes = channels.saturating_sub(4);
        let mut dets = Vec::new();
        // Output is channel-major within each batch element:
        // value(b, c, i) = output[b * channels * n + c * n + i].
        let base = b * channels * n;
        let at = |c: usize, i: usize| output[base + c * n + i];
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
        // One window covering everything, then the per-window NMS a single pass
        // has always done. `detect_regions` deliberately leaves suppression to
        // its caller, because a tile grid has to be merged globally instead.
        let dets = self.detect_regions(frame, &[Region::whole(frame)])?;
        Ok(nms(dets, self.nms_iou))
    }

    /// The batched path: every window letterboxed into one tensor and pushed
    /// through the graph in as few runs as `batch_limit` allows.
    fn detect_regions(
        &self,
        frame: &Frame,
        windows: &[Region],
    ) -> Result<Vec<Detection>, DetectError> {
        // Clip first, so a window hanging off an edge contributes the part that
        // exists and the transform describes what was actually resampled.
        let windows: Vec<Region> = windows
            .iter()
            .filter_map(|w| w.clamped(frame.width, frame.height))
            .collect();
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let mut all = Vec::new();
        for chunk in windows.chunks(self.batch_limit.max(1)) {
            let (input, boxes) = letterbox_batch(frame, chunk, self.input_size, self.resampler);
            let per_window = self.run_batch(input, chunk.len())?;

            for ((dets, lb), w) in per_window.into_iter().zip(&boxes).zip(chunk) {
                all.extend(dets.into_iter().map(|mut d| {
                    // Model space back to window space, clamped to the window —
                    // a box the model pushed past the edge of a tile is not
                    // evidence about the frame beyond it — then offset into
                    // frame coordinates.
                    d.bbox = lb.invert(&d.bbox);
                    d.bbox.x1 = d.bbox.x1.clamp(0.0, w.width_f()) + w.x as f32;
                    d.bbox.y1 = d.bbox.y1.clamp(0.0, w.height_f()) + w.y as f32;
                    d.bbox.x2 = d.bbox.x2.clamp(0.0, w.width_f()) + w.x as f32;
                    d.bbox.y2 = d.bbox.y2.clamp(0.0, w.height_f()) + w.y as f32;
                    d
                }));
            }
        }
        Ok(all)
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
