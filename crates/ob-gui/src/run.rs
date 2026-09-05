//! Background batch runs and previews.
//!
//! Both do inference and must never touch the UI thread: a single 4K tiled
//! frame is seconds of work, and a batch is minutes to hours. Each spawns a
//! worker that reports over a channel; the app drains it once per frame.

use ob_core::cancel::CancelToken;
use ob_core::profile::Profile;
use ob_core::settings::SettingValues;
use ob_detect::Detector;
use ob_job::estimate::{Calibration, ProbedItem, Workload};
use ob_job::expand::InputSpec;
use ob_job::{JobConfig, PreviewPair, PreviewSource, ProgressEvent};
use ob_media::video::VideoEncodeOpts;
use ob_track::TrackConfig;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Instant;

/// How many finished-file lines the activity log keeps.
///
/// A batch of 50 000 images would otherwise grow an unbounded `Vec` of paths
/// in a long-running window. The tail is what anyone actually reads.
const LOG_CAPACITY: usize = 400;

/// Build a detector for `model_id`, loading the weights from the model cache.
///
/// Deliberately does **not** download: the GUI's model page owns that, so a
/// missing model surfaces as "go and get it" rather than as a surprise
/// hundred-megabyte transfer at the moment the user pressed Run.
pub fn build_detector(
    model_id: &str,
    settings: &SettingValues,
) -> Result<Box<dyn Detector>, String> {
    let entry =
        ob_core::registry::find(model_id).ok_or_else(|| format!("unknown model `{model_id}`"))?;
    let path = ob_models::require(&entry).map_err(|e| e.to_string())?;
    ob_detect::build_detector(&entry, settings, path).map_err(|e| e.to_string())
}

/// IoU at which two members' boxes are treated as the same region. Matches the
/// CLI's `build_ensemble` — the same models voting the same way must produce
/// the same output whichever front end drove them.
const ENSEMBLE_NMS_IOU: f32 = 0.45;

/// Which models a run or preview will use.
///
/// The GUI's counterpart to `--model` + `--also-model` + `--min-votes`. It is
/// one value rather than three loose parameters so the preview can fingerprint
/// it in a single step, and so `effective_votes` cannot be applied in one place
/// and forgotten in another.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct EnsembleSpec {
    pub primary: String,
    /// Companions, by registry id. Never contains `primary`.
    pub extras: Vec<String>,
    pub min_votes: usize,
}

impl EnsembleSpec {
    /// Read the spec out of the saved preferences, dropping a companion that
    /// duplicates the primary — selecting a model as its own corroborator would
    /// otherwise let one model agree with itself.
    pub fn from_prefs(prefs: &crate::prefs::Prefs) -> Self {
        let primary = prefs.profile.model_id.clone();
        let extras = prefs
            .extra_models
            .iter()
            .filter(|id| **id != primary)
            .cloned()
            .collect();
        Self {
            primary,
            extras,
            min_votes: prefs.min_votes,
        }
    }

    pub fn members(&self) -> usize {
        1 + self.extras.len()
    }

    /// The threshold actually applied, clamped to the number of members.
    ///
    /// The CLI rejects an impossible `--min-votes` because a mistyped flag
    /// should not run as something else. A GUI has no equivalent moment to
    /// refuse at: the count changes when a checkbox is cleared, and failing the
    /// run because a stale number outlives the model it referred to would be
    /// obstruction rather than safety. The control is bounded to the same range
    /// on screen, so this clamp only ever catches a saved file whose model list
    /// shrank.
    pub fn effective_votes(&self) -> usize {
        self.min_votes.clamp(1, self.members())
    }

    /// True when this actually needs the ensemble machinery.
    pub fn is_ensemble(&self) -> bool {
        !self.extras.is_empty()
    }

    /// Load every member and wrap them in an ensemble.
    ///
    /// A single model is returned unwrapped rather than as a one-member
    /// ensemble: it keeps the common case free of the extra vote counting, and
    /// an ensemble of one is a no-op anyway.
    pub fn build(&self, settings: &SettingValues) -> Result<Box<dyn Detector>, String> {
        let primary = build_detector(&self.primary, settings)?;
        if self.extras.is_empty() {
            return Ok(primary);
        }
        let mut members = vec![primary];
        for id in &self.extras {
            // `settings` was edited against the primary's metadata;
            // `ob_detect::build_detector` layers it per model and keeps only
            // the keys a companion declares, so each keeps its own published
            // threshold unless the key is genuinely shared.
            members.push(build_detector(id, settings)?);
        }
        Ok(Box::new(ob_detect::ensemble::EnsembleDetector::new(
            members,
            self.effective_votes(),
            ENSEMBLE_NMS_IOU,
        )))
    }
}

/// One line in the activity log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub path: PathBuf,
    /// `None` for success, the message for a failure.
    pub error: Option<String>,
    /// Regions censored, for successful files.
    pub regions: usize,
}

impl LogEntry {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// Live tally and log of a running batch.
#[derive(Default)]
pub struct RunState {
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub current: Option<PathBuf>,
    pub log: VecDeque<LogEntry>,
    pub started: Option<Instant>,
    /// Set when the run ends; drives the summary line and the "Open output"
    /// button.
    pub finished: Option<Finished>,
    /// The measured batch, when a scan finished before the run started.
    /// Progress and ETA fall back to counting files without it.
    pub workload: Option<Arc<Workload>>,
    /// Work units completed, by the scan's reckoning.
    pub done_work: f64,
    /// Seconds per work unit — seeded from the last run on this machine and
    /// refined as this one proceeds.
    pub calibration: Calibration,
}

/// How a run ended.
#[derive(Debug, Clone)]
pub struct Finished {
    pub ok: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub elapsed_secs: f64,
}

impl RunState {
    fn push_log(&mut self, entry: LogEntry) {
        if self.log.len() >= LOG_CAPACITY {
            self.log.pop_front();
        }
        self.log.push_back(entry);
    }

    /// Completed fraction, or `None` before the input scan reports a total.
    ///
    /// Weighted by measured work when the batch was scanned. Counting files
    /// makes a bar that stalls on the 4K video and then leaps through the
    /// thumbnails; counting passes makes it move at something like a constant
    /// rate.
    pub fn fraction(&self) -> Option<f32> {
        if let Some(w) = self.workload.as_ref().filter(|w| !w.is_empty()) {
            return Some((self.done_work / w.total_work).clamp(0.0, 1.0) as f32);
        }
        if self.total == 0 {
            return None;
        }
        Some((self.done as f32 / self.total as f32).clamp(0.0, 1.0))
    }

    /// Work units left to do, when the batch was measured.
    pub fn remaining_work(&self) -> Option<f64> {
        let w = self.workload.as_ref().filter(|w| !w.is_empty())?;
        Some((w.total_work - self.done_work).max(0.0))
    }

    /// Seconds remaining.
    ///
    /// With a scan this is available from the first frame of the run — the
    /// point at which someone deciding whether to leave it going overnight
    /// actually wants it — and is weighted by what each remaining file costs
    /// rather than by how many are left.
    pub fn eta_secs(&self) -> Option<f64> {
        if let Some(remaining) = self.remaining_work() {
            return Some(self.calibration.secs_for(remaining));
        }
        // Unmeasured batch: the old count-based extrapolation, which needs at
        // least one completed file before it can say anything.
        let started = self.started?;
        if self.done == 0 || self.total == 0 {
            return None;
        }
        let elapsed = started.elapsed().as_secs_f64();
        let per_item = elapsed / self.done as f64;
        let remaining = self.total.saturating_sub(self.done) as f64;
        Some(per_item * remaining)
    }

    /// Re-derive the rate from everything finished so far.
    ///
    /// Called as files land, so a batch whose first estimate was built on the
    /// previous machine's rate — or on the built-in default — converges on the
    /// truth within the first few files instead of staying wrong all run.
    fn recalibrate(&mut self) {
        let (Some(started), true) = (self.started, self.done_work > 0.0) else {
            return;
        };
        self.calibration
            .observe(self.done_work, started.elapsed().as_secs_f64());
    }

    /// Credit `path`'s work as done. Unknown paths (an unscanned batch, or a
    /// file that appeared after the scan) simply do not move the weighted bar.
    fn credit(&mut self, path: &Path) {
        if let Some(w) = self.workload.as_ref().and_then(|w| w.work_for(path)) {
            self.done_work += w;
        }
        self.recalibrate();
    }

    pub fn error_count(&self) -> usize {
        self.log.iter().filter(|e| e.is_error()).count()
    }
}

/// A batch run in flight.
pub struct RunHandle {
    rx: Receiver<ProgressEvent>,
    pub cancel: CancelToken,
    /// Where output is being written, for the "Open folder" action.
    pub output_dir: PathBuf,
}

impl RunHandle {
    /// Spawn a batch over `inputs`, writing into `output_dir`.
    ///
    /// The detector is built **inside** the worker so a slow model load (an
    /// ONNX session on a large model takes a moment) does not freeze the
    /// window, and so a load failure arrives as an ordinary job error.
    pub fn spawn(
        profile: &Profile,
        settings: &SettingValues,
        spec: &EnsembleSpec,
        inputs: Vec<PathBuf>,
        output_dir: PathBuf,
        detect_every: u32,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelToken::new();

        let profile = profile.clone();
        let settings = settings.clone();
        let spec = spec.clone();
        let out = output_dir.clone();
        let worker_cancel = cancel.clone();

        std::thread::spawn(move || {
            // `ob_job::run` needs a `Sync` progress closure because images are
            // processed in parallel; a Mutex around the Sender provides that.
            let send = std::sync::Mutex::new(tx);
            let emit = |ev: ProgressEvent| {
                let _ = send.lock().unwrap_or_else(|e| e.into_inner()).send(ev);
            };

            let detector = match spec.build(&settings) {
                Ok(d) => d,
                Err(e) => {
                    emit(ProgressEvent::FileError {
                        path: PathBuf::from(&spec.primary),
                        error: e,
                    });
                    emit(ProgressEvent::Finished { ok: 0, failed: 1 });
                    return;
                }
            };

            let cfg = JobConfig {
                profile: &profile,
                input: InputSpec {
                    inputs,
                    recursive: true,
                    include: Vec::new(),
                    exclude: Vec::new(),
                },
                output_dir: out,
                dry_run: false,
                detect_every,
                track: TrackConfig::default(),
                video_opts: VideoEncodeOpts::default(),
                cancel: worker_cancel,
            };
            if let Err(e) = ob_job::run(&cfg, detector.as_ref(), &emit) {
                // An error out of `run` itself (input expansion, mostly) never
                // reaches the per-file channel, so report it explicitly.
                emit(ProgressEvent::FileError {
                    path: PathBuf::new(),
                    error: e.to_string(),
                });
                emit(ProgressEvent::Finished { ok: 0, failed: 1 });
            }
        });

        Self {
            rx,
            cancel,
            output_dir,
        }
    }

    /// Drain pending events into `state`. Returns false once the worker is
    /// gone, which is the app's signal to drop this handle.
    pub fn pump(&self, state: &mut RunState) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(ev) => apply_event(ev, state),
                Err(std::sync::mpsc::TryRecvError::Empty) => return true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return false,
            }
        }
    }
}

fn apply_event(ev: ProgressEvent, state: &mut RunState) {
    match ev {
        ProgressEvent::Discovered(n) => {
            state.total = n;
            state.started = Some(Instant::now());
        }
        ProgressEvent::FileStarted(p) => state.current = Some(p),
        ProgressEvent::FileDone { path, regions } => {
            state.done += 1;
            state.credit(&path);
            state.push_log(LogEntry {
                path,
                error: None,
                regions,
            });
        }
        ProgressEvent::FileError { path, error } => {
            state.done += 1;
            state.failed += 1;
            // A file that failed still consumed most of its cost getting there,
            // and leaving it uncredited would strand the bar short of 100%.
            state.credit(&path);
            state.push_log(LogEntry {
                path,
                error: Some(error),
                regions: 0,
            });
        }
        ProgressEvent::Cancelled { .. } => {
            // Recorded on the Finished event, which always follows; setting a
            // flag here keeps the two in one place.
            state.current = None;
            state.finished = Some(Finished {
                ok: state.done - state.failed,
                failed: state.failed,
                cancelled: true,
                elapsed_secs: state
                    .started
                    .map(|s| s.elapsed().as_secs_f64())
                    .unwrap_or(0.0),
            });
        }
        ProgressEvent::Finished { ok, failed } => {
            state.current = None;
            let cancelled = state.finished.as_ref().is_some_and(|f| f.cancelled);
            state.finished = Some(Finished {
                ok,
                failed,
                cancelled,
                elapsed_secs: state
                    .started
                    .map(|s| s.elapsed().as_secs_f64())
                    .unwrap_or(0.0),
            });
        }
    }
}

/// A batch being measured on a worker thread.
///
/// Probing is I/O over every file in the batch — a header read each for images,
/// an `ffprobe` subprocess each for videos — so it never runs on the UI thread
/// and never blocks the run. It is started when the input list changes, so by
/// the time anyone presses Run the measurement is usually already in hand.
///
/// Only the probe is backgrounded. Re-costing those measurements under new
/// settings is pure arithmetic and happens inline.
pub struct EstimateJob {
    rx: Receiver<Vec<ProbedItem>>,
    cancel: CancelToken,
}

impl EstimateJob {
    /// Expand `inputs` and measure everything under them.
    pub fn spawn(inputs: Vec<PathBuf>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelToken::new();
        let worker_cancel = cancel.clone();
        std::thread::spawn(move || {
            // The same expansion the run will do, so the scan measures exactly
            // the files the batch will process.
            let spec = InputSpec {
                inputs,
                recursive: true,
                include: Vec::new(),
                exclude: Vec::new(),
            };
            let items = ob_job::expand::expand(&spec).unwrap_or_default();
            let _ = tx.send(ob_job::estimate::probe(&items, &worker_cancel));
        });
        Self { rx, cancel }
    }

    /// `None` while still working; `Some` exactly once, when it finishes.
    pub fn poll(&self) -> Option<Vec<ProbedItem>> {
        self.rx.try_recv().ok()
    }

    /// Abandon a scan whose result is already stale.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// A finished preview.
pub struct PreviewDone {
    pub pair: PreviewPair,
    /// The detections this preview was built from, present only when the job
    /// actually ran the detector. The app keeps it so the next change to a
    /// censor style can be answered without inference.
    pub source: Option<Arc<PreviewSource>>,
}

/// A preview being rendered on a worker thread.
///
/// Two flavours, because the two halves of a preview cost wildly different
/// amounts. [`PreviewJob::detect`] loads the model and runs inference —
/// hundreds of milliseconds at best, seconds on a tiled 4K frame.
/// [`PreviewJob::compose`] only re-paints boxes onto a frame that was already
/// detected, which is fast enough to keep up with a dragged slider. Both still
/// run off the UI thread: even compositing a 4K frame with a large blur sigma
/// is more than one frame's budget.
pub struct PreviewJob {
    rx: Receiver<Result<PreviewDone, String>>,
    /// True for the inference flavour — the UI says "detecting…" rather than
    /// "rendering…", because only one of the two is worth waiting through.
    pub is_detect: bool,
}

impl PreviewJob {
    /// Decode, run the detector, and compose. Use when the model, its settings
    /// or the source file changed.
    pub fn detect(
        input: PathBuf,
        profile: &Profile,
        settings: &SettingValues,
        spec: &EnsembleSpec,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let profile = profile.clone();
        let settings = settings.clone();
        let spec = spec.clone();
        std::thread::spawn(move || {
            // The preview must agree with the batch, so it builds the same
            // ensemble rather than the primary alone -- corroboration changes
            // which regions survive, which is exactly what is being tuned.
            let result = spec
                .build(&settings)
                .and_then(|d| {
                    ob_job::preview_detect(&input, d.as_ref()).map_err(|e| e.to_string())
                })
                .and_then(|src| {
                    let pair =
                        ob_job::preview_compose(&src, &profile).map_err(|e| e.to_string())?;
                    Ok(PreviewDone {
                        pair,
                        source: Some(Arc::new(src)),
                    })
                });
            let _ = tx.send(result);
        });
        Self {
            rx,
            is_detect: true,
        }
    }

    /// Re-paint an already-detected frame under a new profile. Use when only
    /// the filter, the censor styles or the fail-closed policy changed.
    pub fn compose(source: Arc<PreviewSource>, profile: &Profile) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let profile = profile.clone();
        std::thread::spawn(move || {
            let result = ob_job::preview_compose(&source, &profile)
                .map_err(|e| e.to_string())
                .map(|pair| PreviewDone { pair, source: None });
            let _ = tx.send(result);
        });
        Self {
            rx,
            is_detect: false,
        }
    }

    /// `None` while still working; `Some` exactly once, when it finishes.
    pub fn poll(&self) -> Option<Result<PreviewDone, String>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_log_is_bounded() {
        let mut s = RunState::default();
        for i in 0..(LOG_CAPACITY + 50) {
            apply_event(
                ProgressEvent::FileDone {
                    path: PathBuf::from(format!("{i}.png")),
                    regions: 0,
                },
                &mut s,
            );
        }
        // A 50k-image batch must not grow an unbounded Vec of paths.
        assert_eq!(s.log.len(), LOG_CAPACITY);
        // It is the tail that is kept — the oldest entries are dropped.
        assert_eq!(
            s.log.back().unwrap().path,
            PathBuf::from(format!("{}.png", LOG_CAPACITY + 49))
        );
        assert_eq!(s.done, LOG_CAPACITY + 50);
    }

    #[test]
    fn progress_fraction_waits_for_a_total() {
        let mut s = RunState::default();
        assert_eq!(s.fraction(), None);
        apply_event(ProgressEvent::Discovered(4), &mut s);
        assert_eq!(s.fraction(), Some(0.0));
        apply_event(
            ProgressEvent::FileDone {
                path: "a".into(),
                regions: 1,
            },
            &mut s,
        );
        assert_eq!(s.fraction(), Some(0.25));
    }

    #[test]
    fn errors_are_counted_and_kept_in_the_log() {
        let mut s = RunState::default();
        apply_event(ProgressEvent::Discovered(2), &mut s);
        apply_event(
            ProgressEvent::FileError {
                path: "bad.png".into(),
                error: "boom".into(),
            },
            &mut s,
        );
        apply_event(
            ProgressEvent::FileDone {
                path: "good.png".into(),
                regions: 2,
            },
            &mut s,
        );
        assert_eq!(s.failed, 1);
        assert_eq!(s.done, 2);
        assert_eq!(s.error_count(), 1);
        assert!(s.log[0].is_error());
        assert!(!s.log[1].is_error());
    }

    #[test]
    fn a_cancelled_run_stays_marked_cancelled_after_finished() {
        let mut s = RunState::default();
        apply_event(ProgressEvent::Discovered(5), &mut s);
        apply_event(ProgressEvent::Cancelled { remaining: 4 }, &mut s);
        apply_event(ProgressEvent::Finished { ok: 1, failed: 0 }, &mut s);
        // Finished arrives *after* Cancelled and must not overwrite the flag,
        // or the UI would report a stopped run as a clean completion.
        let f = s.finished.expect("finished state");
        assert!(f.cancelled);
        assert_eq!(f.ok, 1);
    }

    fn workload(items: &[(&str, f64)]) -> Arc<Workload> {
        use ob_job::estimate::{ItemCost, Sizing};
        let mut w = Workload::default();
        for (path, work) in items {
            w.total_work += work;
            w.images += 1;
            w.items.push(ItemCost {
                path: PathBuf::from(path),
                kind: ob_media::MediaKind::Image,
                work: *work,
                sizing: Sizing::Probed,
            });
        }
        Arc::new(w)
    }

    /// A measured batch: one 4K video-sized item and one thumbnail.
    fn measured_state() -> RunState {
        let mut s = RunState {
            workload: Some(workload(&[("big.png", 900.0), ("small.png", 100.0)])),
            calibration: Calibration::from_saved(Some(0.01)),
            ..RunState::default()
        };
        apply_event(ProgressEvent::Discovered(2), &mut s);
        s
    }

    #[test]
    fn progress_is_weighted_by_measured_work_not_file_count() {
        let mut s = measured_state();
        apply_event(
            ProgressEvent::FileDone {
                path: "small.png".into(),
                regions: 0,
            },
            &mut s,
        );
        // Half the files, but a tenth of the work. Counting files here would
        // show 50% and then appear to stall for the rest of the run.
        assert_eq!(s.done, 1);
        assert!((s.fraction().unwrap() - 0.1).abs() < 1e-6);

        apply_event(
            ProgressEvent::FileDone {
                path: "big.png".into(),
                regions: 0,
            },
            &mut s,
        );
        assert!((s.fraction().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_measured_batch_has_an_eta_before_anything_finishes() {
        let s = measured_state();
        // The old estimator said nothing until the first file landed, which on
        // a batch of long video is exactly when the answer matters most.
        assert_eq!(s.done, 0);
        let eta = s.eta_secs().expect("an eta from the scan alone");
        assert!((eta - 10.0).abs() < 1e-6, "1000 units at 0.01 s/unit");
    }

    #[test]
    fn a_failed_file_still_counts_towards_progress() {
        let mut s = measured_state();
        apply_event(
            ProgressEvent::FileError {
                path: "big.png".into(),
                error: "boom".into(),
            },
            &mut s,
        );
        // It consumed most of its cost getting to the failure; leaving it
        // uncredited would strand the bar short of 100% for the whole run.
        assert!((s.fraction().unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn a_file_that_was_not_scanned_does_not_move_the_weighted_bar() {
        let mut s = measured_state();
        apply_event(
            ProgressEvent::FileDone {
                path: "appeared-later.png".into(),
                regions: 0,
            },
            &mut s,
        );
        assert_eq!(s.fraction(), Some(0.0));
        assert_eq!(s.done, 1);
    }

    #[test]
    fn an_unmeasured_batch_falls_back_to_counting_files() {
        let mut s = RunState::default();
        apply_event(ProgressEvent::Discovered(4), &mut s);
        // No workload: the old behaviour, including having nothing to say
        // until something completes.
        assert_eq!(s.eta_secs(), None);
        apply_event(
            ProgressEvent::FileDone {
                path: "a.png".into(),
                regions: 0,
            },
            &mut s,
        );
        assert_eq!(s.fraction(), Some(0.25));
        assert!(s.eta_secs().is_some());
    }

    #[test]
    fn the_rate_is_relearned_from_the_run_in_progress() {
        let mut s = measured_state();
        // Seeded at 0.01 s/unit from a previous run.
        assert!((s.calibration.secs_per_unit() - 0.01).abs() < 1e-9);
        // Pretend the run started a while ago, then finish 100 units of work.
        s.started = Some(Instant::now() - Duration::from_secs(10));
        apply_event(
            ProgressEvent::FileDone {
                path: "small.png".into(),
                regions: 0,
            },
            &mut s,
        );
        // 100 units took ~10s, so this machine is running ~10x slower than the
        // saved rate claimed. An estimate that never revisited its seed would
        // stay wrong for the whole batch.
        let rate = s.calibration.secs_per_unit();
        assert!(
            (rate - 0.1).abs() < 0.02,
            "expected ~0.1 s/unit, learned {rate}"
        );
        assert!(s.calibration.is_measured());
    }

    #[test]
    fn eta_needs_at_least_one_completed_item() {
        let mut s = RunState::default();
        apply_event(ProgressEvent::Discovered(10), &mut s);
        assert_eq!(s.eta_secs(), None);
        apply_event(
            ProgressEvent::FileDone {
                path: "a".into(),
                regions: 0,
            },
            &mut s,
        );
        assert!(s.eta_secs().is_some());
    }
}
