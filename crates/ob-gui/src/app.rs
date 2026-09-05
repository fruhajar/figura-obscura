//! The application shell: state, the nav rail, the status bar, and the frame
//! loop that drives the pages.
//!
//! The window is a left nav rail, a page, and a persistent status bar carrying
//! the primary action. Everything slow — inference, downloads, video encoding —
//! happens on worker threads (`run`, `downloads`); `update` only drains their
//! channels and draws.

use crate::downloads::Downloads;
use crate::prefs::{Prefs, Tab};
use crate::run::{EstimateJob, PreviewJob, RunHandle, RunState};
use ob_detect::tile::{TilingConfig, TilingMode};
use ob_job::estimate::{Calibration, CostModel, ProbedItem, Workload};
use crate::theme;
use egui::{CentralPanel, RichText, SidePanel, TopBottomPanel};
use ob_core::registry::ModelEntry;
use ob_core::settings::SettingValues;
use ob_media::tools::Tool;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a toast stays on screen.
const TOAST_LIFETIME: Duration = Duration::from_secs(6);

/// How long the settings must hold still before the live preview re-renders.
///
/// Re-armed on every change, so a dragged slider renders once when the user
/// stops rather than once per pixel of travel. Short enough that letting go of
/// a control and looking at the panel feels like it kept up.
const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(250);

/// A fingerprint of everything a preview depends on, split by what it costs to
/// react to.
///
/// `detect` covers the source file, the model and its settings — changing any
/// of those means running inference again. `compose` covers the filter, the
/// censor styles and the fail-closed policy, which only re-paint an already
/// detected frame. Keeping them apart is what makes tuning a censor style feel
/// instant while changing the confidence threshold honestly costs a re-detect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PreviewKeys {
    pub detect: u64,
    pub compose: u64,
}

/// Hash a serialisable value. Used only to notice change, never persisted, so
/// `DefaultHasher`'s lack of stability across builds does not matter.
fn fingerprint<T: serde::Serialize>(value: &T) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // A value that will not serialise cannot be previewed either; hashing the
    // empty string just means "unchanged", which is the safe direction.
    serde_json::to_string(value).unwrap_or_default().hash(&mut h);
    h.finish()
}

/// What the live preview should do about the settings as they stand.
#[derive(Debug, PartialEq, Eq)]
pub enum PreviewNext {
    /// Up to date, or waiting on something else.
    Idle,
    /// The settings are still moving — come back after this long.
    Wait(Duration),
    /// Start a job for these keys.
    Start(PreviewKeys),
}

/// The debounce behind the live preview.
///
/// Kept as pure state with the clock passed in, so the timing rules can be
/// tested without a model, a display server, or a worker thread — none of which
/// the build container has.
#[derive(Default)]
pub struct PreviewSchedule {
    /// Keys the last finished job was for. Failures count too, or an
    /// unpreviewable file would be retried on every tick forever.
    done: Option<PreviewKeys>,
    /// Keys the in-flight job will produce.
    pending: Option<PreviewKeys>,
    /// The keys last observed, and when they last changed.
    wanted: Option<PreviewKeys>,
    changed_at: Option<Instant>,
}

impl PreviewSchedule {
    /// Decide what to do, given the settings as they stand and whether a job is
    /// already running.
    pub fn poll(&mut self, keys: PreviewKeys, busy: bool, now: Instant) -> PreviewNext {
        // Re-arm on every change, so the render lands when the user pauses
        // rather than partway through a drag.
        if self.wanted != Some(keys) {
            self.wanted = Some(keys);
            self.changed_at = Some(now);
        }
        let settled =
            self.pending == Some(keys) || (self.pending.is_none() && self.done == Some(keys));
        if settled {
            self.changed_at = None;
            return PreviewNext::Idle;
        }
        let Some(changed_at) = self.changed_at else {
            return PreviewNext::Idle;
        };
        let waited = now.saturating_duration_since(changed_at);
        if waited < PREVIEW_DEBOUNCE {
            return PreviewNext::Wait(PREVIEW_DEBOUNCE - waited);
        }
        if busy {
            // The in-flight job is for something else. `changed_at` stays put,
            // so this fires as soon as that one lands.
            return PreviewNext::Idle;
        }
        PreviewNext::Start(keys)
    }

    pub fn started(&mut self, keys: Option<PreviewKeys>) {
        self.pending = keys;
        self.changed_at = None;
    }

    /// Record that the in-flight job finished, and return what it was for.
    pub fn finished(&mut self) -> Option<PreviewKeys> {
        self.done = self.pending;
        self.pending.take()
    }

    /// Forget what has been rendered, so the next poll re-renders. Used when
    /// something outside the fingerprint changed — a model was installed, or
    /// the user asked for a fresh look at a file that may have changed on disk.
    pub fn invalidate(&mut self) {
        self.done = None;
    }
}

/// A transient message in the status bar.
pub struct Toast {
    pub text: String,
    pub kind: ToastKind,
    pub at: Instant,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Success,
    Error,
}

impl ToastKind {
    pub fn color(self) -> egui::Color32 {
        let p = theme::palette();
        match self {
            ToastKind::Info => p.text_dim,
            ToastKind::Success => p.success,
            ToastKind::Error => p.danger,
        }
    }
}

/// A rendered preview, held as GPU textures.
pub struct PreviewView {
    pub original: egui::TextureHandle,
    pub censored: egui::TextureHandle,
    pub regions: usize,
}

pub struct ObApp {
    pub prefs: Prefs,
    /// Effective settings for the selected model (defaults layered with edits).
    pub settings: SettingValues,
    /// The registry, resolved once — it allocates a `Vec<Setting>` per entry,
    /// which is not something to redo sixty times a second.
    pub registry: Vec<ModelEntry>,

    pub inputs: Vec<PathBuf>,
    pub downloads: Downloads,

    pub run: Option<RunHandle>,
    pub run_state: RunState,
    /// Show only failures in the activity log.
    pub errors_only: bool,

    pub preview_job: Option<PreviewJob>,
    pub preview: Option<PreviewView>,
    pub preview_error: Option<String>,
    /// The file the preview renders. `None` means "the first input file", which
    /// is what most batches want; an explicit pick lets someone tune against a
    /// representative frame without adding it to the batch.
    pub preview_source_path: Option<PathBuf>,
    /// The last inference's detections, keyed by the `detect` fingerprint that
    /// produced them. Re-composing from this is what lets a censor style change
    /// render without running the model again.
    pub preview_cache: Option<(u64, Arc<ob_job::PreviewSource>)>,
    /// When the live preview should re-render.
    pub preview_schedule: PreviewSchedule,

    /// The batch as measured on disk: dimensions and lengths, independent of
    /// any setting. `None` until the first scan lands.
    pub probed: Option<Arc<Vec<ProbedItem>>>,
    /// A scan in flight.
    pub estimate_job: Option<EstimateJob>,
    /// Fingerprint of the input list the current scan (or result) is for, so a
    /// changed batch re-scans and an unchanged one never does.
    pub probed_key: Option<u64>,

    pub toast: Option<Toast>,
    /// ffmpeg/ffprobe that could not be run, checked once at startup.
    pub missing_tools: Vec<Tool>,
    /// True until first-run setup is finished or dismissed.
    pub show_setup: bool,
}

impl Default for ObApp {
    fn default() -> Self {
        let prefs = Prefs::load();
        let registry = ob_core::registry::builtin_registry();

        let mut downloads = Downloads::new();
        downloads.refresh_all(&registry);

        // A saved profile may name a model that no longer exists (a downgrade,
        // or a registry entry withdrawn); fall back rather than panicking.
        let model_id = if registry.iter().any(|m| m.id == prefs.profile.model_id) {
            prefs.profile.model_id.clone()
        } else {
            registry[0].id.to_string()
        };

        let entry = registry
            .iter()
            .find(|m| m.id == model_id)
            .expect("model_id was just resolved against the registry");
        // Layer any saved overrides on top of the model's declared defaults, so
        // a setting added since the profile was written picks up its default
        // instead of being absent.
        let mut settings = ob_core::settings::defaults(&entry.settings);
        for (k, v) in &prefs.profile.model_settings {
            if entry.settings.iter().any(|s| s.key == k.as_str()) {
                settings.insert(k.clone(), v.clone());
            }
        }

        let installed_any = registry.iter().any(|m| downloads.is_installed(m.id));
        let show_setup = !prefs.setup_done && !installed_any;

        let mut prefs = prefs;
        prefs.profile.model_id = model_id;

        Self {
            prefs,
            settings,
            registry,
            inputs: Vec::new(),
            downloads,
            run: None,
            run_state: RunState::default(),
            errors_only: false,
            preview_job: None,
            preview: None,
            preview_error: None,
            preview_source_path: None,
            preview_cache: None,
            preview_schedule: PreviewSchedule::default(),
            probed: None,
            estimate_job: None,
            probed_key: None,
            toast: None,
            missing_tools: ob_media::tools::missing_tools(),
            show_setup,
        }
    }
}

impl ObApp {
    /// The currently selected model's registry entry.
    ///
    /// Borrows from `self.registry`, so it must not be called while a page has
    /// moved the registry out (see [`Self::select_model`]). The assertion turns
    /// that mistake into a test failure rather than an index panic in front of
    /// a user; the page smoke tests exercise every page, so it would be caught.
    pub fn current_entry(&self) -> &ModelEntry {
        debug_assert!(
            !self.registry.is_empty(),
            "current_entry() called while the registry was moved out of the app"
        );
        self.registry
            .iter()
            .find(|m| m.id == self.prefs.profile.model_id)
            .unwrap_or(&self.registry[0])
    }

    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }

    /// Whether the selected model's weights are on disk.
    pub fn model_ready(&self) -> bool {
        self.downloads.is_installed(&self.prefs.profile.model_id)
    }

    pub fn toast(&mut self, text: impl Into<String>, kind: ToastKind) {
        self.toast = Some(Toast {
            text: text.into(),
            kind,
            at: Instant::now(),
        });
    }

    /// Switch models, resetting settings to the new model's defaults.
    ///
    /// A reset rather than a merge: thresholds are per-model and published
    /// per-model (0.15 vs 0.278 among the anime weights alone), so carrying a
    /// tuned value across would silently move the new model's operating point.
    ///
    /// Looks the model up in the *global* registry rather than in
    /// `self.registry`. The Models page moves its copy out of the app for the
    /// duration of its card loop, so that each card can take `&mut app`; a
    /// lookup against `self.registry` would find nothing there and this would
    /// silently do nothing — which is exactly what "Use this model" did before.
    pub fn select_model(&mut self, id: &str) {
        if self.prefs.profile.model_id == id {
            return;
        }
        let Some(entry) = ob_core::registry::find(id) else {
            return;
        };
        self.settings = ob_core::settings::defaults(&entry.settings);
        self.prefs.profile.model_id = id.to_string();
        self.prefs.profile.model_settings = self.settings.clone();
        // Promoting a companion to primary drops it from the companion list:
        // leaving it there would have one model corroborate itself, and it
        // would silently come back if the primary were switched away again.
        self.prefs.extra_models.retain(|x| x != id);
        // The old preview's detections came from a different model. The image
        // stays on screen — the auto pass replaces it in a moment, and a panel
        // that blanks on every model switch is harder to compare against.
        self.preview_cache = None;
        self.preview_schedule.invalidate();
        self.preview_error = None;
    }

    pub fn add_inputs(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if !self.inputs.contains(&p) {
                self.inputs.push(p);
            }
        }
    }

    /// Can a batch be started right now, and if not, why?
    pub fn run_blocker(&self) -> Option<String> {
        if self.is_running() {
            return Some("A batch is already running.".into());
        }
        if self.inputs.is_empty() {
            return Some("Add files or folders to the batch first.".into());
        }
        if self.prefs.output_dir.is_none() {
            return Some("Choose an output folder first.".into());
        }
        if !self.model_ready() {
            return Some(format!(
                "The model `{}` is not installed — get it on the Models page.",
                self.prefs.profile.model_id
            ));
        }
        None
    }

    pub fn start_run(&mut self) {
        if let Some(reason) = self.run_blocker() {
            self.toast(reason, ToastKind::Error);
            return;
        }
        let Some(output_dir) = self.prefs.output_dir.clone() else {
            return;
        };

        // Persist before a long job: if the machine goes down mid-batch, the
        // configuration that produced it is still on disk.
        self.sync_profile();
        let _ = self.prefs.save();

        // Hand the run the measurement and this machine's rate, so the ETA is
        // there from the first frame rather than after the first file.
        self.run_state = RunState {
            workload: self.workload().map(Arc::new),
            calibration: self.calibration(),
            ..RunState::default()
        };
        self.run = Some(RunHandle::spawn(
            &self.prefs.profile,
            &self.settings,
            &self.ensemble_spec(),
            self.inputs.clone(),
            output_dir,
            self.prefs.detect_every,
        ));
        self.prefs.tab = Tab::Queue;
    }

    pub fn cancel_run(&mut self) {
        if let Some(run) = &self.run {
            run.cancel.cancel();
            self.toast("Stopping after the current file…", ToastKind::Info);
        }
    }

    /// The file the preview renders: an explicit pick, else the first input
    /// that is actually a file.
    ///
    /// Directories are skipped rather than previewed-and-failed — a batch is
    /// very often a single folder, and "that is a directory" is a useless thing
    /// to tell someone who asked to see a preview.
    pub fn preview_path(&self) -> Option<PathBuf> {
        if let Some(p) = &self.preview_source_path {
            return Some(p.clone());
        }
        self.inputs.iter().find(|p| p.is_file()).cloned()
    }

    /// Fingerprint what the preview currently depends on. `None` when there is
    /// nothing to preview.
    pub fn preview_keys(&self) -> Option<PreviewKeys> {
        let path = self.preview_path()?;
        let p = &self.prefs.profile;
        Some(PreviewKeys {
            // The companions and the vote threshold belong to the detect half:
            // corroboration decides which regions exist at all, so changing it
            // has to cost an inference pass rather than a re-paint.
            detect: fingerprint(&(&path, &self.ensemble_spec(), &self.settings)),
            compose: fingerprint(&(&p.filter, &p.censor, p.on_detect_failure)),
        })
    }

    /// The models a run or preview will use, as configured.
    pub fn ensemble_spec(&self) -> crate::run::EnsembleSpec {
        crate::run::EnsembleSpec::from_prefs(&self.prefs)
    }

    /// What a frame costs under the current settings.
    ///
    /// Everything here is a live control: tiling and its bounds, how often
    /// video frames are analysed, and how many models are voting. Changing any
    /// of them changes the estimate immediately, without re-reading the batch.
    pub fn cost_model(&self) -> CostModel {
        let entry = self.current_entry();
        let num = |k: &str, d: f64| self.settings.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let text = |k: &str| self.settings.get(k).and_then(|v| v.as_str());
        let defaults = TilingConfig::default();
        CostModel {
            input_size: entry.input_size,
            tiling: TilingConfig {
                mode: text("tiling")
                    .and_then(TilingMode::parse)
                    .unwrap_or_default(),
                overlap: num("tile_overlap", defaults.overlap as f64) as f32,
                max_tiles: num("tile_max", defaults.max_tiles as f64).max(1.0) as usize,
                ..defaults
            },
            detect_every: self.prefs.detect_every,
            members: self.ensemble_spec().members(),
        }
    }

    /// The batch costed under the current settings, or `None` before the scan
    /// lands or when nothing in it could be measured.
    pub fn workload(&self) -> Option<Workload> {
        let probed = self.probed.as_ref()?;
        let w = ob_job::estimate::cost(probed, &self.cost_model());
        (!w.is_empty()).then_some(w)
    }

    /// The rate this machine last managed, or the built-in default.
    pub fn calibration(&self) -> Calibration {
        Calibration::from_saved(self.prefs.secs_per_unit)
    }

    /// Start or abandon a scan so the measurement matches the input list.
    ///
    /// Driven off a fingerprint rather than the add/remove call sites: files
    /// arrive by button, by drag-and-drop and by folder expansion, and a scan
    /// that quietly described the previous batch would be worse than none.
    fn pump_estimate(&mut self) {
        let key = fingerprint(&self.inputs);
        if self.probed_key != Some(key) {
            if let Some(job) = &self.estimate_job {
                job.cancel();
            }
            self.probed_key = Some(key);
            self.probed = None;
            self.estimate_job = if self.inputs.is_empty() {
                None
            } else {
                Some(EstimateJob::spawn(self.inputs.clone()))
            };
        }
        if let Some(items) = self.estimate_job.as_ref().and_then(|j| j.poll()) {
            self.probed = Some(Arc::new(items));
            self.estimate_job = None;
        }
    }

    /// Explicitly asked for: always re-runs inference.
    ///
    /// The automatic path reuses cached detections wherever it can, but a
    /// deliberate press means "look again" — the file on disk may have changed
    /// under us, and that is the only way to find out.
    pub fn start_preview(&mut self) {
        let Some(input) = self.preview_path() else {
            self.toast("Add or choose a file to preview.", ToastKind::Error);
            return;
        };
        if !self.model_ready() {
            self.toast(
                "Install the selected model before previewing.",
                ToastKind::Error,
            );
            return;
        }
        self.sync_profile();
        self.preview_cache = None;
        self.preview_schedule.invalidate();
        self.spawn_preview(input, self.preview_keys());
    }

    /// Start the cheapest job that can produce `keys`.
    fn spawn_preview(&mut self, input: PathBuf, keys: Option<PreviewKeys>) {
        self.preview_error = None;
        self.preview_schedule.started(keys);
        // Reuse the detections when only the appearance changed. This is the
        // whole reason tuning a censor style does not cost an inference pass.
        let reuse = keys.and_then(|k| {
            self.preview_cache
                .as_ref()
                .filter(|(cached, _)| *cached == k.detect)
                .map(|(_, src)| src.clone())
        });
        self.preview_job = Some(match reuse {
            Some(src) => PreviewJob::compose(src, &self.prefs.profile),
            None => PreviewJob::detect(
                input,
                &self.prefs.profile,
                &self.settings,
                &self.ensemble_spec(),
            ),
        });
    }

    /// Keep the preview in step with the controls, without a round trip through
    /// another page and a button.
    ///
    /// Runs only while a preview panel is on screen: inference for a window
    /// nobody is looking at is pure heat.
    fn pump_auto_preview(&mut self, ctx: &egui::Context) {
        if !self.prefs.preview_auto
            || self.show_setup
            || !matches!(self.prefs.tab, Tab::Tuning | Tab::Queue)
            || !self.model_ready()
            // A batch already has every core busy; a preview would only take
            // inference time away from the work the user actually started.
            || self.is_running()
        {
            return;
        }
        // The editors write into `self.settings`; the fingerprint reads the
        // profile, so they have to be in step before it is taken.
        self.sync_profile();
        let Some(keys) = self.preview_keys() else {
            return;
        };
        let busy = self.preview_job.is_some();
        match self.preview_schedule.poll(keys, busy, Instant::now()) {
            PreviewNext::Idle => {}
            // Nothing else is animating, so ask for the frame that will fire it.
            PreviewNext::Wait(left) => ctx.request_repaint_after(left),
            PreviewNext::Start(keys) => {
                if let Some(input) = self.preview_path() {
                    self.spawn_preview(input, Some(keys));
                }
            }
        }
    }

    /// Copy the live setting editors back into the persisted profile.
    pub fn sync_profile(&mut self) {
        self.prefs.profile.model_settings = self.settings.clone();
    }

    // --- per-frame plumbing ---------------------------------------------

    fn pump_run(&mut self) {
        let Some(run) = &self.run else { return };
        let alive = run.pump(&mut self.run_state);
        if !alive {
            let output = run.output_dir.clone();
            self.run = None;
            // Keep what this machine actually managed, so the next batch can be
            // estimated before it starts instead of after its first file.
            if self.run_state.calibration.is_measured() {
                self.prefs.secs_per_unit = Some(self.run_state.calibration.secs_per_unit());
                let _ = self.prefs.save();
            }
            match &self.run_state.finished {
                Some(f) if f.cancelled => {
                    self.toast(
                        format!("Stopped. {} file(s) finished.", f.ok),
                        ToastKind::Info,
                    );
                }
                Some(f) if f.failed > 0 => {
                    self.toast(
                        format!("Finished with {} error(s) — see the log.", f.failed),
                        ToastKind::Error,
                    );
                }
                Some(f) => {
                    self.toast(
                        format!("Done — {} file(s) written to {}", f.ok, output.display()),
                        ToastKind::Success,
                    );
                }
                None => {}
            }
        }
    }

    fn pump_preview(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.preview_job else { return };
        let Some(result) = job.poll() else { return };
        self.preview_job = None;
        // Whatever happened, this attempt is spent: record it so an
        // unpreviewable file is not retried on every debounce tick.
        let pending = self.preview_schedule.finished();
        match result {
            Ok(done) => {
                if let (Some(src), Some(keys)) = (done.source, pending) {
                    self.preview_cache = Some((keys.detect, src));
                }
                let pair = done.pair;
                self.preview = Some(PreviewView {
                    original: load_texture(ctx, "ob-preview-original", &pair.original),
                    censored: load_texture(ctx, "ob-preview-censored", &pair.censored),
                    regions: pair.regions,
                });
                self.preview_error = None;
            }
            Err(e) => {
                self.preview_error = Some(e);
                self.preview = None;
                self.preview_cache = None;
            }
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            let n = dropped.len();
            self.add_inputs(dropped);
            self.prefs.tab = Tab::Queue;
            self.toast(format!("Added {n} item(s)."), ToastKind::Success);
        }
    }

    // --- chrome ----------------------------------------------------------

    fn nav_rail(&mut self, ui: &mut egui::Ui) {
        let p = theme::palette();
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.label(
                RichText::new("Figura Obscura")
                    .size(16.0)
                    .strong()
                    .color(p.text),
            );
        });
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                    .size(11.0)
                    .color(p.text_faint),
            );
        });
        ui.add_space(16.0);

        for tab in Tab::ALL {
            let selected = self.prefs.tab == tab;
            let label = RichText::new(format!("  {}   {}", tab.glyph(), tab.label()))
                .color(if selected { p.on_accent } else { p.text_dim });
            let button = egui::Button::new(label)
                .fill(if selected {
                    p.accent
                } else {
                    egui::Color32::TRANSPARENT
                })
                .stroke(egui::Stroke::NONE)
                .rounding(egui::Rounding::same(theme::RADIUS))
                .min_size(egui::vec2(ui.available_width(), 32.0));
            if ui.add(button).clicked() {
                self.prefs.tab = tab;
            }
            // A badge on Models while anything is downloading, so progress is
            // visible from any page.
            if tab == Tab::Models && self.downloads.any_active() {
                ui.horizontal(|ui| {
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(format!("{} downloading…", self.downloads.active_count()))
                            .size(11.0)
                            .color(p.accent_hover),
                    );
                });
            }
        }

        // Pin the model summary to the bottom of the rail: it is the one piece
        // of state that changes what every page does.
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add_space(10.0);
            let ready = self.model_ready();
            ui.horizontal_wrapped(|ui| {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(if ready {
                        format!("{} ready", theme::glyph::DOT)
                    } else {
                        format!("{} not installed", theme::glyph::DOT)
                    })
                    .size(11.0)
                    .color(if ready { p.success } else { p.warning }),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(self.current_entry().id)
                        .size(11.5)
                        .color(p.text_dim),
                );
            });
            theme::section(ui, "Active model");
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        let p = theme::palette();
        ui.add_space(6.0);

        // Progress first: while a run is going it is the most important thing
        // in the window.
        if self.is_running() {
            let text = match (self.run_state.total, self.run_state.eta_secs()) {
                (0, _) => "scanning inputs…".to_string(),
                (total, Some(eta)) => format!(
                    "{}/{} · {} left",
                    self.run_state.done,
                    total,
                    crate::downloads::human_eta(eta)
                ),
                (total, None) => format!("{}/{}", self.run_state.done, total),
            };
            let bar = match self.run_state.fraction() {
                Some(f) => egui::ProgressBar::new(f).text(text),
                None => egui::ProgressBar::new(0.0).animate(true).text(text),
            };
            ui.add(bar.desired_height(16.0).fill(p.accent));
            ui.add_space(4.0);
        }

        ui.horizontal(|ui| {
            if self.is_running() {
                if theme::danger_button(ui, "Stop", true).clicked() {
                    self.cancel_run();
                }
            } else {
                let blocker = self.run_blocker();
                let resp = theme::primary_button(ui, "Run batch", blocker.is_none());
                if resp.clicked() {
                    self.start_run();
                }
                // The disabled state explains itself instead of leaving the
                // user to guess which precondition is missing.
                if let Some(reason) = &blocker {
                    resp.on_disabled_hover_text(reason.clone());
                }
            }

            let can_preview =
                self.preview_path().is_some() && self.model_ready() && self.preview_job.is_none();
            if ui
                .add_enabled(can_preview, egui::Button::new("Refresh preview"))
                .clicked()
            {
                self.start_preview();
            }

            if let Some(job) = &self.preview_job {
                // Naming the slow stage explicitly: a re-detect is worth
                // waiting through, a re-composite is over before it is read.
                let what = if job.is_detect {
                    "detecting…"
                } else {
                    "rendering…"
                };
                ui.add(egui::Spinner::new().size(14.0));
                ui.label(RichText::new(what).color(p.text_dim));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(toast) = &self.toast {
                    if toast.at.elapsed() < TOAST_LIFETIME {
                        ui.label(RichText::new(&toast.text).color(toast.kind.color()));
                    } else {
                        self.toast = None;
                    }
                } else if let Some(f) = &self.run_state.finished {
                    let msg = if f.cancelled {
                        format!("Stopped after {:.0}s", f.elapsed_secs)
                    } else {
                        format!("{} ok, {} failed in {:.0}s", f.ok, f.failed, f.elapsed_secs)
                    };
                    ui.label(RichText::new(msg).color(p.text_dim));
                }
            });
        });
        ui.add_space(6.0);
    }

    /// A dismissible banner for a broken install (no ffmpeg).
    fn tool_banner(&mut self, ui: &mut egui::Ui) {
        if self.missing_tools.is_empty() {
            return;
        }
        let p = theme::palette();
        let names: Vec<&str> = self.missing_tools.iter().map(|t| t.name()).collect();
        egui::Frame::none()
            .fill(p.warning.gamma_multiply(0.12))
            .stroke(egui::Stroke::new(1.0_f32, p.warning.gamma_multiply(0.5)))
            .rounding(egui::Rounding::same(theme::RADIUS))
            .inner_margin(egui::Margin::symmetric(12.0, 9.0))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} {} not found.",
                            theme::glyph::WARN,
                            names.join(" and ")
                        ))
                        .color(p.warning)
                        .strong(),
                    );
                    ui.label(
                        RichText::new(
                            "Images still work; video needs these. They normally ship \
                             with Figura Obscura, so this usually means an incomplete install.",
                        )
                        .color(p.text_dim),
                    );
                });
                // Installing ffmpeg yourself is a fully supported mode, so the
                // banner offers the command rather than only reporting a fault.
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new("Install it with:")
                            .size(12.0)
                            .color(p.text_dim),
                    );
                    ui.code(ob_media::tools::install_hint());
                    if ui.small_button("Copy").clicked() {
                        let cmd = ob_media::tools::install_hint().to_string();
                        ui.output_mut(|o| o.copied_text = cmd);
                        self.toast("Command copied to the clipboard.", ToastKind::Info);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Details").clicked() {
                        let detail = self
                            .missing_tools
                            .iter()
                            .map(|t| ob_media::tools::search_description(*t))
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        ui.output_mut(|o| o.copied_text = detail);
                        self.toast("Search paths and install commands copied.", ToastKind::Info);
                    }
                    if ui.button("Re-check").clicked() {
                        self.missing_tools = ob_media::tools::missing_tools();
                        if self.missing_tools.is_empty() {
                            self.toast("ffmpeg found.", ToastKind::Success);
                        }
                    }
                    if ui.button("Dismiss").clicked() {
                        self.missing_tools.clear();
                    }
                });
            });
        ui.add_space(8.0);
    }
}

/// Upload a frame to the GPU as an egui texture.
fn load_texture(
    ctx: &egui::Context,
    name: &str,
    frame: &ob_core::geometry::Frame,
) -> egui::TextureHandle {
    let pixels: Vec<egui::Color32> = frame
        .data
        .chunks_exact(3)
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();
    let img = egui::ColorImage {
        size: [frame.width as usize, frame.height as usize],
        pixels,
    };
    // Linear filtering: previews are usually shown scaled down to fit.
    ctx.load_texture(name, img, egui::TextureOptions::LINEAR)
}

/// Open a path in the platform file manager.
///
/// Best-effort: a machine with no desktop session (a headless test box, a bare
/// WM) has nothing to open it with, and that is not worth an error dialog.
pub fn open_in_file_manager(path: &Path) {
    #[cfg(target_os = "windows")]
    let cmd = ("explorer", vec![path.to_path_buf()]);
    #[cfg(target_os = "macos")]
    let cmd = ("open", vec![path.to_path_buf()]);
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = ("xdg-open", vec![path.to_path_buf()]);

    let _ = std::process::Command::new(cmd.0)
        .args(cmd.1)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

impl eframe::App for ObApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui(ctx);
    }
}

impl ObApp {
    /// Draw one frame.
    ///
    /// Split out of the `eframe::App` impl so the whole interface can be
    /// exercised against a bare `egui::Context` in tests: `eframe::Frame` has
    /// no public constructor, and egui panics on duplicate widget ids, so
    /// "does the UI lay out at all" is worth having under test rather than
    /// discovering on a user's machine.
    pub fn ui(&mut self, ctx: &egui::Context) {
        self.pump_run();
        self.pump_estimate();
        self.pump_preview(ctx);
        self.pump_auto_preview(ctx);
        self.handle_dropped_files(ctx);
        let registry = std::mem::take(&mut self.registry);
        self.downloads.pump(&registry);
        self.registry = registry;

        // Repaint continuously only while something is actually moving; an idle
        // window should not burn a core redrawing itself.
        if self.is_running() || self.downloads.any_active() || self.preview_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        } else if self.toast.is_some() {
            ctx.request_repaint_after(Duration::from_millis(500));
        }

        // Save on window close, and stop any download so the process can exit
        // promptly instead of waiting on a 100 MB transfer.
        if ctx.input(|i| i.viewport().close_requested()) {
            self.sync_profile();
            let _ = self.prefs.save();
            self.downloads.cancel_all();
            if let Some(run) = &self.run {
                run.cancel.cancel();
            }
        }

        if self.show_setup {
            CentralPanel::default()
                .frame(egui::Frame::none().fill(theme::palette().bg))
                .show(ctx, |ui| crate::pages::setup::show(self, ui));
            return;
        }

        SidePanel::left("nav")
            .exact_width(184.0)
            .resizable(false)
            .frame(
                egui::Frame::none()
                    .fill(theme::palette().panel)
                    .inner_margin(egui::Margin::symmetric(10.0, 0.0)),
            )
            .show(ctx, |ui| self.nav_rail(ui));

        TopBottomPanel::bottom("status")
            .frame(
                egui::Frame::none()
                    .fill(theme::palette().panel)
                    .inner_margin(egui::Margin::symmetric(14.0, 2.0)),
            )
            .show(ctx, |ui| self.status_bar(ui));

        CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(theme::palette().bg)
                    .inner_margin(egui::Margin::symmetric(18.0, 14.0)),
            )
            .show(ctx, |ui| {
                self.tool_banner(ui);
                match self.prefs.tab {
                    Tab::Queue => crate::pages::queue::show(self, ui),
                    Tab::Tuning => crate::pages::tuning::show(self, ui),
                    Tab::Models => crate::pages::models::show(self, ui),
                    Tab::About => crate::pages::about::show(self, ui),
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefs::Tab;
    use std::sync::{Mutex, MutexGuard};

    /// `SB_CONFIG_DIR`/`SB_MODEL_DIR` are process-global but tests run in
    /// parallel threads, so constructing an app must be serialised or one test
    /// reads another's directories.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// A temp directory that cleans itself up even if the test panics.
    pub(super) struct TestDir {
        _lock: MutexGuard<'static, ()>,
        path: PathBuf,
    }

    impl std::ops::Deref for TestDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::env::remove_var("SB_CONFIG_DIR");
            std::env::remove_var("SB_MODEL_DIR");
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Build an app against a private config directory so tests never read or
    /// write the developer's real settings.
    pub(super) fn test_app(name: &str) -> (ObApp, TestDir) {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path = std::env::temp_dir().join(format!("ob-app-test-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        std::env::set_var("SB_CONFIG_DIR", &path);
        std::env::set_var("SB_MODEL_DIR", path.join("models"));
        let app = ObApp::default();
        (app, TestDir { _lock: lock, path })
    }

    /// Run `frames` full layout passes, which is what shakes out id clashes.
    fn draw(app: &mut ObApp, frames: usize) {
        let ctx = egui::Context::default();
        crate::theme::install(&ctx);
        for _ in 0..frames {
            let _ = ctx.run(egui::RawInput::default(), |ctx| app.ui(ctx));
        }
    }

    #[test]
    fn every_page_lays_out_without_panicking() {
        let (mut app, _dir) = test_app("pages");
        // Skip the first-run screen so the real pages are reachable.
        app.show_setup = false;
        for tab in Tab::ALL {
            app.prefs.tab = tab;
            draw(&mut app, 2);
        }
    }

    #[test]
    fn the_first_run_screen_lays_out() {
        let (mut app, _dir) = test_app("setup");
        app.show_setup = true;
        draw(&mut app, 2);
    }

    #[test]
    fn a_populated_queue_and_finished_run_lay_out() {
        let (mut app, dir) = test_app("populated");
        app.show_setup = false;
        app.prefs.tab = Tab::Queue;
        app.inputs = vec![dir.join("a.png"), dir.join("b.mp4"), dir.to_path_buf()];
        app.prefs.output_dir = Some(dir.join("out"));
        // Both log states, including the error branch and the summary bar.
        app.run_state.total = 2;
        app.run_state.done = 2;
        app.run_state.failed = 1;
        app.run_state.log.push_back(crate::run::LogEntry {
            path: dir.join("a.png"),
            error: None,
            regions: 3,
        });
        app.run_state.log.push_back(crate::run::LogEntry {
            path: dir.join("b.mp4"),
            error: Some("decode failed".into()),
            regions: 0,
        });
        app.run_state.finished = Some(crate::run::Finished {
            ok: 1,
            failed: 1,
            cancelled: false,
            elapsed_secs: 2.0,
        });
        draw(&mut app, 2);
        app.errors_only = true;
        draw(&mut app, 1);
    }

    #[test]
    fn the_tool_banner_and_toast_lay_out() {
        let (mut app, _dir) = test_app("banner");
        app.show_setup = false;
        app.missing_tools = vec![Tool::Ffmpeg, Tool::Ffprobe];
        app.toast("something happened", ToastKind::Error);
        draw(&mut app, 2);
    }

    #[test]
    fn run_is_blocked_until_every_precondition_is_met() {
        let (mut app, dir) = test_app("blocker");
        // Each blocker is reported in turn, so the disabled Run button can
        // always explain itself rather than just being grey.
        assert!(app.run_blocker().unwrap().contains("Add files"));
        app.inputs.push(dir.join("a.png"));
        assert!(app.run_blocker().unwrap().contains("output folder"));
        app.prefs.output_dir = Some(dir.join("out"));
        // No model is installed in this temp cache.
        assert!(app.run_blocker().unwrap().contains("not installed"));
    }

    #[test]
    fn select_model_works_while_the_registry_is_borrowed_out() {
        // The Models page moves `registry` out of the app for the duration of
        // its loop (so each card can take `&mut app`). A `select_model` that
        // looked the id up in `self.registry` would silently do nothing there —
        // clicking "Use this model" would appear to be ignored.
        let (mut app, _dir) = test_app("taken-registry");
        let taken = std::mem::take(&mut app.registry);
        app.select_model("anime-censor-v1-n");
        app.registry = taken;
        assert_eq!(app.prefs.profile.model_id, "anime-censor-v1-n");
    }

    #[test]
    fn switching_model_resets_settings_to_that_models_defaults() {
        let (mut app, _dir) = test_app("switch");
        app.select_model("anime-censor-v0.10-s");
        // 0.15 is v0.10_s's own published F1-optimal threshold. Carrying the
        // previous model's value across would silently move its operating point.
        let conf = app
            .settings
            .get("conf_threshold")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((conf - 0.15).abs() < 1e-9, "got {conf}");
        assert_eq!(app.prefs.profile.model_id, "anime-censor-v0.10-s");
    }

    fn keys(detect: u64, compose: u64) -> PreviewKeys {
        PreviewKeys { detect, compose }
    }

    #[test]
    fn the_debounce_waits_for_the_settings_to_stop_moving() {
        let mut sched = PreviewSchedule::default();
        let t0 = Instant::now();
        let k = keys(1, 1);

        // A change arms the timer rather than firing immediately.
        assert!(matches!(
            sched.poll(k, false, t0),
            PreviewNext::Wait(_)
        ));
        // Still moving: the timer restarts, so a drag never renders mid-drag.
        let k2 = keys(1, 2);
        assert!(matches!(
            sched.poll(k2, false, t0 + PREVIEW_DEBOUNCE - Duration::from_millis(1)),
            PreviewNext::Wait(_)
        ));
        // Nothing has changed for a full debounce: now it goes.
        assert_eq!(
            sched.poll(k2, false, t0 + PREVIEW_DEBOUNCE * 2),
            PreviewNext::Start(k2)
        );
    }

    #[test]
    fn a_rendered_state_is_not_rendered_again() {
        let mut sched = PreviewSchedule::default();
        let t0 = Instant::now();
        let k = keys(1, 1);
        let fire = t0 + PREVIEW_DEBOUNCE * 2;

        // First sight arms the timer; the render comes once it has elapsed.
        assert!(matches!(sched.poll(k, false, t0), PreviewNext::Wait(_)));
        assert_eq!(sched.poll(k, false, fire), PreviewNext::Start(k));
        sched.started(Some(k));
        // While it runs, and after it lands, the same settings are left alone —
        // otherwise the preview would loop on itself forever.
        assert_eq!(sched.poll(k, true, fire), PreviewNext::Idle);
        assert_eq!(sched.finished(), Some(k));
        assert_eq!(sched.poll(k, false, fire), PreviewNext::Idle);
    }

    #[test]
    fn a_failed_preview_is_not_retried_until_something_changes() {
        // A file that cannot be decoded fails every time. Without recording the
        // attempt, the debounce would re-run it forever.
        let mut sched = PreviewSchedule::default();
        let t0 = Instant::now();
        let t = t0 + PREVIEW_DEBOUNCE * 2;
        let k = keys(7, 7);
        assert!(matches!(sched.poll(k, false, t0), PreviewNext::Wait(_)));
        assert_eq!(sched.poll(k, false, t), PreviewNext::Start(k));
        sched.started(Some(k));
        sched.finished();
        assert_eq!(sched.poll(k, false, t), PreviewNext::Idle);
        // A different setting is a new question, and is asked.
        let k2 = keys(7, 8);
        assert!(matches!(sched.poll(k2, false, t), PreviewNext::Wait(_)));
        assert_eq!(
            sched.poll(k2, false, t + PREVIEW_DEBOUNCE),
            PreviewNext::Start(k2)
        );
    }

    #[test]
    fn a_change_during_a_render_fires_once_that_render_lands() {
        let mut sched = PreviewSchedule::default();
        let t0 = Instant::now() + PREVIEW_DEBOUNCE * 2;
        let k1 = keys(1, 1);
        sched.started(Some(k1));

        // Something moved while the first job was running.
        let k2 = keys(1, 2);
        let t1 = t0 + PREVIEW_DEBOUNCE * 2;
        // Debounced, then held back only because a job is in flight.
        assert!(matches!(sched.poll(k2, true, t0), PreviewNext::Wait(_)));
        assert_eq!(sched.poll(k2, true, t1), PreviewNext::Idle);
        // The moment it lands, the newer state is picked up — no extra input
        // from the user needed to un-stick it.
        sched.finished();
        assert_eq!(sched.poll(k2, false, t1), PreviewNext::Start(k2));
    }

    #[test]
    fn a_companion_promoted_to_primary_leaves_the_companion_list() {
        let (mut app, _dir) = test_app("ensemble-promote");
        app.select_model("nudenet-320n");
        app.prefs.extra_models = vec!["anime-censor-v1-s".into(), "nudenet-640m".into()];

        app.select_model("anime-censor-v1-s");

        // Otherwise one model would corroborate itself, and the entry would
        // reappear the moment the primary was switched away again.
        assert_eq!(app.prefs.extra_models, vec!["nudenet-640m".to_string()]);
        assert_eq!(app.ensemble_spec().members(), 2);
    }

    #[test]
    fn the_vote_threshold_is_clamped_to_the_models_that_remain() {
        let (mut app, _dir) = test_app("ensemble-votes");
        app.select_model("nudenet-320n");
        app.prefs.extra_models = vec!["nudenet-640m".into(), "anime-censor-v1-s".into()];
        app.prefs.min_votes = 3;
        assert_eq!(app.ensemble_spec().effective_votes(), 3);

        // A saved threshold must not outlive the models it referred to: with
        // one companion left, "3 must agree" can never be satisfied and would
        // censor nothing at all.
        app.prefs.extra_models = vec!["nudenet-640m".into()];
        let spec = app.ensemble_spec();
        assert_eq!(spec.members(), 2);
        assert_eq!(spec.effective_votes(), 2);

        app.prefs.extra_models.clear();
        assert_eq!(app.ensemble_spec().effective_votes(), 1);
    }

    #[test]
    fn corroboration_re_runs_inference_rather_than_re_painting() {
        let (mut app, dir) = test_app("ensemble-keys");
        let file = dir.join("a.png");
        std::fs::write(&file, b"x").unwrap();
        app.inputs = vec![file];
        app.select_model("nudenet-320n");

        let before = app.preview_keys().expect("keys");
        app.prefs.extra_models = vec!["nudenet-640m".into()];
        let after = app.preview_keys().expect("keys");

        // Adding a model changes which regions exist at all, so it belongs to
        // the expensive half — re-painting the cached detections would show a
        // preview the batch would not reproduce.
        assert_ne!(before.detect, after.detect);
        assert_eq!(before.compose, after.compose);

        // Same again for the threshold.
        app.prefs.min_votes = 2;
        let voted = app.preview_keys().expect("keys");
        assert_ne!(after.detect, voted.detect);
        assert_eq!(after.compose, voted.compose);
    }

    #[test]
    fn a_single_model_is_not_wrapped_in_an_ensemble() {
        let (mut app, _dir) = test_app("ensemble-single");
        app.select_model("nudenet-320n");
        let spec = app.ensemble_spec();
        assert!(!spec.is_ensemble());
        assert_eq!(spec.members(), 1);
        assert_eq!(spec.effective_votes(), 1);
    }

    #[test]
    fn the_preview_source_is_a_file_never_a_folder() {
        let (mut app, dir) = test_app("preview-path");
        let file = dir.join("a.png");
        std::fs::write(&file, b"x").unwrap();

        // A batch is very often one folder; previewing it would only ever
        // report "that is a directory".
        app.inputs = vec![dir.to_path_buf(), file.clone()];
        assert_eq!(app.preview_path(), Some(file));

        // An explicit sample wins, and need not be in the batch at all.
        let sample = dir.join("sample.png");
        std::fs::write(&sample, b"x").unwrap();
        app.preview_source_path = Some(sample.clone());
        assert_eq!(app.preview_path(), Some(sample));
    }

    #[test]
    fn a_style_change_does_not_invalidate_the_detections() {
        // The split that makes the live preview affordable: re-painting a
        // censor style must not look like a reason to run the model again.
        let (mut app, dir) = test_app("preview-keys");
        let file = dir.join("a.png");
        std::fs::write(&file, b"x").unwrap();
        app.inputs = vec![file];
        let before = app.preview_keys().expect("a file is queued");

        app.prefs.profile.censor.shape.padding += 0.25;
        let after = app.preview_keys().unwrap();
        assert_eq!(before.detect, after.detect, "padding forced a re-detect");
        assert_ne!(before.compose, after.compose);

        // A detection setting, by contrast, honestly costs a re-detect.
        app.settings.insert(
            "conf_threshold".into(),
            ob_core::settings::SettingValue::Float(0.9),
        );
        app.sync_profile();
        let tuned = app.preview_keys().unwrap();
        assert_ne!(after.detect, tuned.detect);
    }

    /// Drive `pump_estimate` until the background scan lands.
    pub(super) fn settle_estimate(app: &mut ObApp) {
        for _ in 0..400 {
            app.pump_estimate();
            if app.probed.is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("the batch scan never finished");
    }

    pub(super) fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let p = dir.join(name);
        image::RgbImage::new(w, h).save(&p).unwrap();
        p
    }

    #[test]
    fn the_batch_is_measured_and_re_measured_when_the_inputs_change() {
        let (mut app, dir) = test_app("estimate-scan");
        app.select_model("nudenet-320n");
        app.inputs = vec![write_png(&dir, "a.png", 4000, 3000)];
        settle_estimate(&mut app);

        let one = app.workload().expect("a costed batch");
        assert_eq!(one.images, 1);
        assert!(one.total_work > 0.0);

        // A second, much smaller file must change the total — and must trigger
        // a fresh scan, since a measurement of the previous batch would be
        // worse than none at all.
        app.inputs.push(write_png(&dir, "b.png", 64, 64));
        app.pump_estimate();
        assert!(app.probed.is_none(), "the stale scan was kept");
        settle_estimate(&mut app);

        let two = app.workload().expect("a costed batch");
        assert_eq!(two.images, 2);
        assert!(two.total_work > one.total_work);
    }

    #[test]
    fn changing_a_setting_re_costs_without_re_reading_the_batch() {
        let (mut app, dir) = test_app("estimate-recost");
        app.select_model("nudenet-320n");
        app.inputs = vec![write_png(&dir, "big.png", 4000, 3000)];
        settle_estimate(&mut app);

        let before = app.workload().expect("costed").total_work;
        let probed = app.probed.clone().expect("probed");

        // Two models is twice the inference over every frame.
        app.prefs.extra_models = vec!["nudenet-640m".into()];
        let after = app.workload().expect("costed").total_work;
        assert!((after - before * 2.0).abs() < 1e-6);

        // Re-costing is pure: no second pass over the disk, which is what lets
        // the estimate track a slider.
        app.pump_estimate();
        assert!(Arc::ptr_eq(&probed, app.probed.as_ref().unwrap()));
    }

    #[test]
    fn tiling_off_flattens_the_cost_of_a_large_frame() {
        let (mut app, dir) = test_app("estimate-tiling");
        app.select_model("nudenet-320n");
        app.inputs = vec![write_png(&dir, "big.png", 4000, 3000)];
        settle_estimate(&mut app);

        let tiled = app.workload().expect("costed").total_work;

        // With tiling off a 4K frame is letterboxed into one input like any
        // other, so the estimate has to collapse to a single pass.
        app.settings.insert(
            "tiling".to_string(),
            ob_core::settings::SettingValue::Text("off".into()),
        );
        let flat = app.workload().expect("costed").total_work;
        assert!(flat < tiled, "{flat} should be below {tiled}");
        assert!((flat - 1.0).abs() < 1e-6, "one pass, got {flat}");
    }

    #[test]
    fn duplicate_inputs_are_not_added_twice() {
        let (mut app, dir) = test_app("dupes");
        let p = dir.join("a.png");
        app.add_inputs([p.clone(), p.clone()]);
        app.add_inputs([p.clone()]);
        assert_eq!(app.inputs.len(), 1);
    }
}

/// Renders the real UI to PNG files so its *appearance* can be reviewed.
///
/// The build container has no display server, so these are the only images of
/// the app that exist. They land in `target/screenshots/`.
#[cfg(test)]
mod screenshots {
    use super::tests::*;
    use super::Tool;
    use crate::prefs::Tab;
    use crate::screenshot;
    use crate::theme;

    const W: usize = 1180;
    const H: usize = 820;

    fn out_dir() -> std::path::PathBuf {
        // CARGO_MANIFEST_DIR is crates/ob-gui; the workspace target dir is two
        // levels up, which keeps the images out of the source tree.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/screenshots")
    }

    fn shoot(name: &str, app: &mut super::ObApp) {
        shoot_min(name, app, 0.15)
    }

    /// A synthetic photo: a diagonal gradient with a finely striped patch where
    /// the "detection" is.
    ///
    /// The stripes matter — pixelating a flat colour is a no-op, so a blank
    /// patch would make every censor style look like it did nothing, which is
    /// the opposite of what the screenshot is for.
    fn sample_frame(w: u32, h: u32) -> ob_core::geometry::Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let in_patch = x > w / 2 && x < w * 4 / 5 && y > h / 4 && y < h * 3 / 4;
                if in_patch {
                    let stripe = ((x / 3 + y / 5) % 2) == 0;
                    let v: u8 = if stripe { 245 } else { 120 };
                    data[i] = v;
                    data[i + 1] = v.saturating_sub(45);
                    data[i + 2] = v.saturating_sub(70);
                } else {
                    data[i] = (x * 200 / w) as u8;
                    data[i + 1] = (y * 160 / h) as u8;
                    data[i + 2] = 90;
                }
            }
        }
        ob_core::geometry::Frame::new(w, h, data).unwrap()
    }

    /// Put a real, pipeline-rendered preview into the app.
    ///
    /// Built through `ob_job::preview_compose` rather than by painting a
    /// rectangle, so the screenshot shows the censor styles as they will
    /// actually appear — the point of looking at it at all.
    fn install_preview(app: &mut super::ObApp, ctx: &egui::Context) {
        let (w, h) = (520u32, 340u32);
        let src = ob_job::PreviewSource {
            frame: sample_frame(w, h),
            detections: vec![ob_core::geometry::Detection {
                bbox: ob_core::geometry::BBox::new(
                    w as f32 * 0.5,
                    h as f32 * 0.25,
                    w as f32 * 0.8,
                    h as f32 * 0.75,
                ),
                category: ob_core::taxonomy::cat::FEMALE_BREAST_EXPOSED,
                score: 0.87,
            }],
            detect_error: None,
        };
        let pair = ob_job::preview_compose(&src, &app.prefs.profile).expect("compose");
        app.preview = Some(super::PreviewView {
            original: super::load_texture(ctx, "shot-original", &pair.original),
            censored: super::load_texture(ctx, "shot-censored", &pair.censored),
            regions: pair.regions,
        });
    }

    /// `min_painted` is the floor for how much of the window must be non-blank.
    /// It varies by screen: a full page covers most of the window, while the
    /// centred first-run card deliberately leaves the surround empty.
    fn shoot_min(name: &str, app: &mut super::ObApp, min_painted: f32) {
        let clear = theme::palette().bg;
        let canvas = screenshot::render(W, H, clear, 4, |ctx| app.ui(ctx));

        // A blank render would still "pass" every layout test, so assert that
        // a real amount of the window was actually painted.
        let painted = canvas.painted_fraction(clear);
        assert!(
            painted > min_painted,
            "{name}: only {:.1}% of the window was painted — the UI did not render",
            painted * 100.0
        );

        let path = out_dir().join(format!("{name}.png"));
        canvas.save(&path).unwrap();
        eprintln!("wrote {} ({:.0}% painted)", path.display(), painted * 100.0);
    }

    #[test]
    fn render_first_run() {
        let (mut app, _dir) = test_app("shot-setup");
        app.show_setup = true;
        shoot_min("01-first-run", &mut app, 0.05);
    }

    #[test]
    fn render_batch_empty() {
        let (mut app, _dir) = test_app("shot-batch-empty");
        app.show_setup = false;
        app.prefs.tab = Tab::Queue;
        shoot("02-batch-empty", &mut app);
    }

    #[test]
    fn render_batch_populated() {
        let (mut app, dir) = test_app("shot-batch-full");
        app.show_setup = false;
        app.prefs.tab = Tab::Queue;
        app.prefs.output_dir = Some(dir.join("censored"));
        for n in ["beach-01.jpg", "beach-02.jpg", "clip.mp4", "notes.txt"] {
            let p = dir.join(n);
            std::fs::write(&p, vec![0u8; 4096]).unwrap();
            app.inputs.push(p);
        }
        app.inputs.push(dir.to_path_buf());

        app.run_state.total = 5;
        app.run_state.done = 3;
        app.run_state.failed = 1;
        app.run_state.log.push_back(crate::run::LogEntry {
            path: dir.join("beach-01.jpg"),
            error: None,
            regions: 2,
        });
        app.run_state.log.push_back(crate::run::LogEntry {
            path: dir.join("beach-02.jpg"),
            error: None,
            regions: 1,
        });
        app.run_state.log.push_back(crate::run::LogEntry {
            path: dir.join("clip.mp4"),
            error: Some("could not run ffprobe".into()),
            regions: 0,
        });
        app.run_state.finished = Some(crate::run::Finished {
            ok: 2,
            failed: 1,
            cancelled: false,
            elapsed_secs: 12.0,
        });
        shoot("03-batch-populated", &mut app);
    }

    /// The Batch page with the estimate in place: what the queue is going to
    /// cost, before anything has been run.
    #[test]
    fn render_batch_estimated() {
        let (mut app, dir) = test_app("shot-batch-estimate");
        app.show_setup = false;
        app.prefs.tab = Tab::Queue;
        app.select_model("nudenet-320n");
        app.prefs.output_dir = Some(dir.join("out"));
        for (name, w, h) in [
            ("holiday-4k.png", 3840, 2160),
            ("portrait.png", 2000, 3000),
            ("thumb.png", 320, 240),
        ] {
            write_png(&dir, name, w, h);
            app.inputs.push(dir.join(name));
        }
        // A rate measured by an earlier run, so the line reads as a figure
        // rather than as the provisional default.
        app.prefs.secs_per_unit = Some(0.08);
        settle_estimate(&mut app);

        shoot("10-batch-estimate", &mut app);
    }

    #[test]
    fn render_tuning() {
        let (mut app, _dir) = test_app("shot-tuning");
        app.show_setup = false;
        app.prefs.tab = Tab::Tuning;
        shoot("04-tuning", &mut app);
    }

    /// The Tuning page doing its actual job: controls and the result of those
    /// controls on screen together.
    #[test]
    fn render_tuning_live_preview() {
        let (mut app, dir) = test_app("shot-tuning-live");
        app.show_setup = false;
        app.prefs.tab = Tab::Tuning;
        app.prefs.preview_compare = true;
        let sample = dir.join("beach-01.jpg");
        std::fs::write(&sample, vec![0u8; 4096]).unwrap();
        app.preview_source_path = Some(sample);

        let clear = theme::palette().bg;
        let mut installed = false;
        // Textures must be loaded into the same context that renders them, so
        // the preview is installed on the first layout pass.
        let canvas = screenshot::render(1480, 900, clear, 4, |ctx| {
            if !installed {
                install_preview(&mut app, ctx);
                installed = true;
            }
            app.ui(ctx);
        });
        let painted = canvas.painted_fraction(clear);
        assert!(painted > 0.15, "only {:.1}% painted", painted * 100.0);
        canvas
            .save(&out_dir().join("08-tuning-live-preview.png"))
            .unwrap();
    }

    /// Cross-examination with models actually installed: the checkbox list and
    /// the agreement threshold it unlocks.
    #[test]
    fn render_tuning_cross_examination() {
        let (mut app, dir) = test_app("shot-cross-exam");
        app.show_setup = false;
        app.prefs.tab = Tab::Tuning;
        app.select_model("nudenet-320n");

        // Pretend the weights are present. `status` skips checksum
        // verification while a registry entry has no pinned SHA-256, so an
        // empty file at the cache path reads as installed.
        let models = dir.join("models");
        std::fs::create_dir_all(&models).unwrap();
        // `looks_like_onnx` sniffs for a protobuf field-1 tag, so the fixture
        // has to start with 0x08 rather than being merely non-empty.
        for id in ["nudenet-640m", "anime-censor-v1-s", "anime-censor-v1-n"] {
            std::fs::write(models.join(format!("{id}.onnx")), b"\x08\x01").unwrap();
        }
        let registry = std::mem::take(&mut app.registry);
        app.downloads.refresh_all(&registry);
        app.registry = registry;

        app.prefs.extra_models = vec!["nudenet-640m".into(), "anime-censor-v1-s".into()];
        app.prefs.min_votes = 2;

        // Taller than the standard shot: the point of this one is the whole
        // card, and the default viewport clips it mid-list.
        let clear = theme::palette().bg;
        let canvas = screenshot::render(1180, 1180, clear, 4, |ctx| app.ui(ctx));
        assert!(canvas.painted_fraction(clear) > 0.15);
        canvas
            .save(&out_dir().join("09-tuning-cross-examination.png"))
            .unwrap();
    }

    #[test]
    fn render_models() {
        let (mut app, _dir) = test_app("shot-models");
        app.show_setup = false;
        app.prefs.tab = Tab::Models;
        shoot("05-models", &mut app);
    }

    #[test]
    fn render_about() {
        let (mut app, _dir) = test_app("shot-about");
        app.show_setup = false;
        app.prefs.tab = Tab::About;
        shoot("06-about", &mut app);
    }

    #[test]
    fn render_missing_tools_banner() {
        let (mut app, _dir) = test_app("shot-banner");
        app.show_setup = false;
        app.prefs.tab = Tab::Queue;
        app.missing_tools = vec![Tool::Ffmpeg, Tool::Ffprobe];
        shoot("07-ffmpeg-missing", &mut app);
    }
}
