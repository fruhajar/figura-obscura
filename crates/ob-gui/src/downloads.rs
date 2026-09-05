//! Background model downloads.
//!
//! Downloads run on worker threads and report back over a channel, so the UI
//! thread never blocks on the network. The manager owns one entry per model
//! being fetched: its progress, its cancel token, and its outcome.
//!
//! It also caches [`ob_models::ModelStatus`]. That matters more than it looks:
//! `status()` stats the file *and*, once a checksum is pinned, hashes it — a
//! 100 MB SHA-256 every frame would turn the model list into a slideshow. The
//! cache is refreshed at startup and whenever a download or deletion changes
//! the answer.

use ob_core::cancel::CancelToken;
use ob_core::registry::ModelEntry;
use ob_models::{DownloadProgress, FetchOptions, ModelStatus};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

/// A message from a download worker to the UI thread.
enum Msg {
    Progress {
        id: String,
        p: DownloadProgress,
    },
    Finished {
        id: String,
        result: Result<u64, String>,
    },
}

/// Live state of one in-flight download.
pub struct Active {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub cancel: CancelToken,
    started: Instant,
}

impl Active {
    /// Completed fraction, when the server told us the total.
    pub fn fraction(&self) -> Option<f32> {
        DownloadProgress {
            downloaded: self.downloaded,
            total: self.total,
        }
        .fraction()
    }

    /// Bytes per second so far, or `None` before there is enough to divide by.
    pub fn bytes_per_sec(&self) -> Option<f64> {
        let secs = self.started.elapsed().as_secs_f64();
        if secs < 0.5 || self.downloaded == 0 {
            return None;
        }
        Some(self.downloaded as f64 / secs)
    }

    /// Seconds remaining, when both a total and a rate are known.
    pub fn eta_secs(&self) -> Option<f64> {
        let total = self.total?;
        let rate = self.bytes_per_sec()?;
        if rate <= 0.0 || total <= self.downloaded {
            return None;
        }
        Some((total - self.downloaded) as f64 / rate)
    }
}

/// Owns every in-flight download plus the cached on-disk model status.
pub struct Downloads {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    active: HashMap<String, Active>,
    /// Cached `status()` per model id — see the module docs on why.
    status: HashMap<String, ModelStatus>,
    /// Last error per model, shown on its card until the next attempt.
    errors: HashMap<String, String>,
}

impl Default for Downloads {
    fn default() -> Self {
        Self::new()
    }
}

impl Downloads {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            tx,
            rx,
            active: HashMap::new(),
            status: HashMap::new(),
            errors: HashMap::new(),
        }
    }

    /// Re-read every model's on-disk state. Call at startup and after a change.
    pub fn refresh_all(&mut self, entries: &[ModelEntry]) {
        for e in entries {
            self.status.insert(e.id.to_string(), ob_models::status(e));
        }
    }

    fn refresh_one(&mut self, entry: &ModelEntry) {
        self.status
            .insert(entry.id.to_string(), ob_models::status(entry));
    }

    /// Cached status. `Missing` for a model never refreshed, which is the safe
    /// direction: the UI offers a download rather than claiming one exists.
    pub fn status(&self, id: &str) -> ModelStatus {
        self.status.get(id).cloned().unwrap_or(ModelStatus::Missing)
    }

    pub fn is_installed(&self, id: &str) -> bool {
        self.status(id).is_installed()
    }

    pub fn active(&self, id: &str) -> Option<&Active> {
        self.active.get(id)
    }

    pub fn is_downloading(&self, id: &str) -> bool {
        self.active.contains_key(id)
    }

    pub fn any_active(&self) -> bool {
        !self.active.is_empty()
    }

    pub fn error(&self, id: &str) -> Option<&str> {
        self.errors.get(id).map(String::as_str)
    }

    /// Number of models still downloading.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Start downloading `entry`. A no-op if it is already in flight.
    pub fn start(&mut self, entry: &ModelEntry, force: bool) {
        if self.active.contains_key(entry.id) {
            return;
        }
        self.errors.remove(entry.id);

        let cancel = CancelToken::new();
        self.active.insert(
            entry.id.to_string(),
            Active {
                downloaded: 0,
                total: None,
                cancel: cancel.clone(),
                started: Instant::now(),
            },
        );

        // `ModelEntry` owns only `&'static str`s and a `Vec<Setting>`, so it
        // moves into the worker without borrowing anything from the UI.
        let entry = entry.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let id = entry.id.to_string();
            let progress_tx = tx.clone();
            let progress_id = id.clone();
            let on_progress = move |p: DownloadProgress| {
                // A closed receiver means the app is shutting down; the cancel
                // token below is what actually stops us, so ignore the error.
                let _ = progress_tx.send(Msg::Progress {
                    id: progress_id.clone(),
                    p,
                });
            };
            let opts = FetchOptions::interactive()
                .force(force)
                .progress(&on_progress)
                .cancel(cancel.as_flag());

            let result = ob_models::fetch_with(&entry, &opts)
                .map(|o| match o {
                    ob_models::FetchOutcome::Downloaded { bytes, .. } => bytes,
                    ob_models::FetchOutcome::AlreadyPresent(_) => 0,
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(Msg::Finished { id, result });
        });
    }

    /// Ask an in-flight download to stop. The worker removes its partial file.
    pub fn cancel(&self, id: &str) {
        if let Some(a) = self.active.get(id) {
            a.cancel.cancel();
        }
    }

    /// Cancel everything — used when the window is closing.
    pub fn cancel_all(&self) {
        for a in self.active.values() {
            a.cancel.cancel();
        }
    }

    /// Delete a downloaded model and update the cached status.
    pub fn remove(&mut self, entry: &ModelEntry) -> Result<(), String> {
        ob_models::remove(entry).map_err(|e| e.to_string())?;
        self.refresh_one(entry);
        Ok(())
    }

    /// Drain worker messages. Returns true if anything changed, so the caller
    /// knows whether a repaint is warranted.
    pub fn pump(&mut self, entries: &[ModelEntry]) -> bool {
        let mut changed = false;
        // Collect first: handling messages borrows `self` mutably.
        let msgs: Vec<Msg> = self.rx.try_iter().collect();
        for msg in msgs {
            changed = true;
            match msg {
                Msg::Progress { id, p } => {
                    if let Some(a) = self.active.get_mut(&id) {
                        a.downloaded = p.downloaded;
                        a.total = p.total;
                    }
                }
                Msg::Finished { id, result } => {
                    self.active.remove(&id);
                    if let Err(e) = result {
                        self.errors.insert(id.clone(), e);
                    }
                    if let Some(entry) = entries.iter().find(|e| e.id == id) {
                        self.refresh_one(entry);
                    }
                }
            }
        }
        changed
    }
}

/// Format a duration the way a download UI should: coarse, and never more
/// precise than the estimate deserves.
pub fn human_eta(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "—".into();
    }
    let s = secs.round() as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_is_formatted_coarsely() {
        assert_eq!(human_eta(9.4), "9s");
        assert_eq!(human_eta(90.0), "1m 30s");
        assert_eq!(human_eta(3700.0), "1h 01m");
        // A rate estimate can produce nonsense early on; it must not panic or
        // render as "NaNs".
        assert_eq!(human_eta(f64::NAN), "—");
        assert_eq!(human_eta(-1.0), "—");
    }

    #[test]
    fn an_unrefreshed_model_reads_as_missing() {
        let d = Downloads::new();
        // The safe default: offer a download rather than claim a model exists.
        assert_eq!(d.status("never-seen"), ModelStatus::Missing);
        assert!(!d.is_installed("never-seen"));
        assert!(!d.any_active());
    }

    #[test]
    fn progress_and_rate_need_elapsed_time() {
        let a = Active {
            downloaded: 0,
            total: Some(100),
            cancel: CancelToken::new(),
            started: Instant::now(),
        };
        assert_eq!(a.fraction(), Some(0.0));
        // Nothing downloaded and no time elapsed: no rate, and so no ETA.
        assert_eq!(a.bytes_per_sec(), None);
        assert_eq!(a.eta_secs(), None);
    }

    #[test]
    fn cancelling_an_unknown_download_is_harmless() {
        let d = Downloads::new();
        d.cancel("not-running");
        d.cancel_all();
    }
}
