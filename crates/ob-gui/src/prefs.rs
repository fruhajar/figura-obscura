//! Persisted user state.
//!
//! Two things are saved, in one file: the [`Profile`] (what to censor and how —
//! the same structure `obscura process --profile` loads, so the GUI and the CLI can
//! exchange configurations) and the UI-only preferences around it.
//!
//! Saving is deliberately not on every keystroke. The app writes on exit and
//! when a run starts; losing the last few seconds of slider fiddling after a
//! crash is a far smaller cost than rewriting the file sixty times a second.

use ob_core::profile::Profile;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Which page the window is showing. Persisted so the app reopens where the
/// user left it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Tab {
    #[default]
    Queue,
    Tuning,
    Models,
    About,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Queue, Tab::Tuning, Tab::Models, Tab::About];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Queue => "Batch",
            Tab::Tuning => "Tuning",
            Tab::Models => "Models",
            Tab::About => "About",
        }
    }

    /// A glyph for the nav rail, taken from the checked table in `theme::glyph`
    /// — egui's bundled fonts cover an awkward subset of the symbol blocks and
    /// an uncovered character renders as a silent tofu box.
    pub fn glyph(self) -> &'static str {
        use crate::theme::glyph;
        match self {
            Tab::Queue => glyph::NAV_BATCH,
            Tab::Tuning => glyph::NAV_TUNING,
            Tab::Models => glyph::NAV_MODELS,
            Tab::About => glyph::NAV_ABOUT,
        }
    }
}

/// Everything the app remembers between launches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefs {
    #[serde(default = "default_version")]
    pub version: u32,
    /// The censoring configuration. Shared format with the CLI.
    #[serde(default)]
    pub profile: Profile,
    /// Where output goes. `None` until the user picks one.
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    /// Video: run the detector every Nth frame.
    #[serde(default = "default_detect_every")]
    pub detect_every: u32,
    #[serde(default)]
    pub tab: Tab,
    /// Set once the user has been through (or dismissed) first-run setup, so
    /// the wizard does not reappear for someone deliberately running offline.
    #[serde(default)]
    pub setup_done: bool,
    /// Show the uncensored source alongside the result in the preview.
    #[serde(default)]
    pub preview_compare: bool,
    /// Re-render the preview automatically as the settings change, instead of
    /// waiting for the button. On by default: tuning a detector without seeing
    /// the effect is guesswork, and the alternative is a round trip through
    /// another page for every adjustment.
    #[serde(default = "default_true")]
    pub preview_auto: bool,
    /// Companion models that cross-examine the primary, by registry id.
    ///
    /// Empty is the normal case: one model, everything it finds is censored.
    #[serde(default)]
    pub extra_models: Vec<String>,
    /// How many models must independently find a region before it is censored.
    ///
    /// Clamped to the member count when the detector is built, so removing a
    /// model can never leave a threshold that nothing can satisfy.
    #[serde(default = "default_min_votes")]
    pub min_votes: usize,
    /// Seconds per work unit measured by the last run on this machine.
    ///
    /// Persisted so the *second* batch onwards can show an accurate estimate
    /// before it starts: the scan says how much work a batch is, but only this
    /// machine can say how fast it gets through it, and that varies by two
    /// orders of magnitude between a CPU session and a discrete GPU.
    #[serde(default)]
    pub secs_per_unit: Option<f64>,
}

fn default_version() -> u32 {
    1
}

fn default_detect_every() -> u32 {
    3
}

fn default_true() -> bool {
    true
}

fn default_min_votes() -> usize {
    1
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            version: default_version(),
            profile: Profile::default(),
            output_dir: None,
            detect_every: default_detect_every(),
            tab: Tab::default(),
            setup_done: false,
            preview_compare: false,
            preview_auto: true,
            extra_models: Vec::new(),
            min_votes: 1,
            secs_per_unit: None,
        }
    }
}

/// Where the preferences file lives, e.g. `~/.config/figura-obscura/settings.json`.
/// Overridable with `SB_CONFIG_DIR`, which the packaging scripts use to test a
/// clean first run without touching the developer's real config.
pub fn config_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("SB_CONFIG_DIR") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    dirs::config_dir().map(|d| d.join("figura-obscura"))
}

pub fn prefs_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("settings.json"))
}

impl Prefs {
    /// Load saved preferences, falling back to defaults.
    ///
    /// A corrupt or half-written file yields defaults rather than an error: the
    /// app must always start. The bad file is left in place for diagnosis
    /// instead of being deleted behind the user's back.
    pub fn load() -> Self {
        let Some(path) = prefs_path() else {
            return Self::default();
        };
        Self::load_from(&path).unwrap_or_default()
    }

    pub fn load_from(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Write preferences, creating the directory if needed.
    ///
    /// Writes to a temporary file and renames, so an interrupted save cannot
    /// leave a truncated settings file that the next launch silently discards.
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = prefs_path() else {
            return Ok(());
        };
        self.save_to(&path)
    }

    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("obscura-prefs-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("settings.json")
    }

    #[test]
    fn prefs_round_trip_through_disk() {
        let path = tmp("roundtrip");
        let p = Prefs {
            detect_every: 9,
            tab: Tab::Models,
            setup_done: true,
            output_dir: Some(PathBuf::from("/tmp/out")),
            ..Default::default()
        };
        p.save_to(&path).unwrap();

        let back = Prefs::load_from(&path).unwrap();
        assert_eq!(back.detect_every, 9);
        assert_eq!(back.tab, Tab::Models);
        assert!(back.setup_done);
        assert_eq!(back.output_dir, Some(PathBuf::from("/tmp/out")));
        assert_eq!(back.profile, p.profile);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_corrupt_file_yields_defaults_and_is_not_deleted() {
        let path = tmp("corrupt");
        std::fs::write(&path, b"{not json at all").unwrap();
        assert!(Prefs::load_from(&path).is_none());
        // The app must still start, and the bad file must survive for support
        // to look at rather than being silently removed.
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        let path = tmp("partial");
        // A settings file written by an older build that had fewer fields.
        std::fs::write(&path, br#"{"version":1,"detect_every":5}"#).unwrap();
        let p = Prefs::load_from(&path).expect("partial prefs should still load");
        assert_eq!(p.detect_every, 5);
        assert_eq!(p.tab, Tab::Queue);
        assert!(!p.setup_done);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn saving_leaves_no_temporary_file_behind() {
        let path = tmp("atomic");
        Prefs::default().save_to(&path).unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("json.tmp").exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn every_tab_has_a_label_and_glyph() {
        for t in Tab::ALL {
            assert!(!t.label().is_empty());
            assert!(!t.glyph().is_empty());
        }
    }
}
