//! Saveable job profiles: the full, serializable description of *what* to
//! censor and *how*. Both CLI (`obscura profile save/load`) and GUI persist these.

use crate::censor::CensorConfig;
use crate::filter::FilterSet;
use crate::settings::SettingValues;
use serde::{Deserialize, Serialize};

/// Whether a frame that fails detection is emitted uncensored, skipped, or
/// blanked (fail-closed). See `ob-job`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnDetectFailure {
    /// Emit the original frame unchanged (fastest, least safe).
    PassThrough,
    /// Skip the file/frame entirely and record an error.
    Skip,
    /// Blank the whole frame rather than risk leaking uncensored content.
    Blank,
}

impl Default for OnDetectFailure {
    fn default() -> Self {
        OnDetectFailure::Blank
    }
}

/// The complete, portable job configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// Schema version for forward-compatible loading.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Chosen model id (see `registry`).
    pub model_id: String,
    /// Resolved per-model setting values (validated against the model entry).
    pub model_settings: SettingValues,
    pub filter: FilterSet,
    pub censor: CensorConfig,
    #[serde(default)]
    pub on_detect_failure: OnDetectFailure,
}

fn default_version() -> u32 {
    1
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            version: default_version(),
            model_id: "nudenet-320n".to_string(),
            model_settings: SettingValues::new(),
            filter: FilterSet::default(),
            censor: CensorConfig::default(),
            on_detect_failure: OnDetectFailure::default(),
        }
    }
}

impl Profile {
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_round_trips_through_json() {
        let p = Profile::default();
        let json = p.to_json().unwrap();
        let back = Profile::from_json(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn fail_closed_is_the_default() {
        assert_eq!(Profile::default().on_detect_failure, OnDetectFailure::Blank);
    }
}
