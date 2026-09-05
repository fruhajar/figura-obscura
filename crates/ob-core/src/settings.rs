//! Declarative settings metadata — the single source that powers CLI flags,
//! GUI widgets **and** GUI tooltips (requirement R7).
//!
//! A [`Setting`] describes one tunable knob completely: its key, value kind
//! (with range/step where relevant), default, unit and human tooltip. `ob-cli`
//! turns each into a clap argument; `ob-gui` turns each into a widget plus a
//! hover tooltip. Model registry entries reuse this same type for per-model
//! knobs, so a new model exposes documented, bounded settings for free.

use serde::{Deserialize, Serialize};

/// The kind of value a setting holds, plus any constraints the UI enforces.
///
/// Compile-time metadata (holds `&'static str` choices), so it is not
/// serialized; only concrete [`SettingValue`]s are persisted.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingKind {
    Float {
        min: f64,
        max: f64,
        step: f64,
    },
    Int {
        min: i64,
        max: i64,
        step: i64,
    },
    Bool,
    /// One of a fixed set of string choices.
    Enum {
        choices: Vec<&'static str>,
    },
    /// An RGBA color, stored as `[r, g, b, a]`.
    Color,
    /// A filesystem path (e.g. an overlay image).
    Path,
}

/// A concrete value for a [`Setting`], matching one [`SettingKind`] variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SettingValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Text(String),
    Color([u8; 4]),
}

impl SettingValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SettingValue::Float(v) => Some(*v),
            SettingValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SettingValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            SettingValue::Text(v) => Some(v),
            _ => None,
        }
    }
}

/// One documented, bounded, defaulted setting. `'static` strings keep the
/// built-in registry a `const`-friendly table. Compile-time metadata, so not
/// serialized (see [`SettingKind`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Setting {
    /// Stable machine key, e.g. `"conf_threshold"`. Also the CLI flag stem.
    pub key: &'static str,
    /// Short human label for the GUI.
    pub label: &'static str,
    pub kind: SettingKind,
    pub default: SettingValue,
    /// Unit shown in the GUI (e.g. "px", "%"), or "" when unitless.
    pub unit: &'static str,
    /// Full explanation shown as a GUI tooltip and CLI long help (R7).
    pub tooltip: &'static str,
}

impl Setting {
    /// Validate/clamp a value against this setting's kind. Returns the coerced
    /// value or an error describing the mismatch.
    pub fn coerce(&self, v: SettingValue) -> Result<SettingValue, SettingError> {
        match (&self.kind, v) {
            (SettingKind::Float { min, max, .. }, SettingValue::Float(f)) => {
                Ok(SettingValue::Float(f.clamp(*min, *max)))
            }
            (SettingKind::Int { min, max, .. }, SettingValue::Int(i)) => {
                Ok(SettingValue::Int(i.clamp(*min, *max)))
            }
            (SettingKind::Bool, v @ SettingValue::Bool(_)) => Ok(v),
            (SettingKind::Color, v @ SettingValue::Color(_)) => Ok(v),
            (SettingKind::Path, v @ SettingValue::Text(_)) => Ok(v),
            (SettingKind::Enum { choices }, SettingValue::Text(t)) => {
                if choices.contains(&t.as_str()) {
                    Ok(SettingValue::Text(t))
                } else {
                    Err(SettingError::BadChoice {
                        key: self.key,
                        got: t,
                        choices: choices.to_vec(),
                    })
                }
            }
            (_, got) => Err(SettingError::TypeMismatch { key: self.key, got }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingError {
    #[error("setting `{key}`: value {got:?} does not match the expected type")]
    TypeMismatch {
        key: &'static str,
        got: SettingValue,
    },
    #[error("setting `{key}`: `{got}` is not one of {choices:?}")]
    BadChoice {
        key: &'static str,
        got: String,
        choices: Vec<&'static str>,
    },
}

/// A resolved bag of setting values keyed by [`Setting::key`].
pub type SettingValues = std::collections::BTreeMap<String, SettingValue>;

/// Build a [`SettingValues`] map filled with every setting's default.
pub fn defaults(settings: &[Setting]) -> SettingValues {
    settings
        .iter()
        .map(|s| (s.key.to_string(), s.default.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conf() -> Setting {
        Setting {
            key: "conf_threshold",
            label: "Confidence",
            kind: SettingKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default: SettingValue::Float(0.2),
            unit: "",
            tooltip: "Minimum detector confidence to keep a detection.",
        }
    }

    #[test]
    fn coerce_clamps_floats() {
        let s = conf();
        assert_eq!(
            s.coerce(SettingValue::Float(5.0)).unwrap(),
            SettingValue::Float(1.0)
        );
    }

    #[test]
    fn coerce_rejects_wrong_type() {
        assert!(conf().coerce(SettingValue::Bool(true)).is_err());
    }

    #[test]
    fn defaults_are_collected() {
        let d = defaults(&[conf()]);
        assert_eq!(d.get("conf_threshold").unwrap().as_f64(), Some(0.2));
    }
}
