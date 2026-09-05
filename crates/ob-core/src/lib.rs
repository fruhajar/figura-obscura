//! # ob-core
//!
//! The single source of truth for Figura Obscura: the canonical censoring
//! taxonomy, shared geometry/frame types, declarative model registry and
//! settings metadata, filter rules, censor styling, and portable job profiles.
//!
//! This crate performs **no I/O and no inference**. Everything here is pure data
//! and pure functions so it can be reused unchanged by the batch pipeline and by
//! the future real-time screen tool (`ob-screen`, requirement R10).
//!
//! ## Module map
//! - [`taxonomy`] — the canonical `Category` vocabulary (R3).
//! - [`geometry`] — `BBox`, `Detection`, `Frame` (RGB8, codec-free).
//! - [`settings`] — declarative `Setting` metadata powering CLI+GUI+tooltips (R7).
//! - [`registry`] — self-describing downloadable model entries (R7, R8).
//! - [`filter`]   — selective-censoring rules over the taxonomy (R3).
//! - [`censor`]   — box censor styles and per-part overrides (R4).
//! - [`profile`]  — the serializable "what + how" job profile.

pub mod cancel;
pub mod censor;
pub mod filter;
pub mod geometry;
pub mod profile;
pub mod registry;
pub mod settings;
pub mod taxonomy;

// Convenience re-exports for downstream crates.
pub use censor::{CensorConfig, CensorStyle, OverlayFit, RegionShape};
pub use filter::{FilterRule, FilterSet, Match};
pub use geometry::{BBox, Detection, Frame, FrameError};
pub use profile::{OnDetectFailure, Profile};
pub use registry::{builtin_registry, find as find_model, Domain, LabelMap, License, ModelEntry};
pub use settings::{Setting, SettingKind, SettingValue, SettingValues};
pub use taxonomy::{cat, Category, Part, Sex, State, NUDENET_CATEGORIES};
