//! # ob-detect
//!
//! ONNX-based detection for Obscura. Preprocessing (letterbox), postprocessing (NMS,
//! native-label mapping) and the [`Detector`] abstraction are complete and
//! testable here; the ONNX Runtime session wiring is isolated in [`session`] so
//! the `ort` API surface touched by Obscura is small and swappable.
//!
//! The same [`Detector`] trait is consumed by the batch pipeline and, later, by
//! the real-time screen tool (R10).

pub mod ensemble;
pub mod postprocess;
pub mod preprocess;
pub mod session;
pub mod tile;

use ob_core::geometry::{Detection, Frame};
use ob_core::registry::ModelEntry;
use ob_core::settings::SettingValues;
use ob_core::taxonomy::Category;

/// Anything that turns a frame into canonical detections.
///
/// Implemented by [`session::OnnxDetector`] for real models; trivially mockable
/// in tests. Real-time and batch code depend only on this trait.
pub trait Detector: Send + Sync {
    /// Run detection on a single frame, returning canonical detections in
    /// original-image pixel coordinates.
    fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, DetectError>;

    /// Whether this detector is *capable* of emitting `category` at all.
    ///
    /// Not "did it find one" — "could it ever". A model's [`LabelMap`] fixes
    /// this: the three-class anime detectors can never report buttocks, and a
    /// hypothetical nipple specialist can never report anything else.
    ///
    /// [`crate::ensemble::EnsembleDetector`] needs this to count votes fairly.
    /// Counting a consensus threshold against members that are structurally
    /// unable to vote means adding a specialist *removes* every category
    /// outside its competence — see the tests there.
    ///
    /// Defaults to `true` so a detector that does not declare its capabilities
    /// keeps today's behaviour and is simply always eligible.
    ///
    /// [`LabelMap`]: ob_core::registry::LabelMap
    fn can_emit(&self, _category: Category) -> bool {
        true
    }
}

/// Ordered execution-provider preference (plan §4). Registration attempts each
/// in order and silently falls back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecProvider {
    Cuda,
    Rocm,
    DirectMl,
    CoreMl,
    Cpu,
}

/// The EP order Obscura requests on the current platform. GPU variants are only
/// included when their Cargo feature is enabled.
pub fn preferred_execution_providers() -> Vec<ExecProvider> {
    let mut eps = Vec::new();
    #[cfg(feature = "cuda")]
    eps.push(ExecProvider::Cuda);
    #[cfg(feature = "rocm")]
    eps.push(ExecProvider::Rocm);
    #[cfg(feature = "directml")]
    eps.push(ExecProvider::DirectMl);
    #[cfg(feature = "coreml")]
    eps.push(ExecProvider::CoreMl);
    eps.push(ExecProvider::Cpu); // always last
    eps
}

#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error("failed to load model `{0}`: {1}")]
    Load(String, String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("model file not found for `{0}` (run `obscura models fetch`)")]
    ModelMissing(String),
    #[error(transparent)]
    Frame(#[from] ob_core::geometry::FrameError),
}

/// Build the detector for a model entry, honouring its resampling and tiling
/// settings.
///
/// Returns a boxed [`Detector`] because tiling is a wrapper: with `tiling=off`
/// this is a bare [`session::OnnxDetector`], otherwise that detector inside a
/// [`tile::TiledDetector`]. Callers depend only on the trait, so the batch
/// pipeline, the GUI preview and the future screen tool all pick up tiling
/// without knowing it exists.
pub fn build_detector(
    entry: &ModelEntry,
    settings: &SettingValues,
    model_path: std::path::PathBuf,
) -> Result<Box<dyn Detector>, DetectError> {
    let resolved = resolve_settings(entry, settings);
    let inner = session::OnnxDetector::load(entry, &resolved, model_path)?;

    let text = |k: &str| resolved.get(k).and_then(|v| v.as_str()).map(str::to_owned);
    let mode = text("tiling")
        .and_then(|s| tile::TilingMode::parse(&s))
        .unwrap_or_default();
    if mode == tile::TilingMode::Off {
        return Ok(Box::new(inner));
    }

    let num = |k: &str, d: f64| resolved.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let cfg = tile::TilingConfig {
        mode,
        overlap: num("tile_overlap", 0.25) as f32,
        max_tiles: num("tile_max", 12.0).max(1.0) as usize,
        ..Default::default()
    };
    // The detector may have adopted the model file's real input size, which can
    // differ from the registry's declared one; tile planning must use the real
    // one or the grid is sized for the wrong model.
    let input_size = inner.input_size();
    let nms_iou = inner.nms_iou();
    Ok(Box::new(tile::TiledDetector::new(
        inner, input_size, nms_iou, cfg,
    )))
}

/// Resolve effective per-model settings by layering user values over defaults.
pub fn resolve_settings(entry: &ModelEntry, user: &SettingValues) -> SettingValues {
    let mut merged = ob_core::settings::defaults(&entry.settings);
    for (k, v) in user {
        if merged.contains_key(k) {
            merged.insert(k.clone(), v.clone());
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_last_ep() {
        let eps = preferred_execution_providers();
        assert_eq!(*eps.last().unwrap(), ExecProvider::Cpu);
    }

    #[test]
    fn resolve_settings_layers_over_defaults() {
        let entry = ob_core::registry::find("nudenet-320n").unwrap();
        let mut user = SettingValues::new();
        user.insert(
            "conf_threshold".into(),
            ob_core::settings::SettingValue::Float(0.5),
        );
        let merged = resolve_settings(&entry, &user);
        assert_eq!(merged.get("conf_threshold").unwrap().as_f64(), Some(0.5));
        // Untouched defaults survive.
        assert!(merged.contains_key("nms_iou"));
    }
}
