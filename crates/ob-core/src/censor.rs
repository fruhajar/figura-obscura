//! Censor styling configuration (requirement R4).
//!
//! `ob-core` only *describes* how a region should be censored; the actual
//! pixel work lives in `ob-censor`. Styles are box-based in v1 (no masks).

use crate::taxonomy::{Category, Part};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How a matched region is obscured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum CensorStyle {
    /// Flat rectangle of a single RGBA color.
    SolidFill { color: [u8; 4] },
    /// Mosaic; `block` is the pixel size of each mosaic cell.
    Pixelate { block: u32 },
    /// Gaussian blur; higher `sigma` = stronger blur.
    Blur { sigma: f32 },
    /// Composite a preset image over the region.
    ImageOverlay {
        path: String,
        fit: OverlayFit,
        /// 0.0 transparent .. 1.0 opaque.
        opacity: f32,
    },
}

impl Default for CensorStyle {
    fn default() -> Self {
        CensorStyle::Pixelate { block: 16 }
    }
}

/// How an overlay image is scaled into the region rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayFit {
    /// Fill the box, distorting aspect ratio.
    Stretch,
    /// Preserve aspect, covering the box (may crop the overlay).
    Cover,
    /// Preserve aspect, fitting inside the box (may letterbox).
    Contain,
    /// Tile the overlay at native size.
    Tile,
}

/// Geometry applied to every region before the style is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegionShape {
    /// Grow each box by this fraction of its size per side (R4 padding).
    pub padding: f32,
    /// Corner rounding radius as a fraction of the shorter side (0.0 = square).
    pub rounding: f32,
}

impl Default for RegionShape {
    fn default() -> Self {
        Self {
            padding: 0.10,
            rounding: 0.0,
        }
    }
}

/// The complete censor policy: a default style plus optional per-part overrides,
/// and the region geometry. Keyed by [`Part`] so, e.g., genitalia can be solid
/// while faces are pixelated regardless of sex/state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensorConfig {
    pub default_style: CensorStyle,
    #[serde(default)]
    pub per_part: HashMap<Part, CensorStyle>,
    #[serde(default)]
    pub shape: RegionShape,
}

impl Default for CensorConfig {
    fn default() -> Self {
        Self {
            default_style: CensorStyle::default(),
            per_part: HashMap::new(),
            shape: RegionShape::default(),
        }
    }
}

impl CensorConfig {
    /// Resolve the style to use for a given category (part override wins).
    pub fn style_for(&self, category: &Category) -> &CensorStyle {
        self.per_part
            .get(&category.part)
            .unwrap_or(&self.default_style)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taxonomy::cat;

    #[test]
    fn per_part_override_wins() {
        let mut cfg = CensorConfig::default();
        cfg.per_part.insert(
            Part::Genitalia,
            CensorStyle::SolidFill {
                color: [0, 0, 0, 255],
            },
        );
        assert!(matches!(
            cfg.style_for(&cat::FEMALE_GENITALIA_EXPOSED),
            CensorStyle::SolidFill { .. }
        ));
        // Face falls back to default (pixelate).
        assert!(matches!(
            cfg.style_for(&cat::FACE_FEMALE),
            CensorStyle::Pixelate { .. }
        ));
    }
}
