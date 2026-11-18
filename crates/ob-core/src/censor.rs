//! Censor styling configuration (requirement R4).
//!
//! `ob-core` only *describes* how a region should be censored; the actual
//! pixel work lives in `ob-censor`. Styles are box-based in v1 (no masks).

use crate::geometry::{BBox, Detection};
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

/// One region to cover, with the exact appearance it gets: the **final** rect,
/// already padded, and the style already resolved.
///
/// This is the hand-off between deciding what to censor and doing it. A
/// `Detection` still needs a policy applied to it — which style, how much
/// padding — and two code paths applying that policy separately is how a
/// preview comes to disagree with the file it claims to preview. A `CensorBox`
/// has no policy left in it, so anything holding one draws the same pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensorBox {
    /// The rectangle to cover, in frame coordinates. Padding is already in it.
    pub rect: BBox,
    pub style: CensorStyle,
    /// Corner radius as a fraction of the shorter side (0.0 = square).
    pub rounding: f32,
    /// The category the style was resolved from.
    ///
    /// Kept for two things that happen after resolution: replaying a video's
    /// tracker at commit, which matches boxes within a category, and telling
    /// the user in a review what the model thought it found. `None` for a box
    /// drawn by hand, which belongs to no taxonomy — and which must therefore
    /// never be matched against a detected one.
    #[serde(default)]
    pub category: Option<Category>,
}

impl CensorBox {
    /// A box drawn by hand: no category, and the rect is exactly what was drawn.
    pub fn manual(rect: BBox, style: CensorStyle, rounding: f32) -> Self {
        Self {
            rect,
            style,
            rounding,
            category: None,
        }
    }
}

impl CensorConfig {
    /// Resolve the style to use for a given category (part override wins).
    ///
    /// [`cat::UNCLASSIFIED`](crate::taxonomy::cat::UNCLASSIFIED) always takes
    /// the default. It borrows `Part::Face` to have a value at all, and a
    /// per-part Face override would otherwise reach a box someone drew over
    /// something that is not a face.
    pub fn style_for(&self, category: &Category) -> &CensorStyle {
        if *category == crate::taxonomy::cat::UNCLASSIFIED {
            return &self.default_style;
        }
        self.per_part
            .get(&category.part)
            .unwrap_or(&self.default_style)
    }

    /// Apply the whole policy to a set of detections, yielding the boxes that
    /// will actually be drawn.
    ///
    /// `w`/`h` are the frame's dimensions, which padding is clamped against so
    /// a grown box never leaves the frame.
    pub fn resolve(&self, dets: &[Detection], w: f32, h: f32) -> Vec<CensorBox> {
        dets.iter()
            .map(|d| CensorBox {
                rect: d.bbox.expanded(self.shape.padding, w, h),
                style: self.style_for(&d.category).clone(),
                rounding: self.shape.rounding,
                category: Some(d.category),
            })
            .collect()
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
