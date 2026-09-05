//! # ob-censor
//!
//! Renders censor styles over rectangular regions of an RGB8 [`Frame`]
//! (requirement R4, boxes-only in v1). Solid-fill and pixelate are implemented
//! as direct buffer operations (pure, tested here); blur and image-overlay use
//! the `image` crate.
//!
//! The entry point is [`apply`], which the batch pipeline and the future screen
//! tool both call with `frame → detections → censored frame`.

use ob_core::censor::{CensorConfig, CensorStyle};
use ob_core::geometry::{BBox, Detection, Frame};

mod overlay;

#[derive(Debug, thiserror::Error)]
pub enum CensorError {
    #[error("failed to load overlay image `{0}`: {1}")]
    Overlay(String, String),
}

/// Integer pixel rectangle clamped to the frame, half-open `[x0,x1) × [y0,y1)`.
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Rect {
    fn from_bbox(b: &BBox, w: u32, h: u32) -> Option<Rect> {
        let x0 = b.x1.floor().clamp(0.0, w as f32) as usize;
        let y0 = b.y1.floor().clamp(0.0, h as f32) as usize;
        let x1 = b.x2.ceil().clamp(0.0, w as f32) as usize;
        let y1 = b.y2.ceil().clamp(0.0, h as f32) as usize;
        if x1 > x0 && y1 > y0 {
            Some(Rect { x0, y0, x1, y1 })
        } else {
            None
        }
    }
}

/// Apply the censor config to every selected detection, mutating `frame`.
///
/// `selected` are the detections the filter already chose (see
/// `ob_core::filter`). Region padding/rounding from `cfg.shape` is applied here.
pub fn apply(
    frame: &mut Frame,
    selected: &[Detection],
    cfg: &CensorConfig,
) -> Result<(), CensorError> {
    let (w, h) = (frame.width, frame.height);
    for det in selected {
        let padded = det.bbox.expanded(cfg.shape.padding, w as f32, h as f32);
        let style = cfg.style_for(&det.category).clone();
        let Some(rect) = Rect::from_bbox(&padded, w, h) else {
            continue;
        };
        // Snapshot the region first when rounding is requested, so the corners
        // outside the rounded boundary can be restored after drawing.
        let original = (cfg.shape.rounding > 0.0).then(|| copy_rect(frame, &rect));
        match style {
            CensorStyle::SolidFill { color } => solid_fill(frame, &rect, color),
            CensorStyle::Pixelate { block } => pixelate(frame, &rect, block.max(1)),
            CensorStyle::Blur { sigma } => overlay::blur(frame, &rect, sigma),
            CensorStyle::ImageOverlay { path, fit, opacity } => {
                overlay::image_overlay(frame, &rect, &path, fit, opacity)?
            }
        }
        if let Some(orig) = original {
            round_corners(frame, &rect, &orig, cfg.shape.rounding);
        }
    }
    Ok(())
}

/// Copy a rectangle's RGB pixels out of the frame into a packed buffer
/// (`rect_w * rect_h * 3` bytes, row-major).
fn copy_rect(frame: &Frame, r: &Rect) -> Vec<u8> {
    let w = frame.width as usize;
    let rw = r.x1 - r.x0;
    let mut out = Vec::with_capacity(rw * (r.y1 - r.y0) * 3);
    for y in r.y0..r.y1 {
        let row = (y * w + r.x0) * 3;
        out.extend_from_slice(&frame.data[row..row + rw * 3]);
    }
    out
}

/// Restore the original pixels in the four corners that fall outside the
/// rounded-rectangle boundary, giving the censored box rounded corners.
///
/// `rounding` is the corner radius as a fraction of the region's shorter side
/// (0.5 makes the shorter axis a full semicircle — a stadium shape). `orig` is
/// the packed snapshot from [`copy_rect`] for the same rectangle.
fn round_corners(frame: &mut Frame, r: &Rect, orig: &[u8], rounding: f32) {
    let w = frame.width as usize;
    let rw = r.x1 - r.x0;
    let rh = r.y1 - r.y0;
    let radius = (rounding.clamp(0.0, 0.5) as f64) * (rw.min(rh) as f64);
    if radius <= 0.0 {
        return;
    }
    let rad2 = radius * radius;
    for dy in 0..rh {
        for dx in 0..rw {
            // Pixel center in region-local coordinates.
            let xc = dx as f64 + 0.5;
            let yc = dy as f64 + 0.5;
            // Snap to the nearest corner arc center on each axis; interior/edge
            // pixels keep their own coordinate on that axis (distance stays ≤ r).
            let cx = if xc < radius {
                radius
            } else if (rw as f64 - xc) < radius {
                rw as f64 - radius
            } else {
                xc
            };
            let cy = if yc < radius {
                radius
            } else if (rh as f64 - yc) < radius {
                rh as f64 - radius
            } else {
                yc
            };
            let (ex, ey) = (xc - cx, yc - cy);
            if ex * ex + ey * ey > rad2 {
                let p = ((r.y0 + dy) * w + (r.x0 + dx)) * 3;
                let q = (dy * rw + dx) * 3;
                frame.data[p] = orig[q];
                frame.data[p + 1] = orig[q + 1];
                frame.data[p + 2] = orig[q + 2];
            }
        }
    }
}

/// Fill a rectangle with a flat RGB color (alpha ignored for opaque fill).
fn solid_fill(frame: &mut Frame, r: &Rect, color: [u8; 4]) {
    let w = frame.width as usize;
    for y in r.y0..r.y1 {
        let row = (y * w + r.x0) * 3;
        for x in 0..(r.x1 - r.x0) {
            let p = row + x * 3;
            frame.data[p] = color[0];
            frame.data[p + 1] = color[1];
            frame.data[p + 2] = color[2];
        }
    }
}

/// Mosaic: average each `block×block` cell and paint it flat.
fn pixelate(frame: &mut Frame, r: &Rect, block: u32) {
    let w = frame.width as usize;
    let block = block as usize;
    let mut by = r.y0;
    while by < r.y1 {
        let mut bx = r.x0;
        while bx < r.x1 {
            let cx1 = (bx + block).min(r.x1);
            let cy1 = (by + block).min(r.y1);
            // Average the cell.
            let (mut sr, mut sg, mut sb, mut count) = (0u64, 0u64, 0u64, 0u64);
            for y in by..cy1 {
                for x in bx..cx1 {
                    let p = (y * w + x) * 3;
                    sr += frame.data[p] as u64;
                    sg += frame.data[p + 1] as u64;
                    sb += frame.data[p + 2] as u64;
                    count += 1;
                }
            }
            if count > 0 {
                let (ar, ag, ab) = ((sr / count) as u8, (sg / count) as u8, (sb / count) as u8);
                for y in by..cy1 {
                    for x in bx..cx1 {
                        let p = (y * w + x) * 3;
                        frame.data[p] = ar;
                        frame.data[p + 1] = ag;
                        frame.data[p + 2] = ab;
                    }
                }
            }
            bx += block;
        }
        by += block;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::taxonomy::cat;

    fn checkerboard(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                let p = ((y * w + x) * 3) as usize;
                data[p] = v;
                data[p + 1] = v;
                data[p + 2] = v;
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    fn det(b: BBox) -> Detection {
        Detection {
            bbox: b,
            category: cat::FEMALE_GENITALIA_EXPOSED,
            score: 1.0,
        }
    }

    #[test]
    fn solid_fill_paints_region_only() {
        let mut f = checkerboard(4, 4);
        let mut cfg = CensorConfig::default();
        cfg.shape.padding = 0.0;
        cfg.default_style = CensorStyle::SolidFill {
            color: [10, 20, 30, 255],
        };
        apply(&mut f, &[det(BBox::new(0.0, 0.0, 2.0, 2.0))], &cfg).unwrap();
        // (0,0) is inside -> filled.
        assert_eq!(&f.data[0..3], &[10, 20, 30]);
        // (3,3) is outside -> untouched checkerboard value.
        let p = ((3 * 4 + 3) * 3) as usize;
        assert_ne!(&f.data[p..p + 3], &[10, 20, 30]);
    }

    #[test]
    fn rounding_restores_corners_but_keeps_center() {
        let mut f = checkerboard(8, 8);
        let before = f.clone();
        let mut cfg = CensorConfig::default();
        cfg.shape.padding = 0.0;
        cfg.shape.rounding = 0.5; // full stadium: corners cut hard
        cfg.default_style = CensorStyle::SolidFill {
            color: [7, 7, 7, 255],
        };
        apply(&mut f, &[det(BBox::new(0.0, 0.0, 8.0, 8.0))], &cfg).unwrap();
        // The exact corner pixel lies outside the inscribed circle -> restored.
        assert_eq!(&f.data[0..3], &before.data[0..3]);
        // The center is well inside -> filled.
        let c = ((4 * 8 + 4) * 3) as usize;
        assert_eq!(&f.data[c..c + 3], &[7, 7, 7]);
    }

    #[test]
    fn pixelate_makes_cell_uniform() {
        let mut f = checkerboard(4, 4);
        let mut cfg = CensorConfig::default();
        cfg.shape.padding = 0.0;
        cfg.default_style = CensorStyle::Pixelate { block: 4 };
        apply(&mut f, &[det(BBox::new(0.0, 0.0, 4.0, 4.0))], &cfg).unwrap();
        // Whole 4x4 averaged: checkerboard of 0/255 -> 127 or 128 everywhere.
        let first = f.data[0];
        assert!(f.data.iter().all(|&v| v == first));
        assert!((first as i32 - 127).abs() <= 1);
    }
}
