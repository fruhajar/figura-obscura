//! Letterbox preprocessing — pure and fully testable without ONNX Runtime.
//!
//! YOLO-family models expect a square input with the image scaled to fit and
//! padded (letterboxed). We record the scale/pad so detections can be mapped
//! back to original-image coordinates.
//!
//! # Why the resampler matters
//!
//! Detection runs on a downscaled copy of the frame — a 4K frame reaching a
//! 640px model input is scaled by ~0.167 — so the resampler decides whether a
//! small feature survives the trip. Point sampling (nearest neighbour) reads one
//! source pixel per output pixel and simply misses everything between samples,
//! which is why small subjects in high-resolution sources used to vanish before
//! the model ever saw them.
//!
//! [`Resampler`] implements proper separable filtering with **support scaling**:
//! when minifying, the filter's radius is widened by `1/scale` so every source
//! pixel contributes to some output pixel. That is the difference between
//! anti-aliased downscaling and aliased point sampling, and it matters far more
//! than the choice of kernel.
//!
//! The default is [`Resampler::Triangle`]. With support scaling a triangle
//! filter approaches a box/area average at large minification factors — the
//! standard choice for feeding detectors — while staying close in character to
//! the bilinear resize these models saw during training, which keeps inference
//! preprocessing consistent with training preprocessing. [`Resampler::Lanczos3`]
//! retains slightly more acutance on fine detail but rings around hard edges,
//! and a ringing halo is a false edge the detector can fire on; it is offered
//! for users who want it, not as the default.

use ob_core::geometry::{BBox, Frame};

/// The resampling kernel used when scaling a frame into model input space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Resampler {
    /// Point sampling. Fast, aliases badly on downscale — kept for parity with
    /// the original implementation and for benchmarking, not recommended.
    Nearest,
    /// Bilinear / tent filter, support-scaled on minification. The default.
    #[default]
    Triangle,
    /// Cubic (Catmull-Rom). Slightly sharper than triangle, mild ringing.
    CatmullRom,
    /// Lanczos windowed sinc (a = 3). Sharpest, most prone to ringing halos.
    Lanczos3,
}

impl Resampler {
    /// Kernel radius in *destination*-normalised units, before support scaling.
    fn support(&self) -> f32 {
        match self {
            Resampler::Nearest => 0.5,
            Resampler::Triangle => 1.0,
            Resampler::CatmullRom => 2.0,
            Resampler::Lanczos3 => 3.0,
        }
    }

    /// Evaluate the kernel at `x` (distance from the sample centre).
    fn kernel(&self, x: f32) -> f32 {
        let x = x.abs();
        match self {
            Resampler::Nearest => {
                if x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Resampler::Triangle => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Resampler::CatmullRom => {
                // Catmull-Rom spline (B = 0, C = 0.5).
                if x < 1.0 {
                    1.5 * x * x * x - 2.5 * x * x + 1.0
                } else if x < 2.0 {
                    -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0
                } else {
                    0.0
                }
            }
            Resampler::Lanczos3 => {
                if x < 1e-6 {
                    1.0
                } else if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Human-readable name, used in CLI/GUI pickers.
    pub fn name(&self) -> &'static str {
        match self {
            Resampler::Nearest => "nearest",
            Resampler::Triangle => "triangle",
            Resampler::CatmullRom => "catmull-rom",
            Resampler::Lanczos3 => "lanczos3",
        }
    }

    /// Parse from the CLI/profile spelling; `None` if unrecognised.
    pub fn parse(s: &str) -> Option<Resampler> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nearest" | "point" => Some(Resampler::Nearest),
            "triangle" | "bilinear" | "linear" => Some(Resampler::Triangle),
            "catmull-rom" | "catmullrom" | "cubic" => Some(Resampler::CatmullRom),
            "lanczos3" | "lanczos" => Some(Resampler::Lanczos3),
            _ => None,
        }
    }
}

fn sinc(x: f32) -> f32 {
    let px = std::f32::consts::PI * x;
    px.sin() / px
}

/// Precomputed contributions for one output coordinate along one axis.
struct Contribution {
    start: usize,
    weights: Vec<f32>,
}

/// Build the per-output-pixel filter taps for resizing `src_len` to `dst_len`.
///
/// `scale` is `dst_len / src_len`. When it is below 1 (minification) the filter
/// support is widened by `1/scale`, which is what makes the downscale an
/// average over all covered source pixels rather than a point sample.
fn contributions(src_len: usize, dst_len: usize, filter: Resampler) -> Vec<Contribution> {
    let scale = dst_len as f32 / src_len as f32;
    // Nearest is deliberately exempt: widening its support would turn it into a
    // box/area average, which is a different (and much better) filter wearing
    // the wrong name. Point sampling has to stay point sampling so the
    // comparison against it is honest.
    let filter_scale = if scale < 1.0 && filter != Resampler::Nearest {
        1.0 / scale
    } else {
        1.0
    };
    let support = filter.support() * filter_scale;

    let mut out = Vec::with_capacity(dst_len);
    for d in 0..dst_len {
        // Centre of this destination pixel expressed in source coordinates.
        let center = (d as f32 + 0.5) / scale - 0.5;
        let left = ((center - support).ceil() as isize).max(0) as usize;
        let right = ((center + support).floor() as isize).min(src_len as isize - 1);
        let right = right.max(left as isize) as usize;

        let mut weights = Vec::with_capacity(right - left + 1);
        let mut total = 0.0f32;
        for s in left..=right {
            let w = filter.kernel((s as f32 - center) / filter_scale);
            weights.push(w);
            total += w;
        }
        // Normalise so constant regions keep their value exactly. A degenerate
        // all-zero tap set (possible for Lanczos at exact zero crossings) falls
        // back to the nearest source pixel.
        if total.abs() > 1e-8 {
            for w in &mut weights {
                *w /= total;
            }
            out.push(Contribution {
                start: left,
                weights,
            });
        } else {
            let nearest = (center.round() as isize).clamp(0, src_len as isize - 1) as usize;
            out.push(Contribution {
                start: nearest,
                weights: vec![1.0],
            });
        }
    }
    out
}

/// The transform applied when letterboxing, needed to invert box coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Uniform scale applied to the source image.
    pub scale: f32,
    /// Horizontal padding added on the left (pixels, in model space).
    pub pad_x: f32,
    /// Vertical padding added on the top (pixels, in model space).
    pub pad_y: f32,
    /// Model input side length.
    pub size: u32,
}

impl Letterbox {
    /// Compute the letterbox transform for fitting `frame` into a `size×size`
    /// square while preserving aspect ratio.
    pub fn compute(frame: &Frame, size: u32) -> Letterbox {
        let s = size as f32;
        let scale = (s / frame.width_f()).min(s / frame.height_f());
        let new_w = frame.width_f() * scale;
        let new_h = frame.height_f() * scale;
        Letterbox {
            scale,
            pad_x: (s - new_w) / 2.0,
            pad_y: (s - new_h) / 2.0,
            size,
        }
    }

    /// Map a box from model space back to original-image pixel coordinates.
    pub fn invert(&self, b: &BBox) -> BBox {
        BBox {
            x1: (b.x1 - self.pad_x) / self.scale,
            y1: (b.y1 - self.pad_y) / self.scale,
            x2: (b.x2 - self.pad_x) / self.scale,
            y2: (b.y2 - self.pad_y) / self.scale,
        }
    }
}

/// Produce a letterboxed CHW `f32` tensor (RGB, scaled to 0..1) of length
/// `3 * size * size`, plus the transform, using the default [`Resampler`].
pub fn letterbox_chw(frame: &Frame, size: u32) -> (Vec<f32>, Letterbox) {
    letterbox_chw_with(frame, size, Resampler::default())
}

/// As [`letterbox_chw`], with an explicit resampling kernel. Padding is neutral
/// gray (0.5), matching the 114/255 gray YOLO letterboxing conventionally uses
/// closely enough that it does not shift detections.
pub fn letterbox_chw_with(frame: &Frame, size: u32, filter: Resampler) -> (Vec<f32>, Letterbox) {
    let lb = Letterbox::compute(frame, size);
    let s = size as usize;
    let mut out = vec![0.5f32; 3 * s * s];

    // Destination extent of the image inside the square, in whole pixels.
    let new_w = ((frame.width_f() * lb.scale).round() as usize).clamp(1, s);
    let new_h = ((frame.height_f() * lb.scale).round() as usize).clamp(1, s);
    let off_x = lb.pad_x.round().max(0.0) as usize;
    let off_y = lb.pad_y.round().max(0.0) as usize;

    let resized = resample_rgb(frame, new_w, new_h, filter);

    let (r_plane, gb) = out.split_at_mut(s * s);
    let (g_plane, b_plane) = gb.split_at_mut(s * s);
    for y in 0..new_h {
        let dy = y + off_y;
        if dy >= s {
            break;
        }
        for x in 0..new_w {
            let dx = x + off_x;
            if dx >= s {
                break;
            }
            let src = (y * new_w + x) * 3;
            let dst = dy * s + dx;
            r_plane[dst] = resized[src];
            g_plane[dst] = resized[src + 1];
            b_plane[dst] = resized[src + 2];
        }
    }
    (out, lb)
}

/// Separable filtered resize of an RGB8 frame to `dst_w × dst_h`, returning
/// interleaved RGB `f32` in 0..1.
///
/// Horizontal pass first (into a `dst_w × src_h` intermediate), then vertical.
/// Doing it separably costs `O(w·h·taps)` instead of `O(w·h·taps²)`.
pub fn resample_rgb(frame: &Frame, dst_w: usize, dst_h: usize, filter: Resampler) -> Vec<f32> {
    let src_w = frame.width as usize;
    let src_h = frame.height as usize;
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return vec![0.0; dst_w * dst_h * 3];
    }

    // --- Horizontal ---
    let xc = contributions(src_w, dst_w, filter);
    let mut tmp = vec![0.0f32; dst_w * src_h * 3];
    for y in 0..src_h {
        let row = y * src_w * 3;
        for (dx, c) in xc.iter().enumerate() {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (i, w) in c.weights.iter().enumerate() {
                let sx = c.start + i;
                let p = row + sx * 3;
                r += frame.data[p] as f32 * w;
                g += frame.data[p + 1] as f32 * w;
                b += frame.data[p + 2] as f32 * w;
            }
            let d = (y * dst_w + dx) * 3;
            tmp[d] = r;
            tmp[d + 1] = g;
            tmp[d + 2] = b;
        }
    }

    // --- Vertical ---
    let yc = contributions(src_h, dst_h, filter);
    let mut out = vec![0.0f32; dst_w * dst_h * 3];
    for (dy, c) in yc.iter().enumerate() {
        for x in 0..dst_w {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (i, w) in c.weights.iter().enumerate() {
                let sy = c.start + i;
                let p = (sy * dst_w + x) * 3;
                r += tmp[p] * w;
                g += tmp[p + 1] * w;
                b += tmp[p + 2] * w;
            }
            let d = (dy * dst_w + x) * 3;
            // Cubic and Lanczos kernels overshoot; clamp before normalising.
            out[d] = (r / 255.0).clamp(0.0, 1.0);
            out[d + 1] = (g / 255.0).clamp(0.0, 1.0);
            out[d + 2] = (b / 255.0).clamp(0.0, 1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32) -> Frame {
        Frame::new(w, h, vec![128u8; (w * h * 3) as usize]).unwrap()
    }

    #[test]
    fn wide_image_scales_by_width() {
        let f = frame(200, 100);
        let lb = Letterbox::compute(&f, 640);
        assert!((lb.scale - 3.2).abs() < 1e-4); // 640/200
        assert!(lb.pad_x.abs() < 1e-4);
        assert!(lb.pad_y > 0.0);
    }

    #[test]
    fn invert_is_inverse_of_scale_and_pad() {
        let f = frame(200, 100);
        let lb = Letterbox::compute(&f, 640);
        // A box covering the whole scaled image in model space...
        let model_box = BBox::new(lb.pad_x, lb.pad_y, 640.0 - lb.pad_x, 640.0 - lb.pad_y);
        let orig = lb.invert(&model_box);
        assert!((orig.x1).abs() < 1e-2);
        assert!((orig.x2 - 200.0).abs() < 1e-2);
        assert!((orig.y2 - 100.0).abs() < 1e-2);
    }

    #[test]
    fn tensor_has_correct_length() {
        let (t, _) = letterbox_chw(&frame(50, 70), 320);
        assert_eq!(t.len(), 3 * 320 * 320);
    }

    #[test]
    fn uniform_image_survives_resize_exactly() {
        // A flat field must come back flat under every kernel: weights are
        // normalised, so no ringing or energy loss is allowed here.
        for filter in [
            Resampler::Nearest,
            Resampler::Triangle,
            Resampler::CatmullRom,
            Resampler::Lanczos3,
        ] {
            let out = resample_rgb(&frame(97, 61), 32, 20, filter);
            for v in &out {
                assert!(
                    (*v - 128.0 / 255.0).abs() < 1e-4,
                    "{} shifted a flat field to {v}",
                    filter.name()
                );
            }
        }
    }

    /// The regression this whole module exists for: a small bright feature in a
    /// large dark frame must still be visible after an 8x downscale. Nearest
    /// neighbour drops it entirely whenever it falls between sample points.
    #[test]
    fn small_feature_survives_heavy_downscale() {
        let (w, h) = (512u32, 512u32);
        let mut data = vec![0u8; (w * h * 3) as usize];
        // A 4x4 white square at (101, 101) — deliberately not on an 8px grid.
        for y in 101..105 {
            for x in 101..105 {
                let p = ((y * w + x) * 3) as usize;
                data[p] = 255;
                data[p + 1] = 255;
                data[p + 2] = 255;
            }
        }
        let f = Frame::new(w, h, data).unwrap();

        let nearest = resample_rgb(&f, 64, 64, Resampler::Nearest);
        let triangle = resample_rgb(&f, 64, 64, Resampler::Triangle);

        let energy = |v: &[f32]| v.iter().sum::<f32>();
        // Nearest samples every 8th pixel, missing the 4px feature completely.
        assert_eq!(
            energy(&nearest),
            0.0,
            "nearest unexpectedly kept the feature"
        );
        // The filtered path preserves the feature's energy (16 white px / 64 per
        // output pixel => a visible ~0.25-intensity blob).
        assert!(
            energy(&triangle) > 0.5,
            "triangle lost the feature: energy {}",
            energy(&triangle)
        );
    }

    #[test]
    fn letterbox_places_image_inside_padding() {
        // A 2:1 image letterboxed into a square keeps gray bars top and bottom.
        let f = frame(64, 32);
        let (t, lb) = letterbox_chw_with(&f, 64, Resampler::Triangle);
        let s = 64usize;
        assert!(lb.pad_y > 0.0);
        // Top row is padding...
        assert!((t[0] - 0.5).abs() < 1e-6);
        // ...and the centre row is image (128/255).
        let mid = (s / 2) * s + s / 2;
        assert!((t[mid] - 128.0 / 255.0).abs() < 1e-3);
    }

    #[test]
    fn resampler_parses_its_own_names() {
        for r in [
            Resampler::Nearest,
            Resampler::Triangle,
            Resampler::CatmullRom,
            Resampler::Lanczos3,
        ] {
            assert_eq!(Resampler::parse(r.name()), Some(r));
        }
        assert_eq!(Resampler::parse("bilinear"), Some(Resampler::Triangle));
        assert_eq!(Resampler::parse("nope"), None);
    }
}
