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

use ob_core::geometry::{BBox, Frame, Region};

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

/// Precomputed filter taps for resizing one axis, flattened into three vectors.
///
/// The obvious shape for this is a `Vec` of per-destination structs each owning
/// its own weight `Vec`, which is what this was. For a 640px target that is 640
/// separate allocations, rebuilt on every pass — and a tiled 4K frame pays it
/// twice per pass across thirteen passes. One allocation per axis instead, with
/// `weights[offsets[d]..offsets[d + 1]]` holding destination `d`'s taps.
struct Taps {
    /// First source index contributing to each destination position.
    starts: Vec<u32>,
    /// Where each destination's run of weights begins. `dst_len + 1` entries,
    /// so the last one closes the final run.
    offsets: Vec<u32>,
    weights: Vec<f32>,
}

impl Taps {
    /// Build the taps for resizing `src_len` to `dst_len`.
    ///
    /// `scale` is `dst_len / src_len`. When it is below 1 (minification) the
    /// filter support is widened by `1/scale`, which is what makes the
    /// downscale an average over all covered source pixels rather than a point
    /// sample.
    fn build(src_len: usize, dst_len: usize, filter: Resampler) -> Taps {
        let scale = dst_len as f32 / src_len as f32;
        // Nearest is deliberately exempt: widening its support would turn it
        // into a box/area average, which is a different (and much better)
        // filter wearing the wrong name. Point sampling has to stay point
        // sampling so the comparison against it is honest.
        let filter_scale = if scale < 1.0 && filter != Resampler::Nearest {
            1.0 / scale
        } else {
            1.0
        };
        let support = filter.support() * filter_scale;

        let mut starts = Vec::with_capacity(dst_len);
        let mut offsets = Vec::with_capacity(dst_len + 1);
        let mut weights = Vec::with_capacity(dst_len * 4);
        offsets.push(0);

        for d in 0..dst_len {
            // Centre of this destination pixel expressed in source coordinates.
            let center = (d as f32 + 0.5) / scale - 0.5;
            let left = ((center - support).ceil() as isize).max(0) as usize;
            let right = ((center + support).floor() as isize).min(src_len as isize - 1);
            let right = right.max(left as isize) as usize;

            let run_start = weights.len();
            let mut total = 0.0f32;
            for sx in left..=right {
                let w = filter.kernel((sx as f32 - center) / filter_scale);
                weights.push(w);
                total += w;
            }
            // Normalise so constant regions keep their value exactly. A
            // degenerate all-zero tap set (possible for Lanczos at exact zero
            // crossings) falls back to the nearest source pixel.
            if total.abs() > 1e-8 {
                for w in &mut weights[run_start..] {
                    *w /= total;
                }
                starts.push(left as u32);
            } else {
                weights.truncate(run_start);
                weights.push(1.0);
                let nearest = (center.round() as isize).clamp(0, src_len as isize - 1) as usize;
                starts.push(nearest as u32);
            }
            offsets.push(weights.len() as u32);
        }

        Taps {
            starts,
            offsets,
            weights,
        }
    }

    /// Destination `d`'s first source index and its weights.
    #[inline]
    fn get(&self, d: usize) -> (usize, &[f32]) {
        let a = self.offsets[d] as usize;
        let b = self.offsets[d + 1] as usize;
        (self.starts[d] as usize, &self.weights[a..b])
    }
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
        Letterbox::for_size(frame.width_f(), frame.height_f(), size)
    }

    /// As [`Letterbox::compute`], from dimensions alone.
    ///
    /// A tile is a sub-rectangle of a frame, not a frame, so the transform has
    /// to be derivable from the window's size rather than from an owned image.
    /// Taking dimensions is also what lets cost estimation ask the same
    /// question without decoding anything.
    pub fn for_size(w: f32, h: f32, size: u32) -> Letterbox {
        let s = size as f32;
        let scale = (s / w).min(s / h);
        let new_w = w * scale;
        let new_h = h * scale;
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

/// Above this many source pixels, the filter passes are worth spreading across
/// rayon's pool. Below it the split costs more than it saves: a 320px tile is
/// 102k pixels and stays on one thread, while a 4K frame is 8.3M and does not.
const PARALLEL_PIXEL_THRESHOLD: usize = 1 << 20;

/// Whether to split this much work across the pool.
///
/// Size is only half the question. The other half is who is asking: the batch
/// engine already runs whole images in parallel, so a letterbox called from one
/// of its workers is on a machine whose cores are *fully committed*, and
/// splitting it further buys nothing while adding scheduling overhead and cache
/// pressure to every other worker. Measured, that nesting cost about a third of
/// the image throughput — enough to swallow an eightfold speedup in this very
/// function and still come out behind.
///
/// `current_thread_index` answers it exactly: `Some` means a rayon worker is
/// calling, so the pool is already busy; `None` means a plain thread is — the
/// video pipeline, a GUI preview, a single-file CLI run — and the cores are
/// idle unless this function uses them.
fn worth_splitting(pixels: usize) -> bool {
    pixels >= PARALLEL_PIXEL_THRESHOLD && rayon::current_thread_index().is_none()
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
    let s = size as usize;
    let mut out = vec![0.5f32; 3 * s * s];
    let lb = letterbox_into(frame, Region::whole(frame), size, filter, &mut out);
    (out, lb)
}

/// Scale one region of `frame` into a `size × size` planar CHW tensor, written
/// directly into `out` (which must be `3 * size * size` long and pre-filled
/// with the 0.5 padding value). Returns the transform for inverting boxes.
///
/// Writing straight into the caller's planar buffer is the point. The previous
/// route allocated an interleaved `dst_w × dst_h × 3` result, then copied it
/// pixel by pixel into the planar tensor — two costs that exist only because
/// the two stages did not know about each other. It also means a batch of
/// windows can be assembled into one contiguous tensor with no per-window
/// allocation at all (see [`letterbox_batch`]).
pub fn letterbox_into(
    frame: &Frame,
    src: Region,
    size: u32,
    filter: Resampler,
    out: &mut [f32],
) -> Letterbox {
    let s = size as usize;
    // The transform has to describe what is actually resampled, so the clamp
    // comes first: a window running off the frame contributes only its
    // overlapping part, and a letterbox computed from the requested size would
    // invert boxes through a scale that was never applied. This mirrors what
    // cropping the window out into its own frame used to do implicitly.
    let clamped = src.clamped(frame.width, frame.height);
    let src = clamped.unwrap_or(src);
    let lb = Letterbox::for_size(src.width_f().max(1.0), src.height_f().max(1.0), size);
    if clamped.is_none() || out.len() < 3 * s * s {
        return lb;
    }

    // Destination extent of the image inside the square, in whole pixels.
    let new_w = ((src.width_f() * lb.scale).round() as usize).clamp(1, s);
    let new_h = ((src.height_f() * lb.scale).round() as usize).clamp(1, s);
    let off_x = lb.pad_x.round().max(0.0) as usize;
    let off_y = lb.pad_y.round().max(0.0) as usize;

    let (r_plane, gb) = out.split_at_mut(s * s);
    let (g_plane, b_plane) = gb.split_at_mut(s * s);
    let planes = [r_plane, g_plane, b_plane];

    // A tile is cropped at exactly the model's input side, so the general path
    // would run the entire separable-filter machinery to perform an identity
    // resize. Lanczos is excluded because its taps at scale 1 are not exactly
    // (0, 1, 0): `sinc(1)` evaluates to about -2.8e-8 in f32 rather than zero,
    // so the general path there is very slightly not the identity, and taking
    // the shortcut would change its output. The other three kernels are exact.
    if new_w == src.w as usize && new_h == src.h as usize && filter != Resampler::Lanczos3 {
        copy_region(frame, src, planes, off_x, off_y, s);
        return lb;
    }

    let xt = Taps::build(src.w as usize, new_w, filter);
    let yt = Taps::build(src.h as usize, new_h, filter);
    let parallel = worth_splitting(src.w as usize * src.h as usize);

    let mut tmp = vec![0.0f32; new_w * src.h as usize * 3];
    pass_h(frame, src, new_w, &xt, &mut tmp, parallel);
    pass_v(&tmp, new_w, new_h, &yt, planes, off_x, off_y, s, parallel);
    lb
}

/// The scale-1 shortcut: convert `src` straight from `u8` to `f32` in 0..1 and
/// scatter it into the three planes. No filtering, no intermediate.
fn copy_region(
    frame: &Frame,
    src: Region,
    planes: [&mut [f32]; 3],
    off_x: usize,
    off_y: usize,
    s: usize,
) {
    let [r_plane, g_plane, b_plane] = planes;
    let fw = frame.width as usize;
    for y in 0..src.h as usize {
        let dy = y + off_y;
        if dy >= s {
            break;
        }
        let row = ((src.y as usize + y) * fw + src.x as usize) * 3;
        let dst = dy * s + off_x;
        let cols = (src.w as usize).min(s.saturating_sub(off_x));
        for x in 0..cols {
            let p = row + x * 3;
            r_plane[dst + x] = frame.data[p] as f32 / 255.0;
            g_plane[dst + x] = frame.data[p + 1] as f32 / 255.0;
            b_plane[dst + x] = frame.data[p + 2] as f32 / 255.0;
        }
    }
}

/// Horizontal pass: every row of `src` filtered to `dst_w`, into an
/// interleaved `dst_w × src.h` `f32` intermediate still on the 0..255 scale.
fn pass_h(frame: &Frame, src: Region, dst_w: usize, xt: &Taps, tmp: &mut [f32], parallel: bool) {
    let fw = frame.width as usize;
    let row_of = |y: usize, out_row: &mut [f32]| {
        let row = ((src.y as usize + y) * fw + src.x as usize) * 3;
        for dx in 0..dst_w {
            let (start, weights) = xt.get(dx);
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (i, w) in weights.iter().enumerate() {
                let p = row + (start + i) * 3;
                r += frame.data[p] as f32 * w;
                g += frame.data[p + 1] as f32 * w;
                b += frame.data[p + 2] as f32 * w;
            }
            let d = dx * 3;
            out_row[d] = r;
            out_row[d + 1] = g;
            out_row[d + 2] = b;
        }
    };

    if parallel {
        use rayon::prelude::*;
        tmp.par_chunks_mut(dst_w * 3)
            .enumerate()
            .for_each(|(y, out_row)| row_of(y, out_row));
    } else {
        for (y, out_row) in tmp.chunks_mut(dst_w * 3).enumerate() {
            row_of(y, out_row);
        }
    }
}

/// Vertical pass: reduce the intermediate to `dst_h` rows and write the result
/// into the three planes at the letterbox offset, normalised to 0..1.
#[allow(clippy::too_many_arguments)]
fn pass_v(
    tmp: &[f32],
    dst_w: usize,
    dst_h: usize,
    yt: &Taps,
    planes: [&mut [f32]; 3],
    off_x: usize,
    off_y: usize,
    s: usize,
    parallel: bool,
) {
    let [r_plane, g_plane, b_plane] = planes;
    let cols = dst_w.min(s.saturating_sub(off_x));
    let rows = dst_h.min(s.saturating_sub(off_y));

    let one_row = |dy: usize, r_row: &mut [f32], g_row: &mut [f32], b_row: &mut [f32]| {
        let (start, weights) = yt.get(dy);
        for x in 0..cols {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (i, w) in weights.iter().enumerate() {
                let p = ((start + i) * dst_w + x) * 3;
                r += tmp[p] * w;
                g += tmp[p + 1] * w;
                b += tmp[p + 2] * w;
            }
            // Cubic and Lanczos kernels overshoot; clamp before normalising.
            r_row[x] = (r / 255.0).clamp(0.0, 1.0);
            g_row[x] = (g / 255.0).clamp(0.0, 1.0);
            b_row[x] = (b / 255.0).clamp(0.0, 1.0);
        }
    };

    // Each destination row writes a disjoint run of each plane, so the rows can
    // be handed out independently. `chunks_mut(s)` walks the plane a row at a
    // time; `skip(off_y)` starts it at the letterbox offset.
    if parallel {
        use rayon::prelude::*;
        r_plane
            .par_chunks_mut(s)
            .zip(g_plane.par_chunks_mut(s))
            .zip(b_plane.par_chunks_mut(s))
            .skip(off_y)
            .take(rows)
            .enumerate()
            .for_each(|(dy, ((r_row, g_row), b_row))| {
                one_row(
                    dy,
                    &mut r_row[off_x..],
                    &mut g_row[off_x..],
                    &mut b_row[off_x..],
                );
            });
    } else {
        for (dy, ((r_row, g_row), b_row)) in r_plane
            .chunks_mut(s)
            .zip(g_plane.chunks_mut(s))
            .zip(b_plane.chunks_mut(s))
            .skip(off_y)
            .take(rows)
            .enumerate()
        {
            one_row(
                dy,
                &mut r_row[off_x..],
                &mut g_row[off_x..],
                &mut b_row[off_x..],
            );
        }
    }
}

/// Letterbox several windows of one frame into a single contiguous batch
/// tensor, shaped `[windows.len(), 3, size, size]`.
///
/// This is what makes batched inference possible: one allocation, one tensor,
/// one `Session::run` for a whole tile grid instead of thirteen of each. The
/// per-window [`Letterbox`] transforms come back alongside, because each window
/// needs its own to map boxes back into frame coordinates.
pub fn letterbox_batch(
    frame: &Frame,
    windows: &[Region],
    size: u32,
    filter: Resampler,
) -> (Vec<f32>, Vec<Letterbox>) {
    let s = size as usize;
    let stride = 3 * s * s;
    let mut out = vec![0.5f32; stride * windows.len()];
    let mut boxes = vec![Letterbox::for_size(1.0, 1.0, size); windows.len()];

    // One window is the untiled case: leave the cores to the row-level split
    // inside, which is where the work actually is for a full 4K frame. Several
    // windows are tiles, each too small to be worth splitting by rows, so the
    // windows themselves are the unit of work — but only when the caller is not
    // itself a rayon worker, for the reason [`worth_splitting`] gives.
    if windows.len() > 1 && rayon::current_thread_index().is_none() {
        use rayon::prelude::*;
        out.par_chunks_mut(stride)
            .zip(windows.par_iter())
            .zip(boxes.par_iter_mut())
            .for_each(|((slot, w), lb)| {
                *lb = letterbox_into(frame, *w, size, filter, slot);
            });
    } else {
        for ((slot, w), lb) in out
            .chunks_mut(stride)
            .zip(windows.iter())
            .zip(boxes.iter_mut())
        {
            *lb = letterbox_into(frame, *w, size, filter, slot);
        }
    }
    (out, boxes)
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

    let src = Region::whole(frame);
    let xt = Taps::build(src_w, dst_w, filter);
    let yt = Taps::build(src_h, dst_h, filter);
    let parallel = worth_splitting(src_w * src_h);

    let mut tmp = vec![0.0f32; dst_w * src_h * 3];
    pass_h(frame, src, dst_w, &xt, &mut tmp, parallel);

    // Interleaved output, so this cannot share `pass_v` (which writes planes).
    let mut out = vec![0.0f32; dst_w * dst_h * 3];
    for (dy, out_row) in out.chunks_mut(dst_w * 3).enumerate() {
        let (start, weights) = yt.get(dy);
        for x in 0..dst_w {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (i, w) in weights.iter().enumerate() {
                let p = ((start + i) * dst_w + x) * 3;
                r += tmp[p] * w;
                g += tmp[p + 1] * w;
                b += tmp[p + 2] * w;
            }
            let d = x * 3;
            out_row[d] = (r / 255.0).clamp(0.0, 1.0);
            out_row[d + 1] = (g / 255.0).clamp(0.0, 1.0);
            out_row[d + 2] = (b / 255.0).clamp(0.0, 1.0);
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

    /// The pre-rewrite algorithm, transcribed literally: per-destination-pixel
    /// weight vectors, a separable resize into an interleaved buffer, then a
    /// pixel-by-pixel copy into the planar tensor.
    ///
    /// It is here so the rewrite can be held to bit-identity rather than to
    /// "looks about right". Every floating-point operation below happens in the
    /// same order as the shipped code, which is what makes `assert_eq!` on
    /// `f32` the correct assertion instead of an approximate compare.
    fn naive_letterbox(frame: &Frame, size: u32, filter: Resampler) -> Vec<f32> {
        struct Contribution {
            start: usize,
            weights: Vec<f32>,
        }
        fn contributions(src_len: usize, dst_len: usize, filter: Resampler) -> Vec<Contribution> {
            let scale = dst_len as f32 / src_len as f32;
            let filter_scale = if scale < 1.0 && filter != Resampler::Nearest {
                1.0 / scale
            } else {
                1.0
            };
            let support = filter.support() * filter_scale;
            let mut out = Vec::with_capacity(dst_len);
            for d in 0..dst_len {
                let center = (d as f32 + 0.5) / scale - 0.5;
                let left = ((center - support).ceil() as isize).max(0) as usize;
                let right = ((center + support).floor() as isize).min(src_len as isize - 1);
                let right = right.max(left as isize) as usize;
                let mut weights = Vec::with_capacity(right - left + 1);
                let mut total = 0.0f32;
                for sx in left..=right {
                    let w = filter.kernel((sx as f32 - center) / filter_scale);
                    weights.push(w);
                    total += w;
                }
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

        let lb = Letterbox::compute(frame, size);
        let s = size as usize;
        let mut out = vec![0.5f32; 3 * s * s];
        let new_w = ((frame.width_f() * lb.scale).round() as usize).clamp(1, s);
        let new_h = ((frame.height_f() * lb.scale).round() as usize).clamp(1, s);
        let off_x = lb.pad_x.round().max(0.0) as usize;
        let off_y = lb.pad_y.round().max(0.0) as usize;

        let (src_w, src_h) = (frame.width as usize, frame.height as usize);
        let xc = contributions(src_w, new_w, filter);
        let mut tmp = vec![0.0f32; new_w * src_h * 3];
        for y in 0..src_h {
            let row = y * src_w * 3;
            for (dx, c) in xc.iter().enumerate() {
                let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
                for (i, w) in c.weights.iter().enumerate() {
                    let p = row + (c.start + i) * 3;
                    r += frame.data[p] as f32 * w;
                    g += frame.data[p + 1] as f32 * w;
                    b += frame.data[p + 2] as f32 * w;
                }
                let d = (y * new_w + dx) * 3;
                tmp[d] = r;
                tmp[d + 1] = g;
                tmp[d + 2] = b;
            }
        }
        let yc = contributions(src_h, new_h, filter);
        let mut resized = vec![0.0f32; new_w * new_h * 3];
        for (dy, c) in yc.iter().enumerate() {
            for x in 0..new_w {
                let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
                for (i, w) in c.weights.iter().enumerate() {
                    let p = ((c.start + i) * new_w + x) * 3;
                    r += tmp[p] * w;
                    g += tmp[p + 1] * w;
                    b += tmp[p + 2] * w;
                }
                let d = (dy * new_w + x) * 3;
                resized[d] = (r / 255.0).clamp(0.0, 1.0);
                resized[d + 1] = (g / 255.0).clamp(0.0, 1.0);
                resized[d + 2] = (b / 255.0).clamp(0.0, 1.0);
            }
        }

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
        out
    }

    /// A frame with structure at several scales, so a resampling difference has
    /// somewhere to show up. A flat field would pass any implementation.
    fn textured(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = ((y * w + x) * 3) as usize;
                data[p] = (x.wrapping_mul(7) ^ y.wrapping_mul(13)) as u8;
                data[p + 1] = ((x / 3).wrapping_add(y / 5) % 251) as u8;
                data[p + 2] = if (x / 2 + y / 2) % 3 == 0 { 240 } else { 12 };
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    #[test]
    fn the_rewritten_letterbox_is_bit_identical_to_the_old_one() {
        // The whole justification for the rewrite is that it is faster and
        // changes nothing. `assert_eq!` on raw f32 is the point: anything
        // weaker would let a reordered accumulation through, and a detector fed
        // subtly different pixels is a detector with subtly different output.
        for filter in [
            Resampler::Nearest,
            Resampler::Triangle,
            Resampler::CatmullRom,
            Resampler::Lanczos3,
        ] {
            for (w, h, size) in [
                (1920u32, 1080u32, 320u32), // heavy minification, wide
                (200, 700, 320),            // heavy minification, tall
                (320, 320, 320),            // the scale-1 tile case
                (97, 61, 64),               // odd sizes, magnification
                (64, 32, 64),               // exact width fit, padded height
            ] {
                let f = textured(w, h);
                let (got, _) = letterbox_chw_with(&f, size, filter);
                let want = naive_letterbox(&f, size, filter);
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{} {w}x{h}->{size}: length",
                    filter.name()
                );
                let diff = got.iter().zip(&want).position(|(a, b)| a != b);
                assert!(
                    diff.is_none(),
                    "{} {w}x{h}->{size}: first difference at index {:?} ({:?} vs {:?})",
                    filter.name(),
                    diff,
                    diff.map(|i| got[i]),
                    diff.map(|i| want[i]),
                );
            }
        }
    }

    #[test]
    fn a_region_letterboxes_exactly_as_that_crop_would() {
        // This is the property that lets tiled detection stop calling
        // `Frame::crop`: reading a sub-rectangle in place has to produce the
        // same tensor as copying it out into its own frame first. If it did
        // not, removing the copy would change detections.
        let f = textured(900, 640);
        for filter in [Resampler::Triangle, Resampler::Lanczos3] {
            for r in [
                Region::new(0, 0, 320, 320),
                Region::new(101, 37, 320, 320),
                Region::new(580, 320, 320, 320),
                Region::new(11, 13, 417, 260), // not square: exercises padding
            ] {
                let s = 320usize;
                let mut direct = vec![0.5f32; 3 * s * s];
                let lb_direct = letterbox_into(&f, r, 320, filter, &mut direct);

                let cropped = f.crop(r.x, r.y, r.w, r.h).unwrap();
                let (via_crop, lb_crop) = letterbox_chw_with(&cropped, 320, filter);

                assert_eq!(lb_direct, lb_crop, "{:?} {}: transform", r, filter.name());
                assert!(
                    direct == via_crop,
                    "{:?} {}: tensor differs from the cropped equivalent",
                    r,
                    filter.name()
                );
            }
        }
    }

    #[test]
    fn a_batch_matches_letterboxing_each_window_on_its_own() {
        // Batching is only sound if assembling N windows into one tensor is
        // exactly N independent letterboxes laid end to end.
        let f = textured(1500, 900);
        let windows = [
            Region::whole(&f),
            Region::new(0, 0, 320, 320),
            Region::new(400, 200, 320, 320),
            Region::new(1180, 580, 320, 320),
        ];
        let (batch, boxes) = letterbox_batch(&f, &windows, 320, Resampler::Triangle);
        let stride = 3 * 320 * 320;
        assert_eq!(batch.len(), stride * windows.len());
        assert_eq!(boxes.len(), windows.len());

        for (i, w) in windows.iter().enumerate() {
            let mut one = vec![0.5f32; stride];
            let lb = letterbox_into(&f, *w, 320, Resampler::Triangle, &mut one);
            assert_eq!(lb, boxes[i], "window {i}: transform");
            assert!(
                batch[i * stride..(i + 1) * stride] == one[..],
                "window {i}: batched slice differs from the standalone tensor"
            );
        }
    }

    #[test]
    fn a_window_running_off_the_edge_is_clipped_not_panicked() {
        // The tile planner deliberately places the last row and column flush
        // with the far edge, and a caller may hand in something wider still.
        let f = textured(100, 80);
        let mut out = vec![0.5f32; 3 * 64 * 64];
        let lb = letterbox_into(
            &f,
            Region::new(90, 70, 320, 320),
            64,
            Resampler::Triangle,
            &mut out,
        );
        assert_eq!(lb.size, 64);
        // The transform must describe the 10x10 overlap that was actually
        // resampled, not the 320x320 that was asked for -- otherwise boxes
        // invert through a scale nothing was ever scaled by.
        let clipped = f.crop(90, 70, 320, 320).unwrap();
        assert_eq!((clipped.width, clipped.height), (10, 10));
        assert_eq!(lb, Letterbox::compute(&clipped, 64));
        // Something was written: the clipped 10x10 corner, not the padding.
        assert!(out.iter().any(|v| (*v - 0.5).abs() > 1e-6));

        // Entirely outside leaves the padding untouched rather than panicking.
        let mut empty = vec![0.5f32; 3 * 64 * 64];
        letterbox_into(
            &f,
            Region::new(500, 500, 32, 32),
            64,
            Resampler::Triangle,
            &mut empty,
        );
        assert!(empty.iter().all(|v| *v == 0.5));
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
