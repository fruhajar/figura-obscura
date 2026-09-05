//! Procedural generation of the Figura Obscura app icon.
//!
//! The mark is a **redaction bar over a mosaic field**: the two visual idioms
//! for "this has been covered up", and both survive being shrunk to a 16px
//! taskbar glyph, which a literal illustration would not.
//!
//! Everything is drawn analytically rather than loaded from an artboard so the
//! whole icon set is reproducible from `cargo xtask icons` with no design tool
//! in the loop — the build container has neither ImageMagick nor a rasteriser.

use image::{Rgba, RgbaImage};

/// Supersampling factor. The shapes are axis-aligned rounded rectangles, so
/// 4× box-filtered down is indistinguishable from analytic coverage and is a
/// great deal less code.
const SS: u32 = 4;

/// Linear RGB colour with alpha, 0.0..=1.0. Compositing in linear light keeps
/// the gradient from going muddy through the middle.
#[derive(Clone, Copy)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    /// From 8-bit sRGB, the form the palette is written in.
    const fn srgb(r: u8, g: u8, b: u8, a: f32) -> Self {
        Self {
            r: r as f32,
            g: g as f32,
            b: b as f32,
            a,
        }
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

/// The same accent the GUI theme uses, so the icon and the app agree.
const GRAD_TOP: Color = Color::srgb(0x8B, 0x7C, 0xFF, 1.0);
const GRAD_BOTTOM: Color = Color::srgb(0x4B, 0x35, 0xC4, 1.0);
const MOSAIC: Color = Color::srgb(0xF2, 0xF0, 0xFF, 1.0);
const BAR: Color = Color::srgb(0x14, 0x11, 0x2B, 1.0);

/// Signed coverage of a rounded rectangle at a point, 0.0..=1.0.
///
/// Returns partial coverage within one pixel of the edge, which is what makes
/// the supersampled result clean rather than stair-stepped.
fn rounded_rect_coverage(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32, r: f32) -> f32 {
    let r = r.min(w * 0.5).min(h * 0.5);
    // Distance from the rounded-rect boundary (negative = inside).
    let cx = (x + r).max(px.min(x + w - r));
    let cy = (y + r).max(py.min(y + h - r));
    let dx = px - cx;
    let dy = py - cy;
    let dist = (dx * dx + dy * dy).sqrt() - r;
    // One-pixel-wide linear ramp across the edge.
    (0.5 - dist).clamp(0.0, 1.0)
}

/// Alpha-over composite in place.
fn blend(dst: &mut Color, src: Color, coverage: f32) {
    let a = src.a * coverage;
    if a <= 0.0 {
        return;
    }
    dst.r = src.r * a + dst.r * (1.0 - a);
    dst.g = src.g * a + dst.g * (1.0 - a);
    dst.b = src.b * a + dst.b * (1.0 - a);
    dst.a = a + dst.a * (1.0 - a);
}

/// Per-cell opacity of the mosaic field.
///
/// Hand-tuned rather than random: a fixed pattern that reads as a censored
/// *blob* — denser in the middle, fading at the corners — instead of noise,
/// and it renders identically on every machine.
const MOSAIC_ALPHA: [[f32; 5]; 5] = [
    [0.00, 0.18, 0.30, 0.14, 0.00],
    [0.16, 0.42, 0.66, 0.38, 0.12],
    [0.28, 0.70, 0.95, 0.62, 0.26],
    [0.14, 0.46, 0.72, 0.40, 0.16],
    [0.00, 0.20, 0.34, 0.18, 0.00],
];

/// Render the icon at `size` × `size` pixels.
pub fn render(size: u32) -> RgbaImage {
    let hi = size * SS;
    let s = hi as f32;
    let mut buf = vec![
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0
        };
        (hi * hi) as usize
    ];

    // --- 1. the rounded-square plate, with a vertical gradient -----------
    // 22% radius is the macOS "squircle" proportion; close enough that the
    // icon does not look alien on any of the three platforms.
    let inset = s * 0.045;
    let plate = s - inset * 2.0;
    let radius = s * 0.22;

    for y in 0..hi {
        let fy = y as f32 + 0.5;
        let t = (fy / s).clamp(0.0, 1.0);
        let grad = GRAD_TOP.lerp(GRAD_BOTTOM, t);
        for x in 0..hi {
            let fx = x as f32 + 0.5;
            let cov = rounded_rect_coverage(fx, fy, inset, inset, plate, plate, radius);
            if cov > 0.0 {
                blend(&mut buf[(y * hi + x) as usize], grad, cov);
            }
        }
    }

    // --- 2. the mosaic field ---------------------------------------------
    // A 5×5 grid inset from the plate, each cell a small rounded square.
    let field = s * 0.60;
    let field_x = (s - field) * 0.5;
    let field_y = (s - field) * 0.5;
    let cell = field / 5.0;
    let gap = cell * 0.11;
    let cell_r = cell * 0.16;

    for (row, alphas) in MOSAIC_ALPHA.iter().enumerate() {
        for (col, alpha) in alphas.iter().enumerate() {
            if *alpha <= 0.0 {
                continue;
            }
            let cx = field_x + col as f32 * cell + gap * 0.5;
            let cy = field_y + row as f32 * cell + gap * 0.5;
            let cw = cell - gap;
            let mut color = MOSAIC;
            // Scaled down from the authored table: at 16-32px the mosaic must
            // read as texture behind the bar, not compete with it.
            color.a = *alpha * 0.82;
            fill_rounded_rect(
                &mut buf,
                hi,
                Rect {
                    x: cx,
                    y: cy,
                    w: cw,
                    h: cw,
                    r: cell_r,
                },
                color,
            );
        }
    }

    // --- 3. the redaction bar --------------------------------------------
    // Dark rather than light: it must read as something *removed* from the
    // image, and dark-on-bright holds contrast better when the icon is scaled
    // down onto a light desktop background.
    let bar_h = s * 0.175;
    let bar_w = s * 0.70;
    let bar_x = (s - bar_w) * 0.5;
    let bar_y = (s - bar_h) * 0.5;
    fill_rounded_rect(
        &mut buf,
        hi,
        Rect {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            r: bar_h * 0.34,
        },
        BAR,
    );

    downsample(&buf, hi, size)
}

/// A rounded rectangle in supersampled pixel coordinates.
#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    r: f32,
}

fn fill_rounded_rect(buf: &mut [Color], hi: u32, rect: Rect, color: Color) {
    let Rect { x, y, w, h, r } = rect;
    // Only touch the pixels the shape can reach, plus a one-pixel margin for
    // the antialiasing ramp.
    let x0 = (x.floor() as i64 - 1).max(0) as u32;
    let y0 = (y.floor() as i64 - 1).max(0) as u32;
    let x1 = ((x + w).ceil() as i64 + 1).min(hi as i64) as u32;
    let y1 = ((y + h).ceil() as i64 + 1).min(hi as i64) as u32;
    for py in y0..y1 {
        for px in x0..x1 {
            let cov = rounded_rect_coverage(px as f32 + 0.5, py as f32 + 0.5, x, y, w, h, r);
            if cov > 0.0 {
                blend(&mut buf[(py * hi + px) as usize], color, cov);
            }
        }
    }
}

/// Box-filter the supersampled buffer down to the target size.
fn downsample(buf: &[Color], hi: u32, size: u32) -> RgbaImage {
    let mut img = RgbaImage::new(size, size);
    let n = (SS * SS) as f32;
    for y in 0..size {
        for x in 0..size {
            let (mut r, mut g, mut b, mut a) = (0.0, 0.0, 0.0, 0.0);
            for sy in 0..SS {
                for sx in 0..SS {
                    let c = buf[((y * SS + sy) * hi + (x * SS + sx)) as usize];
                    // Premultiply before averaging: averaging straight colour
                    // across a transparent edge drags the halo toward black.
                    r += c.r * c.a;
                    g += c.g * c.a;
                    b += c.b * c.a;
                    a += c.a;
                }
            }
            let (r, g, b, a) = (r / n, g / n, b / n, a / n);
            let px = if a > 0.0 {
                [
                    (r / a).round().clamp(0.0, 255.0) as u8,
                    (g / a).round().clamp(0.0, 255.0) as u8,
                    (b / a).round().clamp(0.0, 255.0) as u8,
                    (a * 255.0).round().clamp(0.0, 255.0) as u8,
                ]
            } else {
                [0, 0, 0, 0]
            };
            img.put_pixel(x, y, Rgba(px));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plate_fills_the_centre_and_clears_the_corners() {
        let img = render(64);
        // Centre is the redaction bar: fully opaque.
        assert_eq!(img.get_pixel(32, 32).0[3], 255);
        // The rounded corner is outside the plate, so fully transparent —
        // this is what stops the icon rendering as a hard square.
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn edges_are_antialiased_not_binary() {
        let img = render(256);
        // Somewhere along the left edge of the plate there must be a pixel of
        // partial coverage; a purely binary mask would look ragged when scaled.
        let partial = (0..256).any(|y| {
            (0..30).any(|x| {
                let a = img.get_pixel(x, y).0[3];
                a > 0 && a < 255
            })
        });
        assert!(partial, "no antialiased edge pixels found");
    }

    #[test]
    fn rendering_is_deterministic() {
        // The mosaic is a fixed table, not noise: two runs must be identical,
        // or every rebuild would churn the committed icon files.
        assert_eq!(render(32).into_raw(), render(32).into_raw());
    }
}
