//! Blur and preset-image overlay renderers, backed by the `image` crate.

use crate::{CensorError, Rect};
use image::{imageops, RgbImage};
use ob_core::censor::OverlayFit;
use ob_core::geometry::Frame;

/// Copy the frame rectangle into an owned `RgbImage`.
fn crop(frame: &Frame, r: &Rect) -> RgbImage {
    let w = frame.width as usize;
    let cw = (r.x1 - r.x0) as u32;
    let ch = (r.y1 - r.y0) as u32;
    let mut img = RgbImage::new(cw, ch);
    for (dy, y) in (r.y0..r.y1).enumerate() {
        for (dx, x) in (r.x0..r.x1).enumerate() {
            let p = (y * w + x) * 3;
            img.put_pixel(
                dx as u32,
                dy as u32,
                image::Rgb([frame.data[p], frame.data[p + 1], frame.data[p + 2]]),
            );
        }
    }
    img
}

/// Write an `RgbImage` back into the frame rectangle (sizes must match).
fn paste(frame: &mut Frame, r: &Rect, img: &RgbImage) {
    let w = frame.width as usize;
    for (dy, y) in (r.y0..r.y1).enumerate() {
        for (dx, x) in (r.x0..r.x1).enumerate() {
            let px = img.get_pixel(dx as u32, dy as u32);
            let p = (y * w + x) * 3;
            frame.data[p] = px[0];
            frame.data[p + 1] = px[1];
            frame.data[p + 2] = px[2];
        }
    }
}

/// Gaussian-blur the rectangle in place.
pub fn blur(frame: &mut Frame, r: &Rect, sigma: f32) {
    let region = crop(frame, r);
    let blurred = imageops::blur(&region, sigma.max(0.1));
    paste(frame, r, &blurred);
}

/// Composite a preset image over the rectangle.
pub fn image_overlay(
    frame: &mut Frame,
    r: &Rect,
    path: &str,
    fit: OverlayFit,
    opacity: f32,
) -> Result<(), CensorError> {
    let src = image::open(path)
        .map_err(|e| CensorError::Overlay(path.to_string(), e.to_string()))?
        .to_rgb8();
    let cw = (r.x1 - r.x0) as u32;
    let ch = (r.y1 - r.y0) as u32;

    let scaled = fit_overlay(&src, cw, ch, fit);
    let mut base = crop(frame, r);
    let a = opacity.clamp(0.0, 1.0);

    for y in 0..ch {
        for x in 0..cw {
            // `scaled` is guaranteed to cover cw×ch for every fit mode below.
            let o = scaled.get_pixel(x.min(scaled.width() - 1), y.min(scaled.height() - 1));
            let b = base.get_pixel_mut(x, y);
            for c in 0..3 {
                b[c] = ((o[c] as f32) * a + (b[c] as f32) * (1.0 - a)).round() as u8;
            }
        }
    }
    paste(frame, r, &base);
    Ok(())
}

/// Scale/lay out the overlay to exactly `cw×ch` per the requested fit.
fn fit_overlay(src: &RgbImage, cw: u32, ch: u32, fit: OverlayFit) -> RgbImage {
    match fit {
        OverlayFit::Stretch => imageops::resize(src, cw, ch, imageops::FilterType::Triangle),
        OverlayFit::Cover | OverlayFit::Contain => {
            let (sw, sh) = (src.width() as f32, src.height() as f32);
            let scale = if matches!(fit, OverlayFit::Cover) {
                (cw as f32 / sw).max(ch as f32 / sh)
            } else {
                (cw as f32 / sw).min(ch as f32 / sh)
            };
            let rw = (sw * scale).round().max(1.0) as u32;
            let rh = (sh * scale).round().max(1.0) as u32;
            let resized = imageops::resize(src, rw, rh, imageops::FilterType::Triangle);
            // Center-crop / center-place into a cw×ch canvas.
            let mut canvas = RgbImage::new(cw, ch);
            let ox = (rw as i64 - cw as i64) / 2;
            let oy = (rh as i64 - ch as i64) / 2;
            for y in 0..ch {
                for x in 0..cw {
                    let sx = x as i64 + ox;
                    let sy = y as i64 + oy;
                    if sx >= 0 && sy >= 0 && (sx as u32) < rw && (sy as u32) < rh {
                        canvas.put_pixel(x, y, *resized.get_pixel(sx as u32, sy as u32));
                    }
                }
            }
            canvas
        }
        OverlayFit::Tile => {
            let mut canvas = RgbImage::new(cw, ch);
            for y in 0..ch {
                for x in 0..cw {
                    canvas.put_pixel(x, y, *src.get_pixel(x % src.width(), y % src.height()));
                }
            }
            canvas
        }
    }
}
