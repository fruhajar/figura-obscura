//! A software renderer for the app's own UI, used only by tests.
//!
//! The build container has no display server and no GPU, so the interface could
//! otherwise only be checked for *panics*, never for how it actually looks. egui
//! is a two-stage renderer, and only the second stage needs hardware: `Context`
//! produces `ClippedPrimitive`s — textured, vertex-coloured triangles — and the
//! usual glow/wgpu backends just draw them. Drawing them on the CPU instead is
//! a few hundred lines and yields real PNGs of the real UI.
//!
//! This is a faithful rasterisation of what ships, not a mock: same widgets,
//! same theme, same layout pass, same font atlas. What it does not reproduce is
//! anything the GPU backend does differently — it renders in gamma space, like
//! egui's non-linear-framebuffer path, so colours may differ marginally from a
//! machine configured for a linear framebuffer.
//!
//! Test-only (`#[cfg(test)]` at the module site), so none of it ships.

use egui::epaint::{ClippedPrimitive, Primitive, Vertex};
use egui::{Color32, ImageData, Pos2, Rect, TextureId};
use std::collections::HashMap;

/// An RGBA8 image with premultiplied alpha, matching egui's `Color32`.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u8; 4]>,
}

impl Canvas {
    fn new(width: usize, height: usize, clear: Color32) -> Self {
        Self {
            width,
            height,
            pixels: vec![clear.to_array(); width * height],
        }
    }

    fn blend(&mut self, x: usize, y: usize, src: [u8; 4]) {
        // egui colours are premultiplied, so "over" is src + dst*(1-a).
        let a = src[3] as u32;
        if a == 0 {
            return;
        }
        let dst = &mut self.pixels[y * self.width + x];
        if a == 255 {
            *dst = src;
            return;
        }
        let inv = 255 - a;
        for c in 0..4 {
            dst[c] = (src[c] as u32 + (dst[c] as u32 * inv) / 255).min(255) as u8;
        }
    }

    /// Write the canvas out as a PNG.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut flat = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            // Un-premultiply for storage so the PNG looks right in a viewer.
            let a = p[3];
            if a == 0 || a == 255 {
                flat.extend_from_slice(p);
            } else {
                for c in 0..3 {
                    flat.push(((p[c] as u32 * 255) / a as u32).min(255) as u8);
                }
                flat.push(a);
            }
        }
        image::RgbaImage::from_raw(self.width as u32, self.height as u32, flat)
            .ok_or_else(|| "canvas size mismatch".to_string())?
            .save(path)
            .map_err(|e| e.to_string())
    }

    /// Fraction of pixels that differ from the clear colour — a cheap check
    /// that something was actually drawn.
    pub fn painted_fraction(&self, clear: Color32) -> f32 {
        let c = clear.to_array();
        let n = self.pixels.iter().filter(|p| **p != c).count();
        n as f32 / self.pixels.len() as f32
    }
}

/// A decoded texture the meshes can sample.
struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<Color32>,
}

impl Texture {
    /// Bilinear sample at normalised uv, clamped at the edges.
    ///
    /// Bilinear rather than nearest because the font atlas is sampled at very
    /// nearly 1:1 and any sampling error shows up directly as ragged text,
    /// which is exactly what these screenshots exist to judge.
    fn sample(&self, u: f32, v: f32) -> Color32 {
        if self.width == 0 || self.height == 0 {
            return Color32::WHITE;
        }
        let x = (u * self.width as f32 - 0.5).clamp(0.0, self.width as f32 - 1.0);
        let y = (v * self.height as f32 - 0.5).clamp(0.0, self.height as f32 - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);

        let at = |px: usize, py: usize| self.pixels[py * self.width + px].to_array();
        let (p00, p10, p01, p11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let mut out = [0u8; 4];
        for c in 0..4 {
            let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
            let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
            out[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
        }
        Color32::from_rgba_premultiplied(out[0], out[1], out[2], out[3])
    }
}

/// Accumulates the texture set from egui's per-frame deltas.
#[derive(Default)]
pub struct Textures(HashMap<TextureId, Texture>);

impl Textures {
    fn apply(&mut self, delta: &egui::TexturesDelta) {
        for (id, image_delta) in &delta.set {
            let (w, h, pixels) = match &image_delta.image {
                // The font atlas is coverage-only; `srgba_pixels` turns it into
                // the premultiplied white-on-alpha the shader expects.
                ImageData::Font(font) => (
                    font.size[0],
                    font.size[1],
                    font.srgba_pixels(None).collect::<Vec<_>>(),
                ),
                ImageData::Color(color) => (color.size[0], color.size[1], color.pixels.clone()),
            };

            match image_delta.pos {
                // A patch into an existing atlas — egui grows the font atlas
                // this way as new glyphs are rasterised, so ignoring patches
                // would silently lose characters.
                Some([px, py]) => {
                    if let Some(tex) = self.0.get_mut(id) {
                        for row in 0..h {
                            for col in 0..w {
                                let (tx, ty) = (px + col, py + row);
                                if tx < tex.width && ty < tex.height {
                                    tex.pixels[ty * tex.width + tx] = pixels[row * w + col];
                                }
                            }
                        }
                    }
                }
                None => {
                    self.0.insert(
                        *id,
                        Texture {
                            width: w,
                            height: h,
                            pixels,
                        },
                    );
                }
            }
        }
        for id in &delta.free {
            self.0.remove(id);
        }
    }

    fn get(&self, id: TextureId) -> Option<&Texture> {
        self.0.get(&id)
    }
}

/// Rasterise one frame's primitives onto `canvas`.
fn paint(canvas: &mut Canvas, primitives: &[ClippedPrimitive], textures: &Textures) {
    for ClippedPrimitive {
        clip_rect,
        primitive,
    } in primitives
    {
        let Primitive::Mesh(mesh) = primitive else {
            // Paint callbacks are GPU-backend escape hatches; the app uses none.
            continue;
        };
        let Some(texture) = textures.get(mesh.texture_id) else {
            continue;
        };
        for tri in mesh.indices.chunks_exact(3) {
            let v = [
                &mesh.vertices[tri[0] as usize],
                &mesh.vertices[tri[1] as usize],
                &mesh.vertices[tri[2] as usize],
            ];
            triangle(canvas, v, texture, *clip_rect);
        }
    }
}

/// Fill one triangle with barycentric-interpolated colour and uv.
fn triangle(canvas: &mut Canvas, v: [&Vertex; 3], texture: &Texture, clip: Rect) {
    let (p0, p1, p2) = (v[0].pos, v[1].pos, v[2].pos);

    // Signed area; a degenerate (zero-area) triangle covers nothing. egui emits
    // plenty of these along antialiased edges.
    let area = edge(p0, p1, p2);
    if area.abs() < 1e-6 {
        return;
    }

    let min_x = p0.x.min(p1.x).min(p2.x).max(clip.min.x).max(0.0).floor() as i64;
    let max_x = (p0
        .x
        .max(p1.x)
        .max(p2.x)
        .min(clip.max.x)
        .min(canvas.width as f32))
    .ceil() as i64;
    let min_y = p0.y.min(p1.y).min(p2.y).max(clip.min.y).max(0.0).floor() as i64;
    let max_y = (p0
        .y
        .max(p1.y)
        .max(p2.y)
        .min(clip.max.y)
        .min(canvas.height as f32))
    .ceil() as i64;
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    for y in min_y..max_y {
        for x in min_x..max_x {
            let p = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
            // Barycentric coordinates, normalised by the signed area so the
            // winding order does not matter — egui emits both.
            let w0 = edge(p1, p2, p) / area;
            let w1 = edge(p2, p0, p) / area;
            let w2 = edge(p0, p1, p) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
            let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
            let tex = texture.sample(u, vv).to_array();

            let mut out = [0u8; 4];
            for c in 0..4 {
                let vc = w0 * v[0].color.to_array()[c] as f32
                    + w1 * v[1].color.to_array()[c] as f32
                    + w2 * v[2].color.to_array()[c] as f32;
                // egui's shader multiplies vertex colour by texel, in gamma
                // space, both premultiplied.
                out[c] = ((vc * tex[c] as f32) / 255.0).round().clamp(0.0, 255.0) as u8;
            }
            canvas.blend(x as usize, y as usize, out);
        }
    }
}

/// Twice the signed area of the triangle (a, b, c) — positive for one winding.
fn edge(a: Pos2, b: Pos2, c: Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Render `run_ui` at the given size and return the finished canvas.
///
/// Runs several frames before capturing. egui is an immediate-mode library that
/// settles over a frame or two — sizes that depend on last frame's content
/// (`available_width` inside a just-created panel, a `ScrollArea`'s extent) are
/// wrong on frame one, and capturing that would show layout artefacts the user
/// never sees.
pub fn render(
    width: usize,
    height: usize,
    clear: Color32,
    frames: usize,
    mut run_ui: impl FnMut(&egui::Context),
) -> Canvas {
    let ctx = egui::Context::default();
    crate::theme::install(&ctx);

    let mut textures = Textures::default();
    let mut canvas = Canvas::new(width, height, clear);

    for frame in 0..frames.max(1) {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        let output = ctx.run(input, &mut run_ui);
        textures.apply(&output.textures_delta);

        // Only the last frame is painted; the earlier ones exist to settle
        // layout, and painting them would just be overdraw.
        if frame + 1 == frames.max(1) {
            let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
            paint(&mut canvas, &primitives, &textures);
        }
    }
    canvas
}
