//! Tiled (multi-pass) detection.
//!
//! A single whole-frame pass letterboxes the entire image into the model's
//! square input, so on a large source everything shrinks: a 4K frame reaching a
//! 640px input is scaled by ~0.167, turning a 30px feature into 5px — at or
//! below YOLOv8's smallest stride (8px). Anti-aliased resampling
//! ([`crate::preprocess::Resampler`]) stops such features from disappearing
//! outright, but it cannot give the model back the resolution it needs.
//!
//! [`TiledDetector`] wraps any [`Detector`] and runs it several times per frame:
//! once over the whole frame (which keeps large regions and global context), then
//! once per tile of an overlapping grid, each tile cropped at native resolution
//! so small regions arrive at a workable size. All passes are merged and put
//! through one global NMS.
//!
//! It is a [`Detector`] itself, so the batch pipeline, the GUI preview and the
//! future screen tool get tiling by wrapping, with no changes to their code.

use crate::postprocess::nms;
use crate::{DetectError, Detector};
use ob_core::geometry::{Detection, Frame};
use ob_core::taxonomy::Category;

/// When to spend the extra inference passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TilingMode {
    /// Never tile — one whole-frame pass, the pre-tiling behaviour.
    Off,
    /// Tile only when the whole-frame pass would downscale past
    /// [`TilingConfig::min_scale`]. The default.
    #[default]
    Auto,
    /// Always tile, however small the frame.
    Always,
}

impl TilingMode {
    pub fn name(&self) -> &'static str {
        match self {
            TilingMode::Off => "off",
            TilingMode::Auto => "auto",
            TilingMode::Always => "always",
        }
    }

    pub fn parse(s: &str) -> Option<TilingMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" => Some(TilingMode::Off),
            "auto" => Some(TilingMode::Auto),
            "always" | "on" | "true" => Some(TilingMode::Always),
            _ => None,
        }
    }
}

/// How the tile grid is built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TilingConfig {
    pub mode: TilingMode,
    /// Fraction of a tile shared with its neighbour. Overlap is what stops a
    /// region that straddles a seam from being seen only in fragments.
    pub overlap: f32,
    /// In [`TilingMode::Auto`], tile when the whole-frame letterbox scale is
    /// below this. 0.5 means "the frame is more than 2x the model input".
    pub min_scale: f32,
    /// Hard cap on tiles per frame. Inference cost is linear in tile count, so
    /// this bounds the worst case; the grid coarsens (accepting some downscale
    /// per tile) rather than exceeding it.
    pub max_tiles: usize,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            mode: TilingMode::Auto,
            overlap: 0.25,
            min_scale: 0.5,
            max_tiles: 12,
        }
    }
}

/// One tile's source-pixel window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Plan the tile grid for a `frame_w × frame_h` frame and a `input_size` model.
///
/// Tiles start at the model's native input size (scale 1.0 — no downscale at
/// all) and grow until the grid fits within `max_tiles`. Returns an empty vec
/// when a single tile would already cover the frame, i.e. when tiling would just
/// repeat the whole-frame pass.
pub fn plan_tiles(frame_w: u32, frame_h: u32, input_size: u32, cfg: &TilingConfig) -> Vec<Tile> {
    if frame_w == 0 || frame_h == 0 || input_size == 0 {
        return Vec::new();
    }
    let overlap = cfg.overlap.clamp(0.0, 0.9);
    let max_tiles = cfg.max_tiles.max(1);

    let mut tile = input_size as f32;
    loop {
        let tw = tile.min(frame_w as f32).max(1.0);
        let th = tile.min(frame_h as f32).max(1.0);
        // One tile already covers everything: the whole-frame pass is enough.
        if tw >= frame_w as f32 && th >= frame_h as f32 {
            return Vec::new();
        }
        let step_x = (tw * (1.0 - overlap)).max(1.0);
        let step_y = (th * (1.0 - overlap)).max(1.0);
        let cols = ((frame_w as f32 - tw) / step_x).ceil().max(0.0) as usize + 1;
        let rows = ((frame_h as f32 - th) / step_y).ceil().max(0.0) as usize + 1;

        if cols * rows <= max_tiles {
            let mut tiles = Vec::with_capacity(cols * rows);
            for r in 0..rows {
                for c in 0..cols {
                    // Place the last row/column flush with the far edge so the
                    // border is covered without a runt tile.
                    let x = ((c as f32 * step_x) as u32).min(frame_w.saturating_sub(tw as u32));
                    let y = ((r as f32 * step_y) as u32).min(frame_h.saturating_sub(th as u32));
                    tiles.push(Tile {
                        x,
                        y,
                        w: tw as u32,
                        h: th as u32,
                    });
                }
            }
            tiles.dedup();
            return tiles;
        }
        // Too many tiles: coarsen. Each step trades some per-tile resolution for
        // a quarter fewer tiles along each axis.
        tile *= 1.25;
    }
}

/// The grid a frame of this size will actually be tiled with, mode included.
///
/// [`plan_tiles`] answers "what grid would cover this frame"; this answers
/// "what grid will actually be run", which is the one that also consults
/// [`TilingMode`] and `min_scale`.
///
/// It takes dimensions rather than a [`Frame`] so that anything holding only a
/// size can ask — notably `ob_job::estimate`, which costs a batch from image
/// headers without decoding a pixel. Cost estimates and the detector therefore
/// share one implementation and cannot drift apart.
pub fn tiles_for_size(w: u32, h: u32, input_size: u32, cfg: &TilingConfig) -> Vec<Tile> {
    match cfg.mode {
        TilingMode::Off => return Vec::new(),
        TilingMode::Auto => {
            let s = (input_size as f32 / w as f32).min(input_size as f32 / h as f32);
            if s >= cfg.min_scale {
                return Vec::new();
            }
        }
        TilingMode::Always => {}
    }
    plan_tiles(w, h, input_size, cfg)
}

/// A [`Detector`] that runs its inner detector over the whole frame plus an
/// overlapping tile grid, merging every pass through one NMS.
pub struct TiledDetector<D: Detector> {
    inner: D,
    input_size: u32,
    nms_iou: f32,
    cfg: TilingConfig,
}

impl<D: Detector> TiledDetector<D> {
    /// Wrap `inner`. `input_size` is the model's square input side and
    /// `nms_iou` the IoU used to merge overlapping passes — pass the same
    /// values the inner detector was built with.
    pub fn new(inner: D, input_size: u32, nms_iou: f32, cfg: TilingConfig) -> Self {
        Self {
            inner,
            input_size,
            nms_iou,
            cfg,
        }
    }

    /// Whether this frame will actually be tiled, and with what grid.
    pub fn tiles_for(&self, frame: &Frame) -> Vec<Tile> {
        tiles_for_size(frame.width, frame.height, self.input_size, &self.cfg)
    }
}

impl<D: Detector> Detector for TiledDetector<D> {
    /// Tiling changes *where* the inner model looks, never *what* it can find.
    fn can_emit(&self, category: Category) -> bool {
        self.inner.can_emit(category)
    }

    fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, DetectError> {
        // Pass 1: the whole frame. Always run — it is the only pass that sees
        // large regions in full and the only one that runs when tiling is off.
        let mut all = self.inner.detect(frame)?;

        for t in self.tiles_for(frame) {
            let Some(sub) = frame.crop(t.x, t.y, t.w, t.h) else {
                continue;
            };
            let dets = self.inner.detect(&sub)?;
            // Tile-local coordinates back to frame coordinates.
            all.extend(dets.into_iter().map(|mut d| {
                d.bbox.x1 += t.x as f32;
                d.bbox.x2 += t.x as f32;
                d.bbox.y1 += t.y as f32;
                d.bbox.y2 += t.y as f32;
                d
            }));
        }

        // One global NMS over every pass. Note this keeps a box that a tile saw
        // only partially (cut at a seam) alongside the full box from the
        // overlapping neighbour when their IoU is low — for censoring, covering
        // a region twice is harmless and covering it not at all is not.
        Ok(nms(all, self.nms_iou))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::geometry::BBox;
    use ob_core::taxonomy::cat;
    use std::sync::Mutex;

    /// Records every frame size it is asked about and returns one fixed box.
    struct MockDetector {
        seen: Mutex<Vec<(u32, u32)>>,
        emit: Option<BBox>,
    }

    impl MockDetector {
        fn new(emit: Option<BBox>) -> Self {
            Self {
                seen: Mutex::new(Vec::new()),
                emit,
            }
        }
    }

    impl Detector for MockDetector {
        fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, DetectError> {
            self.seen.lock().unwrap().push((frame.width, frame.height));
            Ok(self
                .emit
                .map(|bbox| {
                    vec![Detection {
                        bbox,
                        category: cat::FEMALE_BREAST_EXPOSED,
                        score: 0.9,
                    }]
                })
                .unwrap_or_default())
        }
    }

    fn frame(w: u32, h: u32) -> Frame {
        Frame::new(w, h, vec![0u8; (w * h * 3) as usize]).unwrap()
    }

    #[test]
    fn small_frame_is_not_tiled() {
        // 640px model, 800x600 frame -> scale 0.8, above min_scale 0.5.
        let tiles = TiledDetector::new(MockDetector::new(None), 640, 0.45, TilingConfig::default())
            .tiles_for(&frame(800, 600));
        assert!(tiles.is_empty());
    }

    #[test]
    fn large_frame_is_tiled_within_the_cap() {
        let cfg = TilingConfig::default();
        let d = TiledDetector::new(MockDetector::new(None), 640, 0.45, cfg);
        let tiles = d.tiles_for(&frame(3840, 2160));
        assert!(!tiles.is_empty(), "4K frame should tile");
        assert!(
            tiles.len() <= cfg.max_tiles,
            "grid of {} exceeded the cap of {}",
            tiles.len(),
            cfg.max_tiles
        );
    }

    #[test]
    fn tiles_cover_every_pixel_of_the_frame() {
        // Coverage is the whole point: a gap in the grid is a region that only
        // the downscaled whole-frame pass ever sees.
        let (w, h) = (3000u32, 1700u32);
        let tiles = plan_tiles(w, h, 640, &TilingConfig::default());
        assert!(!tiles.is_empty());
        // Sample a lattice of points inside the frame; every one must fall in
        // some tile. Checked directly rather than through a coverage bitmap, so
        // the assertion cannot drift from the frame's real bounds.
        let mut y = 0;
        while y < h {
            let mut x = 0;
            while x < w {
                assert!(
                    tiles
                        .iter()
                        .any(|t| x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h),
                    "tile grid left a gap at ({x}, {y}); tiles: {tiles:?}"
                );
                x += 7;
            }
            y += 7;
        }
    }

    #[test]
    fn tiles_overlap_their_neighbours() {
        let tiles = plan_tiles(2000, 700, 640, &TilingConfig::default());
        assert!(tiles.len() >= 2);
        // Consecutive tiles in a row must share a strip.
        let a = tiles[0];
        let b = tiles[1];
        assert!(b.x < a.x + a.w, "tiles {a:?} and {b:?} do not overlap");
    }

    #[test]
    fn tiling_off_runs_exactly_one_pass() {
        let cfg = TilingConfig {
            mode: TilingMode::Off,
            ..Default::default()
        };
        let d = TiledDetector::new(MockDetector::new(None), 640, 0.45, cfg);
        d.detect(&frame(3840, 2160)).unwrap();
        assert_eq!(d.inner.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn tiled_run_makes_one_pass_per_tile_plus_the_whole_frame() {
        let d = TiledDetector::new(MockDetector::new(None), 640, 0.45, TilingConfig::default());
        let f = frame(3840, 2160);
        let expected = d.tiles_for(&f).len() + 1;
        d.detect(&f).unwrap();
        let seen = d.inner.seen.lock().unwrap();
        assert_eq!(seen.len(), expected);
        // The first pass is the whole frame; the rest are tile-sized crops.
        assert_eq!(seen[0], (3840, 2160));
        assert!(seen[1..].iter().all(|(w, h)| *w < 3840 && *h < 2160));
    }

    #[test]
    fn tile_detections_are_offset_into_frame_coordinates() {
        // The mock reports the same tile-local box for every pass, so any box
        // landing outside the first tile proves the offset was applied.
        let d = TiledDetector::new(
            MockDetector::new(Some(BBox::new(10.0, 10.0, 30.0, 30.0))),
            640,
            0.45,
            TilingConfig::default(),
        );
        let dets = d.detect(&frame(3840, 2160)).unwrap();
        assert!(dets.len() > 1, "expected boxes from several tiles");
        assert!(
            dets.iter().any(|x| x.bbox.x1 > 100.0 || x.bbox.y1 > 100.0),
            "no detection was offset out of the first tile"
        );
    }

    #[test]
    fn mode_parses_its_own_names() {
        for m in [TilingMode::Off, TilingMode::Auto, TilingMode::Always] {
            assert_eq!(TilingMode::parse(m.name()), Some(m));
        }
        assert_eq!(TilingMode::parse("nope"), None);
    }
}
