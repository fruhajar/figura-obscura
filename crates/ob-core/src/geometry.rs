//! Frame and detection geometry shared by every crate.
//!
//! `ob-core` holds no image codecs; a [`Frame`] is a plain interleaved RGB8
//! buffer so both the batch pipeline (`ob-media`) and the future screen tool
//! (`ob-screen`) can construct one without depending on any I/O crate.

use crate::taxonomy::Category;
use serde::{Deserialize, Serialize};

/// An axis-aligned bounding box in pixel coordinates (top-left origin).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Intersection-over-union with another box (0.0 when disjoint).
    pub fn iou(&self, other: &BBox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Grow the box by `frac` of its own size on every side, clamped to
    /// `[0, w] × [0, h]`. Used for censor padding (R4).
    pub fn expanded(&self, frac: f32, w: f32, h: f32) -> BBox {
        let dx = self.width() * frac;
        let dy = self.height() * frac;
        BBox {
            x1: (self.x1 - dx).clamp(0.0, w),
            y1: (self.y1 - dy).clamp(0.0, h),
            x2: (self.x2 + dx).clamp(0.0, w),
            y2: (self.y2 + dy).clamp(0.0, h),
        }
    }

    /// The top `frac` strip of this box — used to derive an eye band from a
    /// face box in v1 (see plan: "Eyes" category).
    pub fn top_strip(&self, frac: f32) -> BBox {
        BBox {
            x1: self.x1,
            y1: self.y1,
            x2: self.x2,
            y2: self.y1 + self.height() * frac.clamp(0.0, 1.0),
        }
    }
}

/// A single detection: a box, its canonical category, and model confidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BBox,
    pub category: Category,
    pub score: f32,
}

/// An interleaved RGB8 image: `data.len() == width * height * 3`.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Frame {
    /// Build a frame, validating the buffer length.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self, FrameError> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(FrameError::BadLength {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn width_f(&self) -> f32 {
        self.width as f32
    }

    pub fn height_f(&self) -> f32 {
        self.height as f32
    }

    /// Copy an axis-aligned sub-rectangle out as its own frame.
    ///
    /// The rectangle is clamped to the frame, so a caller may pass a window that
    /// runs off the edge and get back the overlapping part. Returns `None` when
    /// the clamped rectangle is empty. Used by tiled detection (`ob-detect`) to
    /// feed one region at a time to the model at native resolution.
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Option<Frame> {
        let x0 = x.min(self.width);
        let y0 = y.min(self.height);
        let x1 = (x0.saturating_add(w)).min(self.width);
        let y1 = (y0.saturating_add(h)).min(self.height);
        let (cw, ch) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
        if cw == 0 || ch == 0 {
            return None;
        }
        let mut data = Vec::with_capacity(cw as usize * ch as usize * 3);
        for row in y0..y1 {
            let start = (row as usize * self.width as usize + x0 as usize) * 3;
            let end = start + cw as usize * 3;
            data.extend_from_slice(&self.data[start..end]);
        }
        Frame::new(cw, ch, data).ok()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("frame buffer length {got} does not match width*height*3 = {expected}")]
    BadLength { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(20.0, 20.0, 30.0, 30.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        // inter = 50, union = 150 -> 1/3
        assert!((a.iou(&b) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn expanded_clamps_to_bounds() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        let e = b.expanded(1.0, 12.0, 12.0);
        assert_eq!(e.x1, 0.0);
        assert_eq!(e.x2, 12.0);
    }

    #[test]
    fn crop_extracts_the_right_pixels() {
        // 3x2 frame where each pixel's red channel is its linear index.
        let mut data = vec![0u8; 3 * 2 * 3];
        for i in 0..6 {
            data[i * 3] = i as u8;
        }
        let f = Frame::new(3, 2, data).unwrap();
        let c = f.crop(1, 0, 2, 2).unwrap();
        assert_eq!((c.width, c.height), (2, 2));
        // Row 0: indices 1,2 — row 1: indices 4,5.
        assert_eq!([c.data[0], c.data[3], c.data[6], c.data[9]], [1u8, 2, 4, 5]);
    }

    #[test]
    fn crop_clamps_to_bounds_and_rejects_empty() {
        let f = Frame::new(4, 4, vec![7u8; 4 * 4 * 3]).unwrap();
        // Window running off the right/bottom edge comes back clamped.
        let c = f.crop(3, 3, 10, 10).unwrap();
        assert_eq!((c.width, c.height), (1, 1));
        // Fully outside the frame is empty, not a panic.
        assert!(f.crop(4, 0, 2, 2).is_none());
        assert!(f.crop(0, 0, 0, 2).is_none());
    }

    #[test]
    fn frame_length_is_validated() {
        assert!(Frame::new(2, 2, vec![0u8; 12]).is_ok());
        assert!(Frame::new(2, 2, vec![0u8; 10]).is_err());
    }
}
