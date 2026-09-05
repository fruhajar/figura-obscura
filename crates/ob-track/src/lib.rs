//! # ob-track
//!
//! Temporal smoothing so censoring does not flicker across video frames or in
//! the real-time screen tool (R10). A [`Tracker`] matches new detections to
//! existing tracks by IoU and applies **hysteresis**: a region keeps being
//! censored for a few frames after the detector stops seeing it (`ttl`), and a
//! track must be seen once to appear. This also lets the pipeline detect only
//! every Nth frame and reuse tracks in between.
//!
//! Pure and deterministic — no I/O, no timing — so it is unit-testable.

use ob_core::geometry::{BBox, Detection};

/// Tuning for the tracker.
#[derive(Debug, Clone, Copy)]
pub struct TrackConfig {
    /// IoU above which a new detection is considered the same object.
    pub match_iou: f32,
    /// Frames a track survives without a fresh detection before it is dropped.
    pub ttl: u32,
    /// Exponential smoothing factor for box position (0 = frozen, 1 = snap).
    pub smoothing: f32,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            match_iou: 0.3,
            ttl: 8,
            smoothing: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
struct Track {
    det: Detection,
    /// Frames remaining before this track expires if unseen.
    ttl: u32,
}

/// Stateful multi-object tracker. Feed it each frame's raw detections (or `None`
/// to coast on a skipped frame) and it returns the smoothed set to censor.
#[derive(Debug, Default)]
pub struct Tracker {
    cfg: TrackConfig,
    tracks: Vec<Track>,
}

impl Tracker {
    pub fn new(cfg: TrackConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
        }
    }

    /// Advance one frame.
    ///
    /// `detections = Some(..)` on a frame the detector ran (matches + updates
    /// tracks); `None` on a skipped frame (all tracks age but persist). Returns
    /// the current smoothed detections to censor.
    pub fn update(&mut self, detections: Option<&[Detection]>) -> Vec<Detection> {
        match detections {
            None => self.coast(),
            Some(dets) => self.integrate(dets),
        }
        self.tracks.iter().map(|t| t.det).collect()
    }

    fn coast(&mut self) {
        for t in &mut self.tracks {
            t.ttl = t.ttl.saturating_sub(1);
        }
        self.tracks.retain(|t| t.ttl > 0);
    }

    fn integrate(&mut self, dets: &[Detection]) {
        let mut matched = vec![false; self.tracks.len()];
        for d in dets {
            // Find the best unmatched track of the same category.
            let mut best: Option<(usize, f32)> = None;
            for (i, t) in self.tracks.iter().enumerate() {
                if matched[i] || t.det.category != d.category {
                    continue;
                }
                let iou = t.det.bbox.iou(&d.bbox);
                if iou >= self.cfg.match_iou && best.map_or(true, |(_, b)| iou > b) {
                    best = Some((i, iou));
                }
            }
            match best {
                Some((i, _)) => {
                    matched[i] = true;
                    let t = &mut self.tracks[i];
                    t.det.bbox = smooth(&t.det.bbox, &d.bbox, self.cfg.smoothing);
                    t.det.score = d.score;
                    t.ttl = self.cfg.ttl;
                }
                None => self.tracks.push(Track {
                    det: *d,
                    ttl: self.cfg.ttl,
                }),
            }
        }
        // Age unmatched existing tracks.
        for (i, t) in self.tracks.iter_mut().enumerate() {
            if i < matched.len() && !matched[i] {
                t.ttl = t.ttl.saturating_sub(1);
            }
        }
        self.tracks.retain(|t| t.ttl > 0);
    }
}

/// Exponentially move `from` toward `to` by `alpha`.
fn smooth(from: &BBox, to: &BBox, alpha: f32) -> BBox {
    let a = alpha.clamp(0.0, 1.0);
    let lerp = |x: f32, y: f32| x + (y - x) * a;
    BBox {
        x1: lerp(from.x1, to.x1),
        y1: lerp(from.y1, to.y1),
        x2: lerp(from.x2, to.x2),
        y2: lerp(from.y2, to.y2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::taxonomy::cat;

    fn det(x1: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, 0.0, x1 + 10.0, 10.0),
            category: cat::FEMALE_BREAST_EXPOSED,
            score: 0.9,
        }
    }

    #[test]
    fn track_persists_after_detection_drops() {
        let mut t = Tracker::new(TrackConfig {
            ttl: 3,
            ..Default::default()
        });
        assert_eq!(t.update(Some(&[det(0.0)])).len(), 1);
        // Detector loses it, but hysteresis keeps censoring for ttl frames.
        assert_eq!(t.update(Some(&[])).len(), 1);
        assert_eq!(t.update(Some(&[])).len(), 1);
        assert_eq!(t.update(Some(&[])).len(), 0); // expired
    }

    #[test]
    fn coast_frame_keeps_tracks() {
        let mut t = Tracker::new(TrackConfig::default());
        t.update(Some(&[det(0.0)]));
        // Skipped (non-detected) frames coast on existing tracks.
        assert_eq!(t.update(None).len(), 1);
    }

    #[test]
    fn overlapping_detection_updates_not_duplicates() {
        let mut t = Tracker::new(TrackConfig::default());
        t.update(Some(&[det(0.0)]));
        let out = t.update(Some(&[det(2.0)])); // heavy overlap -> same track
        assert_eq!(out.len(), 1);
    }
}
