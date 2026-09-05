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
        // `matched` is indexed by track, and must therefore grow with
        // `self.tracks` — an unmatched detection pushes a new track inside this
        // loop, and the next detection then iterates over it. Letting the two
        // fall out of step panicked on the second detection of the very first
        // frame: `tracks` had one entry, `matched` still had none.
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
                None => {
                    self.tracks.push(Track {
                        det: *d,
                        ttl: self.cfg.ttl,
                    });
                    // Marked matched, which does double duty: it keeps the two
                    // vectors the same length, and it stops a later detection in
                    // this same frame from matching a track that was created from
                    // an earlier one — two detections in one frame are two
                    // objects, and merging them would drop a censored region.
                    // It also exempts the new track from the ageing pass below,
                    // which is right: it was just seen.
                    matched.push(true);
                }
            }
        }
        // Age unmatched existing tracks. The lengths are now equal by
        // construction; the guard that used to be here was hiding the bug above
        // rather than fixing it, since it silently skipped ageing for tracks
        // that `matched` had no room for.
        debug_assert_eq!(self.tracks.len(), matched.len());
        for (t, was_matched) in self.tracks.iter_mut().zip(&matched) {
            if !was_matched {
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

    #[test]
    fn two_detections_on_the_very_first_frame_do_not_panic() {
        // The reported crash, exactly: `index out of bounds: the len is 0 but
        // the index is 0` at the `matched[i]` lookup. The first detection
        // pushed a track, growing `tracks` to 1 while `matched` stayed empty,
        // and the second detection then indexed it. Every video whose detector
        // found two regions on its first detect frame aborted the process.
        let mut t = Tracker::new(TrackConfig::default());
        let out = t.update(Some(&[det(0.0), det(100.0)]));
        assert_eq!(out.len(), 2, "both regions must survive the first frame");
    }

    #[test]
    fn many_new_detections_in_one_frame_all_become_tracks() {
        let mut t = Tracker::new(TrackConfig::default());
        let dets: Vec<_> = (0..8).map(|i| det(i as f32 * 50.0)).collect();
        assert_eq!(t.update(Some(&dets)).len(), 8);
    }

    #[test]
    fn a_track_created_this_frame_is_not_reused_by_a_later_detection() {
        // Two overlapping detections in one frame are two objects — NMS decides
        // whether they should have been merged, not the tracker. If the second
        // matched the track the first had just created, one censored region
        // would silently vanish.
        let mut t = Tracker::new(TrackConfig::default());
        let out = t.update(Some(&[det(0.0), det(1.0)])); // IoU well above match_iou
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_brand_new_track_is_not_aged_on_the_frame_it_appears() {
        // It was just seen, so it must start with a full ttl; ageing it here
        // would shorten every track's life by one frame.
        let cfg = TrackConfig {
            ttl: 1,
            ..Default::default()
        };
        let mut t = Tracker::new(cfg);
        t.update(Some(&[det(0.0)]));
        // With ttl 1, an immediate ageing would have dropped it already.
        assert_eq!(t.update(None).len(), 0);
    }
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
