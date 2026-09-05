//! Estimating how long a batch will take, before it is started.
//!
//! The naive estimate — elapsed / files done × files remaining — has two
//! problems. It says nothing at all until the first file finishes, which on a
//! batch of 4K video is the point at which the user most wants to know; and it
//! treats every item as equal, so a run of thumbnails followed by a feature
//! film reports a steadily shrinking ETA that is wrong by an order of
//! magnitude.
//!
//! So the batch is measured first. Each item is probed for its dimensions —
//! cheap, a header read for images and one `ffprobe` for videos — and costed in
//! **inference passes**, the thing that actually dominates a run.
//!
//! Probing and costing are **separate**, for the same reason the preview splits
//! detection from composition: reading headers is I/O over every file in the
//! batch, while the thing that changes most often is a tuning slider. So
//! [`probe`] walks the files once and [`cost`] re-answers "how long?" from
//! those measurements instantly, as often as the settings move.
//!
//! Two properties make this worth doing:
//!
//! - The pass count comes from [`ob_detect::tile::plan_tiles`], the very
//!   planner the detector uses. A 4K frame that tiles into 12 passes is costed
//!   as 12, not as "one big image", and the estimate cannot drift from the
//!   detector's behaviour because there is only one implementation.
//! - The scan produces *relative* weights. The absolute seconds-per-pass is
//!   calibrated from the run itself (see [`Calibration`]), so an error in the
//!   constants below stretches every item equally and cancels out of the
//!   ratio. What matters is that a 4K video outweighs a thumbnail by roughly
//!   the right factor, not that any figure here is right in absolute terms.

use crate::expand::MediaItem;
use ob_core::cancel::CancelToken;
use ob_detect::tile::{tiles_for_size, TilingConfig};
use ob_media::MediaKind;
use std::path::{Path, PathBuf};

/// Cost of decoding and re-encoding one video frame, relative to one inference
/// pass.
///
/// Video work is not all inference: every frame is decoded, composited and
/// re-encoded even on the frames the detector skips. This is a rough ratio, not
/// a measurement — it exists so that a long video at `detect_every = 10` is not
/// costed as though nine tenths of it were free.
const DECODE_COST_PER_FRAME: f64 = 0.05;

/// Assumed size for an item whose dimensions could not be read.
///
/// A guess is better than dropping the item: an unreadable header usually means
/// an odd container rather than an absent file, and costing it at zero would
/// make the total silently too low. 1080p is the middle of what this tool sees.
const ASSUMED_DIMS: (u32, u32) = (1920, 1080);

/// What the pass count depends on, besides the frame itself.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// The model's square input side, which decides when tiling kicks in.
    pub input_size: u32,
    pub tiling: TilingConfig,
    /// Video: the detector runs every Nth frame, the tracker coasts between.
    pub detect_every: u32,
    /// Ensemble size. Every member sees every frame, so two models is twice
    /// the inference — the single biggest lever a user has on run time.
    pub members: usize,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            input_size: 320,
            tiling: TilingConfig::default(),
            detect_every: 3,
            members: 1,
        }
    }
}

impl CostModel {
    /// Inference passes for one frame of the given size.
    fn passes_per_frame(&self, w: u32, h: u32) -> f64 {
        // `tiles_for_size` is the detector's own decision, mode and min_scale
        // included: it returns an empty grid when tiling is off or the frame is
        // small enough not to need it, leaving just the whole-frame pass.
        let tiles = tiles_for_size(w, h, self.input_size, &self.tiling).len();
        (1 + tiles) as f64 * self.members.max(1) as f64
    }

    /// How many frames of a video the detector actually runs on.
    fn detected_frames(&self, frames: u64) -> f64 {
        let every = self.detect_every.max(1) as f64;
        (frames as f64 / every).ceil()
    }
}

/// How an item's size was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sizing {
    /// Dimensions (and, for video, length) were read from the file.
    Probed,
    /// The probe failed, so [`ASSUMED_DIMS`] was used.
    Assumed,
    /// A video whose length could not be determined. Its cost is unknowable
    /// without inventing a duration, so it is excluded from the total and
    /// reported separately rather than being allowed to invent a number.
    UnknownLength,
}

/// What a file is, as measured. Independent of any setting — this is the half
/// that costs I/O, so it is kept across changes to the cost model.
#[derive(Debug, Clone)]
pub struct ProbedItem {
    pub path: PathBuf,
    pub kind: MediaKind,
    pub dims: Option<(u32, u32)>,
    /// Total frames, for video.
    pub frames: Option<u64>,
    pub sizing: Sizing,
}

impl ProbedItem {
    /// Work units for this item under `model`. Pure and instant.
    pub fn work(&self, model: &CostModel) -> f64 {
        let (w, h) = self.dims.unwrap_or(ASSUMED_DIMS);
        match self.kind {
            MediaKind::Video => match self.frames {
                Some(n) => {
                    model.detected_frames(n) * model.passes_per_frame(w, h)
                        + n as f64 * DECODE_COST_PER_FRAME
                }
                None => 0.0,
            },
            _ => model.passes_per_frame(w, h),
        }
    }
}

/// One item's share of the batch, under a particular cost model.
#[derive(Debug, Clone)]
pub struct ItemCost {
    pub path: PathBuf,
    pub kind: MediaKind,
    /// Work units — inference passes, plus the decode term for video.
    pub work: f64,
    pub sizing: Sizing,
}

/// The measured shape of a batch.
#[derive(Debug, Clone, Default)]
pub struct Workload {
    pub items: Vec<ItemCost>,
    /// Sum of `work` over every item whose size is known well enough to cost.
    pub total_work: f64,
    pub images: usize,
    pub videos: usize,
    /// Videos of unknown length, excluded from `total_work`.
    pub unknown: usize,
}

impl Workload {
    /// Work units for one path, or `None` if it was not in the scan or could
    /// not be costed.
    pub fn work_for(&self, path: &Path) -> Option<f64> {
        self.items
            .iter()
            .find(|i| i.path == path)
            .map(|i| i.work)
            .filter(|w| *w > 0.0)
    }

    /// True when nothing could be costed, so callers fall back to counting
    /// files rather than showing an estimate built on nothing.
    pub fn is_empty(&self) -> bool {
        self.total_work <= 0.0
    }
}

/// Measure one file. The I/O half.
///
/// Never fails: a file that cannot be probed is still going to be processed, so
/// it is recorded at the assumed size and marked as such.
pub fn probe_item(item: &MediaItem) -> ProbedItem {
    match item.kind {
        MediaKind::Video => {
            let info = ob_media::video::probe(&item.path).ok();
            let dims = info.as_ref().map(|i| (i.width, i.height));
            let frames = info.as_ref().and_then(|i| i.frame_count);
            let sizing = match (frames, dims) {
                // Costing a video of unknown length would mean inventing a
                // duration, and being wrong about it by an unbounded factor.
                (None, _) => Sizing::UnknownLength,
                (Some(_), Some(_)) => Sizing::Probed,
                (Some(_), None) => Sizing::Assumed,
            };
            ProbedItem {
                path: item.path.clone(),
                kind: item.kind,
                dims,
                frames,
                sizing,
            }
        }
        _ => {
            // Header read only — `image_dimensions` does not decode the pixels,
            // which is what makes scanning tens of thousands of files
            // affordable.
            let (dims, sizing) = match image::image_dimensions(&item.path) {
                Ok(d) => (Some(d), Sizing::Probed),
                Err(_) => (None, Sizing::Assumed),
            };
            ProbedItem {
                path: item.path.clone(),
                kind: item.kind,
                dims,
                frames: None,
                sizing,
            }
        }
    }
}

/// Measure every file in a batch.
///
/// `cancel` is polled between items: a scan of a large tree of videos spends a
/// subprocess per file and must stop when the user changes their mind. A
/// cancelled scan returns what it managed to measure, which is still a usable
/// lower bound.
pub fn probe(items: &[MediaItem], cancel: &CancelToken) -> Vec<ProbedItem> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if cancel.is_cancelled() {
            break;
        }
        out.push(probe_item(item));
    }
    out
}

/// Cost an already-probed batch. The pure half — no I/O, safe to call as often
/// as a slider moves.
pub fn cost(probed: &[ProbedItem], model: &CostModel) -> Workload {
    let mut out = Workload::default();
    for p in probed {
        match p.kind {
            MediaKind::Video => out.videos += 1,
            _ => out.images += 1,
        }
        if p.sizing == Sizing::UnknownLength {
            out.unknown += 1;
        }
        let work = p.work(model);
        out.total_work += work;
        out.items.push(ItemCost {
            path: p.path.clone(),
            kind: p.kind,
            work,
            sizing: p.sizing,
        });
    }
    out
}

/// Seconds per work unit, learned from actual runs.
///
/// The scan says a batch is 40 000 passes; only the machine can say how long a
/// pass takes. That figure varies by two orders of magnitude between a CPU
/// session and a discrete GPU, so it is measured rather than assumed: seeded
/// from a deliberately conservative default, replaced by the real rate as soon
/// as the run produces one, and worth persisting so the *next* run can show an
/// accurate estimate before it starts.
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    secs_per_unit: f64,
    /// False while still on the built-in default, so callers can mark an
    /// estimate as provisional rather than presenting a guess as measurement.
    measured: bool,
}

/// Roughly a 320px CPU pass. Deliberately pessimistic: an estimate that comes
/// down as the run proceeds reads as progress, one that climbs reads as a lie.
const DEFAULT_SECS_PER_UNIT: f64 = 0.35;

impl Default for Calibration {
    fn default() -> Self {
        Self {
            secs_per_unit: DEFAULT_SECS_PER_UNIT,
            measured: false,
        }
    }
}

impl Calibration {
    /// Restore a rate learned on this machine by an earlier run.
    pub fn from_saved(secs_per_unit: Option<f64>) -> Self {
        match secs_per_unit {
            Some(s) if s.is_finite() && s > 0.0 => Self {
                secs_per_unit: s,
                measured: true,
            },
            _ => Self::default(),
        }
    }

    pub fn secs_per_unit(&self) -> f64 {
        self.secs_per_unit
    }

    pub fn is_measured(&self) -> bool {
        self.measured
    }

    /// Fold in an observation: `work` units took `elapsed` seconds.
    pub fn observe(&mut self, work: f64, elapsed: f64) {
        if work <= 0.0 || !elapsed.is_finite() || elapsed <= 0.0 {
            return;
        }
        // The whole run so far is one observation, so this replaces rather than
        // averages -- the input already covers everything measured.
        self.secs_per_unit = elapsed / work;
        self.measured = true;
    }

    /// Seconds for `work` units at the current rate.
    pub fn secs_for(&self, work: f64) -> f64 {
        work * self.secs_per_unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_detect::tile::TilingMode;

    fn model() -> CostModel {
        CostModel {
            input_size: 320,
            tiling: TilingConfig::default(),
            detect_every: 3,
            members: 1,
        }
    }

    fn item(path: &str, kind: MediaKind) -> MediaItem {
        MediaItem {
            path: PathBuf::from(path),
            kind,
        }
    }

    /// Write a real PNG so `image_dimensions` has a header to read.
    fn png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let p = dir.join(name);
        let buf = image::RgbImage::new(w, h);
        buf.save(&p).unwrap();
        p
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ob-estimate-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_big_frame_costs_more_than_a_small_one() {
        // The entire point: sized by what the detector will actually do, not by
        // file count. A 4K frame tiles; a thumbnail does not.
        let m = model();
        let small = m.passes_per_frame(320, 320);
        let large = m.passes_per_frame(3840, 2160);
        assert_eq!(small, 1.0, "a frame at model size needs one pass");
        assert!(
            large > small * 4.0,
            "a 4K frame costed at {large} against {small} for a thumbnail"
        );
    }

    #[test]
    fn tiling_off_costs_one_pass_whatever_the_size() {
        let m = CostModel {
            tiling: TilingConfig {
                mode: TilingMode::Off,
                ..Default::default()
            },
            ..model()
        };
        // With tiling off the frame is letterboxed into one input regardless of
        // its size, so an estimate that scaled with pixels would be badly wrong.
        assert_eq!(m.passes_per_frame(3840, 2160), 1.0);
        assert_eq!(m.passes_per_frame(64, 64), 1.0);
    }

    #[test]
    fn every_ensemble_member_multiplies_the_cost() {
        let one = model();
        let three = CostModel {
            members: 3,
            ..model()
        };
        // Adding models is the biggest lever a user has on run time, so the
        // estimate has to move when they use it.
        assert_eq!(
            three.passes_per_frame(3840, 2160),
            one.passes_per_frame(3840, 2160) * 3.0
        );
    }

    #[test]
    fn detect_every_reduces_the_frames_inferred_on() {
        let m = model();
        assert_eq!(m.detected_frames(300), 100.0);
        let dense = CostModel {
            detect_every: 1,
            ..model()
        };
        assert_eq!(dense.detected_frames(300), 300.0);
    }

    #[test]
    fn image_dimensions_are_read_from_the_header() {
        let dir = tmp("dims");
        let big = png(&dir, "big.png", 1600, 1200);
        let small = png(&dir, "small.png", 100, 80);

        let m = model();
        let a = probe_item(&item(big.to_str().unwrap(), MediaKind::Image));
        let b = probe_item(&item(small.to_str().unwrap(), MediaKind::Image));

        assert_eq!(a.sizing, Sizing::Probed);
        assert_eq!(a.dims, Some((1600, 1200)));
        assert_eq!(b.dims, Some((100, 80)));
        assert!(a.work(&m) > b.work(&m));
    }

    #[test]
    fn an_unreadable_file_is_assumed_not_dropped() {
        let dir = tmp("unreadable");
        let bad = dir.join("truncated.png");
        std::fs::write(&bad, b"not a png").unwrap();

        let c = probe_item(&item(bad.to_str().unwrap(), MediaKind::Image));
        // Costing it at zero would make the total silently too low for a file
        // that is still going to be processed.
        assert_eq!(c.sizing, Sizing::Assumed);
        assert!(c.work(&model()) > 0.0);
        assert_eq!(c.dims, None);
    }

    #[test]
    fn a_video_of_unknown_length_is_excluded_rather_than_invented() {
        // No ffprobe result here, so this exercises the same path a container
        // with no frame count takes.
        let c = probe_item(&item("/nonexistent/clip.mp4", MediaKind::Video));
        assert_eq!(c.sizing, Sizing::UnknownLength);
        assert_eq!(c.work(&model()), 0.0);

        let w = cost(
            &probe(
                &[item("/nonexistent/clip.mp4", MediaKind::Video)],
                &CancelToken::new(),
            ),
            &model(),
        );
        assert_eq!(w.unknown, 1);
        assert_eq!(w.videos, 1);
        // Reported separately so the UI can say "plus one of unknown length"
        // instead of quietly guessing at a duration.
        assert!(w.is_empty());
    }

    #[test]
    fn a_scan_totals_the_batch_and_can_be_cancelled() {
        let dir = tmp("scan");
        let items: Vec<MediaItem> = (0..4)
            .map(|i| {
                let p = png(&dir, &format!("{i}.png"), 800, 600);
                item(p.to_str().unwrap(), MediaKind::Image)
            })
            .collect();

        let probed = probe(&items, &CancelToken::new());
        let w = cost(&probed, &model());
        assert_eq!(w.images, 4);
        assert_eq!(w.items.len(), 4);
        assert!((w.total_work - w.items.iter().map(|i| i.work).sum::<f64>()).abs() < 1e-9);
        assert_eq!(w.work_for(&items[0].path), Some(w.items[0].work));

        // Re-costing is pure: the same probe answers a new cost model with no
        // second pass over the disk. This is what lets a slider move the
        // estimate without re-reading every header in the batch.
        let doubled = cost(
            &probed,
            &CostModel {
                members: 2,
                ..model()
            },
        );
        assert!((doubled.total_work - w.total_work * 2.0).abs() < 1e-9);

        // A scan of a large tree spends a subprocess per video and has to stop
        // when the user changes their mind.
        let cancel = CancelToken::new();
        cancel.cancel();
        assert!(probe(&items, &cancel).is_empty());
    }

    #[test]
    fn calibration_is_replaced_by_the_first_real_measurement() {
        let mut c = Calibration::default();
        assert!(!c.is_measured());
        let guessed = c.secs_for(100.0);

        // 100 units took 10s: 0.1 s/unit, whatever the default claimed.
        c.observe(100.0, 10.0);
        assert!(c.is_measured());
        assert!((c.secs_per_unit() - 0.1).abs() < 1e-9);
        assert!((c.secs_for(100.0) - 10.0).abs() < 1e-9);
        assert_ne!(guessed, c.secs_for(100.0));
    }

    #[test]
    fn a_nonsense_observation_never_corrupts_the_rate() {
        let mut c = Calibration::default();
        c.observe(100.0, 10.0);
        let good = c.secs_per_unit();

        // A run that finished with no measurable work, or a clock that went
        // backwards, must not turn the rate into zero or NaN and poison every
        // later estimate.
        c.observe(0.0, 5.0);
        c.observe(100.0, 0.0);
        c.observe(100.0, f64::NAN);
        assert_eq!(c.secs_per_unit(), good);
    }

    #[test]
    fn a_saved_rate_is_restored_and_a_bad_one_ignored() {
        assert!(Calibration::from_saved(Some(0.02)).is_measured());
        assert!((Calibration::from_saved(Some(0.02)).secs_per_unit() - 0.02).abs() < 1e-9);
        // Anything a corrupt settings file could hold falls back to the default
        // rather than producing an infinite or negative ETA.
        for bad in [None, Some(0.0), Some(-1.0), Some(f64::NAN)] {
            assert!(!Calibration::from_saved(bad).is_measured());
        }
    }
}
