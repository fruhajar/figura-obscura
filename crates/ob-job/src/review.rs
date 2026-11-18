//! Two-phase batch: detect everything, let someone look at it, then write.
//!
//! The one-phase [`run`](crate::run) decodes a frame, detects on it and
//! censors it in a single pass, which is the right shape when the settings are
//! already trusted. It gives you nothing to correct, though: the first time you
//! see what the detector missed is after the output exists.
//!
//! So this splits the pass in two. [`detect_pass`] runs the model and keeps
//! only the geometry, writing nothing. [`commit_pass`] takes that geometry
//! back, after a human has moved, added or deleted boxes, and writes the files,
//! **with no detector involved at all**. Nothing between the two can change
//! what gets covered except the person reviewing it.
//!
//! What is stored is deliberately small: a box is four floats, a category and a
//! score. A folder of ten thousand photos averaging ten detections each is
//! about two megabytes; an hour of video at the default one-in-three detection
//! rate is under three. The expensive thing a phase boundary costs is not
//! memory, it is that the pixels have to be decoded twice. See
//! [`commit_pass`].
//!
//! Styling is *not* stored. Review decides geometry; the profile decides
//! appearance, and it is re-resolved at commit through
//! [`CensorConfig::resolve`](ob_core::censor::CensorConfig::resolve), the same
//! call the live preview makes. That keeps changing a censor style after a
//! review from needing another detection pass, which is the same seam the
//! preview's detect/compose split already runs along.

use crate::expand::{self, output_path, MediaItem};
use crate::{FrameTicker, JobConfig, JobError, ProgressEvent, RunSummary};
use ob_censor::apply_boxes;
use ob_core::geometry::Detection;
use ob_core::profile::Profile;
use ob_core::taxonomy::{cat, Category};
use ob_detect::Detector;
use ob_media::video::{FfmpegSink, FfmpegSource};
use ob_media::{load_image, save_image_owned, FrameSink, FrameSource, MediaKind};
use ob_track::Tracker;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// The category a box drawn during review carries.
///
/// One name for the sentinel the editor, the review and the censor policy all
/// have to agree on. See [`cat::UNCLASSIFIED`] for why it is the value it is
/// and why the style lookup special-cases it.
pub const MANUAL: Category = cat::UNCLASSIFIED;

/// What one frame holds, after filtering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewFrame {
    /// Frame number in decode order. Always 0 for an image.
    pub index: u64,
    /// Already passed through the profile's category filter, so this is what
    /// would have been censored, not everything the model saw.
    pub detections: Vec<Detection>,
}

impl ReviewFrame {
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }
}

/// One input file, as the detect pass found it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewItem {
    pub path: PathBuf,
    /// Where the commit pass will write it.
    pub output: PathBuf,
    pub kind: MediaKind,
    pub width: u32,
    pub height: u32,
    /// Images: exactly one entry, index 0. Videos: one per frame the detector
    /// ran on, in decode order.
    pub frames: Vec<ReviewFrame>,
    /// Video: how often the detector ran. Kept per item rather than read from
    /// the config at commit, so a stored review still replays correctly if the
    /// setting has since been changed.
    pub detect_every: u32,
    /// Video: frames in the clip, where the probe reported a count.
    pub frame_count: Option<u64>,
    /// Set when the detect pass could not read this file. It is kept in the
    /// list rather than dropped: a file that silently vanished between the two
    /// phases is exactly what a review is supposed to surface.
    pub error: Option<String>,
}

impl ReviewItem {
    /// Total boxes across every frame.
    pub fn region_count(&self) -> usize {
        self.frames.iter().map(|f| f.detections.len()).sum()
    }

    /// Whether the detector found nothing anywhere in this file.
    ///
    /// The interesting case in a review, and the reason the detect pass keeps
    /// empty items instead of discarding them.
    pub fn is_clean(&self) -> bool {
        self.error.is_none() && self.region_count() == 0
    }

    /// The frames worth stepping through: those with something on them.
    pub fn frames_with_regions(&self) -> impl Iterator<Item = &ReviewFrame> {
        self.frames.iter().filter(|f| !f.is_empty())
    }

    /// How many frames a reviewer can step between.
    ///
    /// Every frame the detector *could* have run on, not only the ones it found
    /// something on. Those are different sets, and the difference is the whole
    /// point: the frames with nothing on them are where a miss is, and a slider
    /// that only visited frames with boxes already on them could never be used
    /// to add one.
    pub fn slot_count(&self) -> usize {
        match self.kind {
            MediaKind::Video => {
                let frames = self.frame_count.unwrap_or(0);
                let every = self.detect_every.max(1) as u64;
                // Frame 0 always counts, so a clip of any length has one slot.
                (frames.div_ceil(every)).max(1) as usize
            }
            _ => 1,
        }
    }

    /// The frame number behind slot `slot`.
    pub fn slot_index(&self, slot: usize) -> u64 {
        match self.kind {
            MediaKind::Video => slot as u64 * self.detect_every.max(1) as u64,
            _ => 0,
        }
    }

    /// The slot a frame number falls in, for jumping to a frame that has boxes.
    pub fn slot_of(&self, index: u64) -> usize {
        (index / self.detect_every.max(1) as u64) as usize
    }

    /// The detections on a slot, which may be none at all.
    pub fn detections_at(&self, slot: usize) -> &[Detection] {
        let index = self.slot_index(slot);
        self.frames
            .iter()
            .find(|f| f.index == index)
            .map(|f| f.detections.as_slice())
            .unwrap_or_default()
    }

    /// Replace a slot's detections, creating the frame entry if the detector
    /// found nothing there.
    ///
    /// An entry emptied this way is **kept**, not removed. On video that
    /// distinction is load-bearing: an entry with no detections tells the
    /// tracker at commit that the region is genuinely gone, where a missing
    /// entry would leave it coasting, so a box deleted in review would come
    /// back in the output.
    pub fn set_detections_at(&mut self, slot: usize, dets: Vec<Detection>) {
        let index = self.slot_index(slot);
        match self.frames.iter_mut().find(|f| f.index == index) {
            Some(frame) => frame.detections = dets,
            None => {
                if dets.is_empty() {
                    // Nothing was there and nothing was added; storing an empty
                    // entry for every slot someone merely scrolled past would
                    // turn an untouched review of a long clip into thousands of
                    // rows that all say nothing.
                    return;
                }
                let at = self.frames.partition_point(|f| f.index < index);
                self.frames.insert(
                    at,
                    ReviewFrame {
                        index,
                        detections: dets,
                    },
                );
            }
        }
    }
}

/// A whole batch, reviewed or awaiting review.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Review {
    pub items: Vec<ReviewItem>,
}

impl Review {
    pub fn region_count(&self) -> usize {
        self.items.iter().map(|i| i.region_count()).sum()
    }

    /// Items the detector found nothing in: the ones to look at first.
    pub fn clean_count(&self) -> usize {
        self.items.iter().filter(|i| i.is_clean()).count()
    }

    pub fn failed_count(&self) -> usize {
        self.items.iter().filter(|i| i.error.is_some()).count()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Phase one: run the detector over the batch and keep only the geometry.
///
/// Writes nothing. Emits the same [`ProgressEvent`]s a real run does, so a
/// caller's progress bar needs no special case for it.
pub fn detect_pass(
    cfg: &JobConfig,
    detector: &dyn Detector,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> Result<Review, JobError> {
    let items = expand::expand(&cfg.input)?;
    progress(ProgressEvent::Discovered(items.len()));

    let (images, videos): (Vec<_>, Vec<_>) =
        items.into_iter().partition(|i| i.kind == MediaKind::Image);

    // Images run in parallel as they do in a real batch, so the collector has
    // to be shared. Order is restored afterwards.
    let out: Mutex<Vec<ReviewItem>> = Mutex::new(Vec::new());
    let skipped = AtomicUsize::new(0);

    images.par_iter().for_each(|item| {
        if cfg.cancel.is_cancelled() {
            skipped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        progress(ProgressEvent::FileStarted(item.path.clone()));
        let found = detect_image(item, cfg, detector);
        report(progress, &found);
        out.lock().unwrap_or_else(|e| e.into_inner()).push(found);
    });

    let mut collected = out.into_inner().unwrap_or_else(|e| e.into_inner());

    for item in &videos {
        if cfg.cancel.is_cancelled() {
            skipped.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        progress(ProgressEvent::FileStarted(item.path.clone()));
        let found = detect_video(item, cfg, detector, progress);
        report(progress, &found);
        collected.push(found);
    }

    // Stable, and the same order the queue shows: a review that reshuffled the
    // batch every time it ran would be impossible to work through.
    collected.sort_by(|a, b| a.path.cmp(&b.path));

    if cfg.cancel.is_cancelled() {
        progress(ProgressEvent::Cancelled {
            remaining: skipped.load(Ordering::Relaxed),
        });
    }
    let failed = collected.iter().filter(|i| i.error.is_some()).count();
    progress(ProgressEvent::Finished {
        ok: collected.len() - failed,
        failed,
    });
    Ok(Review { items: collected })
}

fn report(progress: &(dyn Fn(ProgressEvent) + Sync), item: &ReviewItem) {
    match &item.error {
        Some(error) => progress(ProgressEvent::FileError {
            path: item.path.clone(),
            error: error.clone(),
        }),
        None => progress(ProgressEvent::FileDone {
            path: item.path.clone(),
            regions: item.region_count(),
        }),
    }
}

/// An item that could not be read, so the failure survives into the review.
fn failed_item(item: &MediaItem, cfg: &JobConfig, error: String) -> ReviewItem {
    ReviewItem {
        output: output_path(&item.path, &cfg.input.inputs, &cfg.output_dir),
        path: item.path.clone(),
        kind: item.kind,
        width: 0,
        height: 0,
        frames: Vec::new(),
        detect_every: cfg.detect_every.max(1),
        frame_count: None,
        error: Some(error),
    }
}

fn detect_image(item: &MediaItem, cfg: &JobConfig, detector: &dyn Detector) -> ReviewItem {
    let frame = match load_image(&item.path) {
        Ok(f) => f,
        Err(e) => return failed_item(item, cfg, e.to_string()),
    };
    let detections = match detector.detect(&frame) {
        Ok(dets) => cfg.profile.filter.select_owned(&dets),
        Err(e) => {
            // The fail-closed policy has to mean the same thing here as in a
            // real run, or a review would quietly approve a frame the batch
            // would have blanked. `Blank` is expressed as a box over the whole
            // frame, which is both what it does and something the reviewer can
            // see and argue with.
            match cfg.profile.on_detect_failure {
                ob_core::profile::OnDetectFailure::Skip => {
                    return failed_item(item, cfg, e.to_string())
                }
                ob_core::profile::OnDetectFailure::PassThrough => Vec::new(),
                ob_core::profile::OnDetectFailure::Blank => vec![whole_frame(&frame)],
            }
        }
    };
    ReviewItem {
        output: output_path(&item.path, &cfg.input.inputs, &cfg.output_dir),
        path: item.path.clone(),
        kind: item.kind,
        width: frame.width,
        height: frame.height,
        frames: vec![ReviewFrame {
            index: 0,
            detections,
        }],
        detect_every: cfg.detect_every.max(1),
        frame_count: None,
        error: None,
    }
}

/// A box covering everything, for the fail-closed policy.
fn whole_frame(frame: &ob_core::geometry::Frame) -> Detection {
    Detection {
        bbox: ob_core::geometry::BBox::new(0.0, 0.0, frame.width_f(), frame.height_f()),
        category: MANUAL,
        score: 1.0,
    }
}

fn detect_video(
    item: &MediaItem,
    cfg: &JobConfig,
    detector: &dyn Detector,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> ReviewItem {
    let mut source = match FfmpegSource::open(&item.path) {
        Ok(s) => s,
        Err(e) => return failed_item(item, cfg, e.to_string()),
    };
    let info = source.info().clone();
    let every = cfg.detect_every.max(1);
    let mut ticker = FrameTicker::new(&item.path, info.frame_count, progress);
    let mut frames = Vec::new();
    let mut index: u64 = 0;

    // One stage, not the three the real pipeline uses: nothing is being encoded
    // here, so there is no third stage to overlap with, and detection is the
    // only work worth the thread.
    loop {
        if cfg.cancel.is_cancelled() {
            break;
        }
        let frame = match source.next_frame() {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return failed_item(item, cfg, e.to_string()),
        };
        if index % every as u64 == 0 {
            match detector.detect(&frame) {
                Ok(dets) => {
                    let selected = cfg.profile.filter.select_owned(&dets);
                    // Empty frames are not stored. On a clip where most frames
                    // hold nothing this is the difference between a few
                    // kilobytes and a few megabytes, and the reviewer is
                    // stepping between frames that have something on them.
                    if !selected.is_empty() {
                        frames.push(ReviewFrame {
                            index,
                            detections: selected,
                        });
                    }
                }
                Err(e) => {
                    if cfg.profile.on_detect_failure == ob_core::profile::OnDetectFailure::Skip {
                        return failed_item(item, cfg, e.to_string());
                    }
                    if cfg.profile.on_detect_failure == ob_core::profile::OnDetectFailure::Blank {
                        frames.push(ReviewFrame {
                            index,
                            detections: vec![whole_frame(&frame)],
                        });
                    }
                }
            }
        }
        index += 1;
        ticker.tick();
    }

    ReviewItem {
        output: output_path(&item.path, &cfg.input.inputs, &cfg.output_dir),
        path: item.path.clone(),
        kind: item.kind,
        width: info.width,
        height: info.height,
        frames,
        detect_every: every,
        // What was actually decoded, which beats the probe's guess: a clip
        // whose container lies about its length would otherwise give the review
        // slider a range that does not exist.
        frame_count: Some(index),
        error: None,
    }
}

/// Phase two: write the files, using the reviewed geometry and no detector.
///
/// The pixels are decoded a second time here: phase one kept the boxes and
/// threw the frames away, because holding a batch's worth of decoded video in
/// memory is not an option. That second decode is what a review costs: roughly
/// a third again on a folder of PNGs, and a second full pass over a video. It
/// buys the guarantee that nothing is written until someone has looked at it.
pub fn commit_pass(
    cfg: &JobConfig,
    review: &Review,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> Result<RunSummary, JobError> {
    let pending: Vec<&ReviewItem> = review
        .items
        .iter()
        .filter(|i| i.error.is_none() && i.kind != MediaKind::Unknown)
        .collect();
    progress(ProgressEvent::Discovered(pending.len()));

    let (images, videos): (Vec<_>, Vec<_>) = pending
        .into_iter()
        .partition(|i| i.kind == MediaKind::Image);

    let ok = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    images.par_iter().for_each(|item| {
        commit_one(
            item,
            cfg,
            progress,
            &ok,
            &failed,
            &skipped,
            |item, cfg, _| commit_image(item, cfg),
        );
    });
    for item in &videos {
        commit_one(item, cfg, progress, &ok, &failed, &skipped, commit_video);
    }

    let summary = RunSummary {
        ok: ok.load(Ordering::Relaxed),
        failed: failed.load(Ordering::Relaxed),
        skipped: skipped.load(Ordering::Relaxed),
        cancelled: cfg.cancel.is_cancelled(),
    };
    if summary.cancelled {
        progress(ProgressEvent::Cancelled {
            remaining: summary.skipped,
        });
    }
    progress(ProgressEvent::Finished {
        ok: summary.ok,
        failed: summary.failed,
    });
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn commit_one(
    item: &ReviewItem,
    cfg: &JobConfig,
    progress: &(dyn Fn(ProgressEvent) + Sync),
    ok: &AtomicUsize,
    failed: &AtomicUsize,
    skipped: &AtomicUsize,
    write: impl Fn(&ReviewItem, &JobConfig, &(dyn Fn(ProgressEvent) + Sync)) -> Result<usize, JobError>,
) {
    if cfg.cancel.is_cancelled() {
        skipped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    progress(ProgressEvent::FileStarted(item.path.clone()));
    match write(item, cfg, progress) {
        Ok(regions) => {
            ok.fetch_add(1, Ordering::Relaxed);
            progress(ProgressEvent::FileDone {
                path: item.path.clone(),
                regions,
            });
        }
        Err(JobError::Cancelled) => {
            skipped.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            failed.fetch_add(1, Ordering::Relaxed);
            progress(ProgressEvent::FileError {
                path: item.path.clone(),
                error: e.to_string(),
            });
        }
    }
}

fn commit_image(item: &ReviewItem, cfg: &JobConfig) -> Result<usize, JobError> {
    let mut frame = load_image(&item.path)?;
    let detections: &[Detection] = item
        .frames
        .first()
        .map(|f| f.detections.as_slice())
        .unwrap_or_default();
    let boxes = cfg
        .profile
        .censor
        .resolve(detections, frame.width_f(), frame.height_f());
    apply_boxes(&mut frame, &boxes)?;
    if !cfg.dry_run {
        if let Some(parent) = item.output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        save_image_owned(frame, &item.output)?;
    }
    Ok(boxes.len())
}

/// Replay a reviewed video through the tracker and the encoder.
///
/// The tracker is fed exactly as the live pipeline feeds it: the reviewed
/// detections on the frames the detector ran on, `None` in between. So
/// hysteresis and smoothing behave identically and a box moved during review
/// carries through the frames that coast off it. That is also why the tracker
/// has to run at all rather than the stored boxes simply being stamped onto
/// their own frames: a region the detector drops for a frame or two would start
/// flickering, and the stamped version would cover one frame in three.
fn commit_video(
    item: &ReviewItem,
    cfg: &JobConfig,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> Result<usize, JobError> {
    if cfg.dry_run {
        return Ok(item.region_count());
    }
    if let Some(parent) = item.output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let by_index: HashMap<u64, &Vec<Detection>> = item
        .frames
        .iter()
        .map(|f| (f.index, &f.detections))
        .collect();

    let mut source = FfmpegSource::open(&item.path)?;
    let info = source.info().clone();
    let mut sink = Box::new(FfmpegSink::create(
        &item.output,
        &item.path,
        &info,
        cfg.video_opts.clone(),
    )?);

    let result = encode_reviewed(
        item,
        &mut source,
        &mut sink,
        &info,
        cfg,
        &by_index,
        progress,
    );
    match result {
        Ok(total) => {
            if let Err(e) = sink.finish() {
                let _ = std::fs::remove_file(&item.output);
                return Err(e.into());
            }
            Ok(total)
        }
        Err(e) => {
            // Same reasoning as the one-phase path: a partial encode is not a
            // censored copy of anything, and leaving it in the output folder
            // invites someone to publish it.
            drop(sink);
            let _ = std::fs::remove_file(&item.output);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_reviewed(
    item: &ReviewItem,
    source: &mut FfmpegSource,
    sink: &mut FfmpegSink,
    info: &ob_media::video::VideoInfo,
    cfg: &JobConfig,
    by_index: &HashMap<u64, &Vec<Detection>>,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> Result<usize, JobError> {
    let every = item.detect_every.max(1) as u64;
    let mut tracker = Tracker::new(cfg.track);
    let mut ticker = FrameTicker::new(&item.path, info.frame_count, progress);
    let mut total = 0usize;
    let mut index: u64 = 0;
    let empty: Vec<Detection> = Vec::new();

    while let Some(mut frame) = source.next_frame()? {
        if cfg.cancel.is_cancelled() {
            return Err(JobError::Cancelled);
        }
        // A frame the detector ran on gets `Some`, even when the review left it
        // with nothing on it: an empty `Some` tells the tracker the region is
        // genuinely gone and lets its ttl run down, where `None` would have it
        // coast on forever. Dropping that distinction is how a deleted box
        // comes back.
        let fed = if index % every == 0 {
            Some(by_index.get(&index).copied().unwrap_or(&empty).as_slice())
        } else {
            None
        };
        let smoothed = tracker.update(fed);
        let boxes = cfg
            .profile
            .censor
            .resolve(&smoothed, frame.width_f(), frame.height_f());
        apply_boxes(&mut frame, &boxes)?;
        total += boxes.len();
        sink.put_frame(&frame)?;
        ticker.tick();
        index += 1;
    }
    Ok(total)
}

/// Re-resolve a frame's boxes for display, exactly as the commit will.
///
/// The one call a review surface should use to draw: going through
/// [`CensorConfig::resolve`](ob_core::censor::CensorConfig::resolve) here, and
/// through it again at commit, is what stops the screen and the file from being
/// two interpretations of one profile.
pub fn resolve_for_display(
    detections: &[Detection],
    profile: &Profile,
    width: u32,
    height: u32,
) -> Vec<ob_core::censor::CensorBox> {
    profile
        .censor
        .resolve(detections, width as f32, height as f32)
}

/// Load the pixels a review needs to show for one frame of one item.
///
/// Images decode directly. Videos seek: the review steps between frames that
/// are seconds apart, so decoding from the start to reach each one would make
/// stepping through a clip unusable. [`FfmpegSource::frame_at`] does an input
/// seek, which is the same mechanism the dry-run sampler uses.
pub fn load_review_frame(
    item: &ReviewItem,
    index: u64,
) -> Result<ob_core::geometry::Frame, JobError> {
    match item.kind {
        MediaKind::Video => {
            let fps = ob_media::video::probe(&item.path)?.fps;
            Ok(ob_media::video::frame_at(&item.path, index, fps)?)
        }
        _ => Ok(load_image(&item.path)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::geometry::BBox;
    use ob_core::taxonomy::cat;

    fn det(x: f32) -> Detection {
        Detection {
            bbox: BBox::new(x, 10.0, x + 20.0, 40.0),
            category: cat::FEMALE_BREAST_EXPOSED,
            score: 0.9,
        }
    }

    fn image_item(name: &str, dets: Vec<Detection>) -> ReviewItem {
        ReviewItem {
            path: PathBuf::from(name),
            output: PathBuf::from("out").join(name),
            kind: MediaKind::Image,
            width: 200,
            height: 100,
            frames: vec![ReviewFrame {
                index: 0,
                detections: dets,
            }],
            detect_every: 3,
            frame_count: None,
            error: None,
        }
    }

    use crate::expand::InputSpec;
    use crate::{run, DEFAULT_DRY_RUN_FRAMES};
    use ob_core::cancel::CancelToken;
    use ob_core::geometry::Frame;
    use ob_media::video::VideoEncodeOpts;
    use ob_track::TrackConfig;

    struct FakeDetector {
        dets: Vec<Detection>,
    }
    impl Detector for FakeDetector {
        fn detect(&self, _f: &Frame) -> Result<Vec<Detection>, ob_detect::DetectError> {
            Ok(self.dets.clone())
        }
    }

    fn checker(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                if (x + y) % 2 == 0 {
                    let p = (y * w as usize + x) * 3;
                    data[p..p + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ob-review-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cfg<'a>(
        profile: &'a Profile,
        input: &std::path::Path,
        out: &std::path::Path,
    ) -> JobConfig<'a> {
        JobConfig {
            profile,
            input: InputSpec {
                inputs: vec![input.to_path_buf()],
                recursive: true,
                include: Vec::new(),
                exclude: Vec::new(),
            },
            output_dir: out.to_path_buf(),
            dry_run: false,
            dry_run_frames: DEFAULT_DRY_RUN_FRAMES,
            detect_every: 1,
            track: TrackConfig::default(),
            video_opts: VideoEncodeOpts::default(),
            cancel: CancelToken::new(),
        }
    }

    /// The load-bearing property of the whole feature: reviewing a batch and
    /// changing nothing must write exactly the files a direct run writes.
    ///
    /// If these can differ, then "review before writing" is reviewing something
    /// other than the output, and every other guarantee here is decoration.
    #[test]
    fn a_review_that_changes_nothing_writes_the_same_pixels() {
        let dir = scratch("faithful");
        let src = dir.join("in");
        std::fs::create_dir_all(&src).unwrap();
        for i in 0..3 {
            ob_media::save_image(&checker(32, 24), &src.join(format!("{i}.png"))).unwrap();
        }
        let detector = FakeDetector {
            dets: vec![det(4.0), det(18.0)],
        };
        let profile = Profile::default();

        let direct_out = dir.join("direct");
        run(&cfg(&profile, &src, &direct_out), &detector, &|_| {}).unwrap();

        let review_out = dir.join("reviewed");
        let c = cfg(&profile, &src, &review_out);
        let review = detect_pass(&c, &detector, &|_| {}).unwrap();
        assert_eq!(review.items.len(), 3);
        assert_eq!(review.region_count(), 6);
        commit_pass(&c, &review, &|_| {}).unwrap();

        for i in 0..3 {
            let a = ob_media::load_image(&direct_out.join(format!("{i}.png"))).unwrap();
            let b = ob_media::load_image(&review_out.join(format!("{i}.png"))).unwrap();
            assert_eq!(a.data, b.data, "file {i} differs between the two paths");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deleting a box during review has to actually remove it from the output.
    #[test]
    fn an_edited_review_writes_what_the_edit_says() {
        let dir = scratch("edited");
        let src = dir.join("in");
        std::fs::create_dir_all(&src).unwrap();
        ob_media::save_image(&checker(32, 24), &src.join("a.png")).unwrap();
        let detector = FakeDetector {
            dets: vec![det(4.0), det(18.0)],
        };
        let profile = Profile::default();
        let out = dir.join("out");
        let c = cfg(&profile, &src, &out);

        let mut review = detect_pass(&c, &detector, &|_| {}).unwrap();
        // Keep one box, drop the other: the commonest review edit there is.
        review.items[0].frames[0].detections.truncate(1);
        commit_pass(&c, &review, &|_| {}).unwrap();
        let edited = ob_media::load_image(&out.join("a.png")).unwrap();

        // The same batch run directly covers both regions, so the two files
        // must differ: if they matched, the edit was ignored.
        let direct = dir.join("direct");
        run(&cfg(&profile, &src, &direct), &detector, &|_| {}).unwrap();
        let full = ob_media::load_image(&direct.join("a.png")).unwrap();
        assert_ne!(
            edited.data, full.data,
            "the deleted box was censored anyway"
        );

        // And it must match a run that only ever saw the one box.
        let single = FakeDetector {
            dets: vec![det(4.0)],
        };
        let one = dir.join("one");
        run(&cfg(&profile, &src, &one), &single, &|_| {}).unwrap();
        let expected = ob_media::load_image(&one.join("a.png")).unwrap();
        assert_eq!(edited.data, expected.data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file the detector found nothing in still has to reach the review, and
    /// still has to be written.
    #[test]
    fn clean_files_are_kept_and_still_committed() {
        let dir = scratch("clean");
        let src = dir.join("in");
        std::fs::create_dir_all(&src).unwrap();
        ob_media::save_image(&checker(16, 16), &src.join("a.png")).unwrap();
        let detector = FakeDetector { dets: Vec::new() };
        let profile = Profile::default();
        let out = dir.join("out");
        let c = cfg(&profile, &src, &out);

        let review = detect_pass(&c, &detector, &|_| {}).unwrap();
        assert_eq!(review.clean_count(), 1);
        assert_eq!(review.region_count(), 0);

        let summary = commit_pass(&c, &review, &|_| {}).unwrap();
        assert_eq!(summary.ok, 1);
        // Copied through untouched, not skipped: the output folder is supposed
        // to be a complete copy of the batch.
        assert!(out.join("a.png").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_review_round_trips_through_json() {
        // The format is what a saved review and a `--review` handoff between
        // the CLI and the GUI would both use, so it has to survive the trip.
        let review = Review {
            items: vec![
                image_item("a.png", vec![det(10.0), det(50.0)]),
                image_item("b.png", vec![]),
            ],
        };
        let back = Review::from_json(&review.to_json().unwrap()).unwrap();
        assert_eq!(back, review);
    }

    #[test]
    fn counts_separate_clean_files_from_failed_ones() {
        let mut broken = image_item("c.png", vec![]);
        broken.error = Some("could not decode".into());
        let review = Review {
            items: vec![
                image_item("a.png", vec![det(10.0), det(50.0)]),
                image_item("b.png", vec![]),
                broken,
            ],
        };
        assert_eq!(review.region_count(), 2);
        // The file that failed is not "clean"; nothing was looked at in it.
        assert_eq!(review.clean_count(), 1);
        assert_eq!(review.failed_count(), 1);
    }

    #[test]
    fn display_resolution_matches_what_a_commit_would_draw() {
        // Both sides go through `CensorConfig::resolve`, so this is really a
        // guard against someone later giving the review surface its own
        // shortcut for padding or style lookup.
        let profile = Profile::default();
        let dets = vec![det(10.0)];
        let shown = resolve_for_display(&dets, &profile, 200, 100);
        let committed = profile.censor.resolve(&dets, 200.0, 100.0);
        assert_eq!(shown, committed);
        assert_eq!(shown.len(), 1);
        // Padding is in the stored rect, not re-applied at draw time.
        assert_ne!(shown[0].rect, dets[0].bbox);
    }

    #[test]
    fn the_manual_category_is_one_no_detector_emits() {
        // A drawn box must never be matched against a detected one by the
        // tracker, which matches within a category.
        assert!(!ob_core::taxonomy::NUDENET_CATEGORIES.contains(&MANUAL));
        assert_eq!(MANUAL, cat::UNCLASSIFIED);
    }

    #[test]
    fn a_hand_drawn_box_is_not_styled_as_a_face() {
        // It borrows `Part::Face` to have a category at all, and Face is one of
        // the per-part overrides the Tuning page offers, so without the
        // carve-out, drawing a box over a licence plate during a review would
        // style it with whatever the user chose for faces.
        use ob_core::censor::{CensorConfig, CensorStyle};
        let mut cfg = CensorConfig {
            default_style: CensorStyle::Pixelate { block: 8 },
            ..Default::default()
        };
        cfg.per_part.insert(
            ob_core::taxonomy::Part::Face,
            CensorStyle::SolidFill {
                color: [1, 2, 3, 255],
            },
        );
        assert_eq!(cfg.style_for(&MANUAL), &CensorStyle::Pixelate { block: 8 });
        // A real face still gets the override.
        assert_eq!(
            cfg.style_for(&cat::FACE_FEMALE),
            &CensorStyle::SolidFill {
                color: [1, 2, 3, 255]
            }
        );
    }

    #[test]
    fn an_emptied_frame_still_tells_the_tracker_the_region_is_gone() {
        // The distinction this pins is the one that makes a deleted box stay
        // deleted: a detected frame reviewed down to nothing must feed the
        // tracker `Some([])`, which ages the track out, not `None`, which
        // coasts it forward.
        let mut tracker = Tracker::new(ob_track::TrackConfig {
            ttl: 2,
            ..Default::default()
        });
        assert_eq!(tracker.update(Some(&[det(10.0)])).len(), 1);
        // Coasting keeps it alive.
        assert_eq!(tracker.update(None).len(), 1);
        // An explicit empty frame starts it dying; ttl 2 means two of them.
        tracker.update(Some(&[]));
        let left = tracker.update(Some(&[]));
        assert!(left.is_empty(), "a deleted box came back: {left:?}");
    }

    #[test]
    fn video_slots_cover_every_frame_the_detector_could_have_run_on() {
        let mut item = image_item("clip.mp4", vec![]);
        item.kind = MediaKind::Video;
        item.detect_every = 3;
        item.frame_count = Some(10);
        item.frames = vec![ReviewFrame {
            index: 6,
            detections: vec![det(10.0)],
        }];
        // 10 frames at one in three: 0, 3, 6, 9.
        assert_eq!(item.slot_count(), 4);
        assert_eq!(item.slot_index(2), 6);
        assert_eq!(item.slot_of(6), 2);
        // A slot the detector found nothing on is still visitable, and empty.
        assert!(item.detections_at(0).is_empty());
        assert_eq!(item.detections_at(2).len(), 1);
    }

    #[test]
    fn a_box_added_to_an_undetected_slot_lands_in_frame_order() {
        let mut item = image_item("clip.mp4", vec![]);
        item.kind = MediaKind::Video;
        item.detect_every = 3;
        item.frame_count = Some(12);
        item.frames = vec![ReviewFrame {
            index: 6,
            detections: vec![det(10.0)],
        }];
        item.set_detections_at(1, vec![det(40.0)]);
        let idx: Vec<u64> = item.frames.iter().map(|f| f.index).collect();
        assert_eq!(idx, vec![3, 6], "commit replays in frame order");
    }

    #[test]
    fn emptying_a_detected_slot_keeps_the_entry() {
        let mut item = image_item("clip.mp4", vec![]);
        item.kind = MediaKind::Video;
        item.detect_every = 3;
        item.frame_count = Some(12);
        item.frames = vec![ReviewFrame {
            index: 3,
            detections: vec![det(10.0)],
        }];
        item.set_detections_at(1, Vec::new());
        // Kept, and empty: that is what tells the tracker the region is gone.
        assert_eq!(item.frames.len(), 1);
        assert!(item.frames[0].detections.is_empty());

        // But a slot that never had anything and still has nothing adds no row.
        item.set_detections_at(2, Vec::new());
        assert_eq!(item.frames.len(), 1);
    }

    #[test]
    fn frames_with_regions_skips_the_empty_ones() {
        let mut item = image_item("clip.mp4", vec![]);
        item.kind = MediaKind::Video;
        item.frames = vec![
            ReviewFrame {
                index: 0,
                detections: vec![det(10.0)],
            },
            ReviewFrame {
                index: 3,
                detections: vec![],
            },
            ReviewFrame {
                index: 6,
                detections: vec![det(20.0)],
            },
        ];
        let idx: Vec<u64> = item.frames_with_regions().map(|f| f.index).collect();
        assert_eq!(idx, vec![0, 6]);
        assert_eq!(item.region_count(), 2);
    }
}
