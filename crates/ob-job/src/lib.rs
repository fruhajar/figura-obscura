//! # ob-job
//!
//! The batch engine: expands inputs (R6), runs the shared
//! `frame → detections → censored frame` pipeline over each item with per-file
//! error isolation, emits progress, and honors dry-run and the fail-closed
//! policy. Images are processed in parallel; videos are processed one at a time
//! (each already saturates the machine via ffmpeg + inference).

pub mod estimate;
pub mod expand;

use expand::{output_path, InputSpec, MediaItem};
use ob_censor::apply as apply_censor;
use ob_core::cancel::CancelToken;
use ob_core::geometry::{Detection, Frame};
use ob_core::profile::{OnDetectFailure, Profile};
use ob_detect::Detector;
use ob_media::video::{FfmpegSampler, FfmpegSink, FfmpegSource, VideoEncodeOpts};
use ob_media::{classify, load_image, save_image_owned, FrameSink, FrameSource, MediaKind};
use ob_track::{TrackConfig, Tracker};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Frames a dry run samples from each video by default.
///
/// Twelve is a compromise: enough that a clip with any censorable content in it
/// is very unlikely to show zero, few enough that dry-running a folder of
/// videos stays a matter of seconds. `--dry-run-frames` overrides it.
pub const DEFAULT_DRY_RUN_FRAMES: usize = 12;

/// Everything a run needs besides the detector (which the caller builds from the
/// chosen model + `ob-models` cache path).
pub struct JobConfig<'a> {
    pub profile: &'a Profile,
    pub input: InputSpec,
    pub output_dir: PathBuf,
    /// Log what would happen without writing any output.
    pub dry_run: bool,
    /// Video: how many frames a dry run samples per clip. Ignored otherwise.
    pub dry_run_frames: usize,
    /// Video: run the detector every Nth frame; the tracker coasts between.
    pub detect_every: u32,
    pub track: TrackConfig,
    pub video_opts: VideoEncodeOpts,
    /// Cooperative stop signal. Polled between files and between video frames,
    /// so a cancelled run always leaves whole, consistent output files behind
    /// rather than a truncated one. Default: never cancelled.
    pub cancel: CancelToken,
}

/// Progress events emitted during a run. The CLI renders a progress bar from
/// these; the GUI updates its queue.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Discovered(usize),
    FileStarted(PathBuf),
    FileDone {
        path: PathBuf,
        regions: usize,
    },
    FileError {
        path: PathBuf,
        error: String,
    },
    /// The run stopped early at the user's request; `remaining` items were
    /// never started.
    Cancelled {
        remaining: usize,
    },
    Finished {
        ok: usize,
        failed: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error(transparent)]
    Expand(#[from] expand::ExpandError),
    #[error(transparent)]
    Media(#[from] ob_media::MediaError),
    #[error(transparent)]
    Detect(#[from] ob_detect::DetectError),
    #[error(transparent)]
    Censor(#[from] ob_censor::CensorError),
    #[error("detection failed and policy is Skip: {0}")]
    FailClosedSkip(String),
    #[error("cancelled")]
    Cancelled,
}

/// Result tally for a run.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunSummary {
    pub ok: usize,
    pub failed: usize,
    /// Items never started because the run was cancelled.
    pub skipped: usize,
    /// Whether the run ended early at the user's request. Distinct from
    /// `failed > 0`: a cancelled run is not a broken one, and the CLI must not
    /// exit non-zero for it.
    pub cancelled: bool,
}

/// Run a full batch job. `progress` is called for each event (must be `Sync`
/// because image items run in parallel).
pub fn run(
    cfg: &JobConfig,
    detector: &dyn Detector,
    progress: &(dyn Fn(ProgressEvent) + Sync),
) -> Result<RunSummary, JobError> {
    let items = expand::expand(&cfg.input)?;
    progress(ProgressEvent::Discovered(items.len()));

    let (images, videos): (Vec<_>, Vec<_>) =
        items.into_iter().partition(|i| i.kind == MediaKind::Image);

    let ok = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    // Images in parallel, each isolated: one failure never aborts the batch.
    // rayon has no early exit, so cancelled items are skipped rather than
    // dropped from the iterator — the cost is one atomic load per item.
    images.par_iter().for_each(|item| {
        run_one(item, cfg, detector, progress, &ok, &failed, &skipped);
    });

    // Videos sequentially.
    for item in &videos {
        run_one(item, cfg, detector, progress, &ok, &failed, &skipped);
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
fn run_one(
    item: &MediaItem,
    cfg: &JobConfig,
    detector: &dyn Detector,
    progress: &(dyn Fn(ProgressEvent) + Sync),
    ok: &AtomicUsize,
    failed: &AtomicUsize,
    skipped: &AtomicUsize,
) {
    // Stopping *between* files is what makes cancellation safe: a file is
    // either fully processed or never begun, never half-written.
    if cfg.cancel.is_cancelled() {
        skipped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    progress(ProgressEvent::FileStarted(item.path.clone()));
    let out = output_path(&item.path, &cfg.input.inputs, &cfg.output_dir);

    let result = match item.kind {
        MediaKind::Image => process_image(&item.path, &out, cfg, detector),
        MediaKind::Video => process_video(&item.path, &out, cfg, detector),
        MediaKind::Unknown => Ok(0),
    };

    match result {
        Ok(regions) => {
            ok.fetch_add(1, Ordering::Relaxed);
            progress(ProgressEvent::FileDone {
                path: item.path.clone(),
                regions,
            });
        }
        // A run cancelled mid-video is not a failed file. Reporting it as one
        // would make the CLI exit non-zero and the GUI show a red error for
        // something the user asked for.
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

/// Censor a single frame in place per the profile. Returns regions censored.
/// Applies the fail-closed policy when detection errors.
pub fn censor_frame(
    frame: &mut Frame,
    detector: &dyn Detector,
    profile: &Profile,
) -> Result<usize, JobError> {
    match detector.detect(frame) {
        Ok(dets) => {
            let selected = profile.filter.select_owned(&dets);
            apply_censor(frame, &selected, &profile.censor)?;
            Ok(selected.len())
        }
        Err(e) => apply_detect_failure(frame, profile.on_detect_failure, &e.to_string()),
    }
}

/// The fail-closed policy, in one place.
///
/// Both the batch path and the preview path have to answer "detection failed —
/// now what", and they must answer it identically: a preview that quietly
/// passed a frame through while the batch blanked it would be worse than no
/// preview at all.
fn apply_detect_failure(
    frame: &mut Frame,
    policy: OnDetectFailure,
    error: &str,
) -> Result<usize, JobError> {
    match policy {
        OnDetectFailure::PassThrough => Ok(0),
        OnDetectFailure::Skip => Err(JobError::FailClosedSkip(error.to_string())),
        OnDetectFailure::Blank => {
            // Fail-closed: obliterate the whole frame rather than risk a leak.
            frame.data.fill(0);
            Ok(1)
        }
    }
}

fn process_image(
    input: &Path,
    output: &Path,
    cfg: &JobConfig,
    detector: &dyn Detector,
) -> Result<usize, JobError> {
    let mut frame = load_image(input)?;
    let regions = censor_frame(&mut frame, detector, cfg.profile)?;
    if !cfg.dry_run {
        if let Some(parent) = output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // The frame is not needed after this, so hand the encoder its buffer
        // rather than a copy of it.
        save_image_owned(frame, output)?;
    }
    Ok(regions)
}

/// A dry run over a video: detect on a sample of frames, write nothing.
///
/// This used to open the file, confirm it was readable and return 0. That was
/// not merely uninformative — `--dry-run` feeds `--report` and the miss
/// summary, so every clip landed in the "had none" column and dragged the
/// reported miss rate down. A folder of videos could look like a detector
/// failure when no inference had ever run.
///
/// Detection happens on every sampled frame, so `detect_every` does not apply:
/// it exists to let the tracker coast between neighbouring frames, and these
/// samples are seconds apart. For the same reason no tracker is used — there is
/// no continuity between the samples to smooth.
fn sample_video(input: &Path, cfg: &JobConfig, detector: &dyn Detector) -> Result<usize, JobError> {
    let mut source = FfmpegSampler::open(input, cfg.dry_run_frames)?;
    let mut total = 0usize;
    while let Some(mut frame) = source.next_frame()? {
        // Sampling a long clip is not instant, so honour cancellation here too.
        if cfg.cancel.is_cancelled() {
            return Err(JobError::Cancelled);
        }
        // The same call the batch and the preview make, so a dry run answers
        // `on_detect_failure` exactly the way the real run will.
        total += censor_frame(&mut frame, detector, cfg.profile)?;
    }
    Ok(total)
}

fn process_video(
    input: &Path,
    output: &Path,
    cfg: &JobConfig,
    detector: &dyn Detector,
) -> Result<usize, JobError> {
    if cfg.dry_run {
        return sample_video(input, cfg, detector);
    }
    if let Some(parent) = output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let source = FfmpegSource::open(input)?;
    let info = source.info().clone();
    let mut sink = Box::new(FfmpegSink::create(
        output,
        input,
        &info,
        cfg.video_opts.clone(),
    )?);

    let result = run_video_pipeline(source, &mut sink, &info, cfg, detector);

    match result {
        Ok(total) => {
            if let Err(e) = sink.finish() {
                let _ = std::fs::remove_file(output);
                return Err(e.into());
            }
            Ok(total)
        }
        Err(e) => {
            // Whatever the encoder managed to write is not a censored copy of
            // anything, and an output folder containing it invites someone to
            // publish it. Dropping the sink kills ffmpeg rather than letting it
            // see a clean EOF and finalise a truncated file.
            drop(sink);
            let _ = std::fs::remove_file(output);
            Err(e)
        }
    }
}

/// One frame on its way through the pipeline, with whatever the detector had
/// to say about it. `detections` is `None` on a frame the detector skipped —
/// which is a different thing from an empty vec, and the tracker treats it so.
struct Staged {
    frame: Frame,
    detections: Option<Vec<Detection>>,
}

/// Frames in flight per pipeline stage.
///
/// Bounded by *bytes* rather than by a frame count. The point of a queue here
/// is to let one stage run ahead of the next, but a frame is 6 MB at 1080p and
/// 24 MB at 2160p, so a depth that is comfortable at one resolution is most of
/// a gigabyte at the other. Sizing by count is the mistake the GIF encoder used
/// to make, and it ended with the machine in swap.
fn pipeline_depth(width: u32, height: u32) -> usize {
    /// Total in-flight frame memory to allow across both queues.
    const BUDGET: usize = 128 * 1024 * 1024;
    let frame = (width as usize * height as usize * 3).max(1);
    ((BUDGET / 2) / frame).clamp(1, 4)
}

/// Decode, detect, and censor-plus-encode a video as three concurrent stages.
///
/// The serial version did all three in one loop, so while inference ran both
/// ffmpeg processes sat idle and vice versa — on a GPU build, where inference
/// is comparatively quick, that left the device at a fraction of its capacity
/// and the run bounded by codec work that was not overlapping with anything.
///
/// ```text
///   decode    │ f1 │ f2 │ f3 │ f4 │
///   detect    │    │ f1 │ f2 │ f3 │
///   censor+   │    │    │ f1 │ f2 │
///   encode
/// ```
///
/// Three stages and not more, because of what is and is not order-dependent.
/// Detection is per-frame and stateless, so it could be widened further. The
/// tracker is neither: it carries state from frame to frame, and feeding it out
/// of order would change which regions get censored. So tracking, censoring and
/// encoding stay one ordered stage, and the queues between stages preserve
/// decode order end to end.
///
/// Output is unchanged by this. The tracker sees the same frames in the same
/// order with the same detections, and the encoder receives the same bytes.
fn run_video_pipeline(
    mut source: FfmpegSource,
    sink: &mut FfmpegSink,
    info: &ob_media::video::VideoInfo,
    cfg: &JobConfig,
    detector: &dyn Detector,
) -> Result<usize, JobError> {
    use std::sync::mpsc::sync_channel;

    let depth = pipeline_depth(info.width, info.height);
    let every = cfg.detect_every.max(1);

    std::thread::scope(|scope| {
        let (raw_tx, raw_rx) = sync_channel::<Result<(u32, Frame), JobError>>(depth);
        let (det_tx, det_rx) = sync_channel::<Result<Staged, JobError>>(depth);

        // --- Stage 1: decode ---------------------------------------------
        // A send that fails means a later stage has gone away, which is how
        // cancellation and errors propagate backwards: the receiver is dropped,
        // this notices, and the `FfmpegSource`'s own `Drop` reaps the decoder.
        scope.spawn(move || {
            let mut idx: u32 = 0;
            loop {
                if cfg.cancel.is_cancelled() {
                    break;
                }
                match source.next_frame() {
                    Ok(Some(frame)) => {
                        if raw_tx.send(Ok((idx, frame))).is_err() {
                            break;
                        }
                        idx = idx.wrapping_add(1);
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = raw_tx.send(Err(e.into()));
                        break;
                    }
                }
            }
        });

        // --- Stage 2: detect ---------------------------------------------
        // Every Nth frame goes through the model; the rest pass straight
        // through for the tracker to coast over, exactly as before.
        scope.spawn(move || {
            for item in raw_rx {
                let staged = match item {
                    Err(e) => Err(e),
                    Ok((idx, frame)) => {
                        if idx % every == 0 {
                            match detector.detect(&frame) {
                                Ok(d) => Ok(Staged {
                                    frame,
                                    detections: Some(d),
                                }),
                                Err(e) => Err(e.into()),
                            }
                        } else {
                            Ok(Staged {
                                frame,
                                detections: None,
                            })
                        }
                    }
                };
                let failed = staged.is_err();
                if det_tx.send(staged).is_err() || failed {
                    break;
                }
            }
        });

        // --- Stage 3: track, censor, encode (this thread) ----------------
        // Ordered and sequential: the tracker's state and the encoder's input
        // both depend on frames arriving in decode order.
        let mut tracker = Tracker::new(cfg.track);
        let mut total = 0usize;
        for item in det_rx {
            // Stopping between frames is what makes cancellation safe: the
            // caller discards the partial file, so a cancelled encode never
            // leaves a playable-looking truncated copy behind.
            if cfg.cancel.is_cancelled() {
                return Err(JobError::Cancelled);
            }
            let Staged {
                mut frame,
                detections,
            } = item?;
            let smoothed = tracker.update(detections.as_deref());
            let selected = cfg.profile.filter.select_owned(&smoothed);
            apply_censor(&mut frame, &selected, &cfg.profile.censor)?;
            total += selected.len();
            sink.put_frame(&frame)?;
        }
        Ok(total)
    })
}

/// Preview one file's detections/censoring without touching disk — used by the
/// GUI live preview and `obscura preview`.
pub fn preview(
    input: &Path,
    detector: &dyn Detector,
    profile: &Profile,
) -> Result<Frame, JobError> {
    preview_pair(input, detector, profile).map(|p| p.censored)
}

/// A preview's source frame and its censored counterpart.
#[derive(Debug, Clone)]
pub struct PreviewPair {
    /// The frame exactly as decoded, before any censoring.
    pub original: Frame,
    /// The same frame after the profile's filter and censor styles.
    pub censored: Frame,
    /// Regions the profile actually covered.
    pub regions: usize,
}

/// The expensive half of a preview: the decoded frame and everything the
/// detector had to say about it.
///
/// Split out from the cheap half because of how previews are actually used.
/// Decode plus inference costs hundreds of milliseconds to seconds; applying a
/// filter and painting boxes costs a few milliseconds. A GUI that re-previews
/// on every slider nudge must not re-run inference to answer "what does 0.12
/// padding look like" — nothing about the *detections* changed. Keep one of
/// these while the model and its settings hold still, and feed it to
/// [`preview_compose`] as often as the user moves something.
#[derive(Debug, Clone)]
pub struct PreviewSource {
    /// The frame exactly as decoded.
    pub frame: Frame,
    /// Every detection the model reported, before the profile's filter — the
    /// filter is part of what a preview is *for* varying, so it must not be
    /// baked in here.
    pub detections: Vec<Detection>,
    /// Set when the detector failed. Kept rather than returned as an error so
    /// [`preview_compose`] can replay `on_detect_failure` — the user can see
    /// what "blank the frame" does without the failure having to recur.
    pub detect_error: Option<String>,
}

/// Decode the first frame of `input` and run the detector over it.
pub fn preview_detect(input: &Path, detector: &dyn Detector) -> Result<PreviewSource, JobError> {
    // Deliberately the cheap extension-only classifier, not `classify_resolved`.
    // A preview only ever shows one frame, and `load_image` reads the first
    // frame of an animated GIF without needing ffmpeg installed — so previewing
    // a GIF keeps working on a machine that can only do stills.
    let frame = match classify(input) {
        MediaKind::Image => load_image(input)?,
        MediaKind::Video => {
            let mut source = FfmpegSource::open(input)?;
            source
                .next_frame()?
                .ok_or_else(|| ob_media::MediaError::Video("empty video".into()))?
        }
        MediaKind::Unknown => {
            return Err(JobError::Media(ob_media::MediaError::Video(
                "unsupported file type".into(),
            )))
        }
    };
    let (detections, detect_error) = match detector.detect(&frame) {
        Ok(d) => (d, None),
        Err(e) => (Vec::new(), Some(e.to_string())),
    };
    Ok(PreviewSource {
        frame,
        detections,
        detect_error,
    })
}

/// Apply a profile's filter and censor styles to an already-detected frame.
///
/// Cheap and pure — no I/O, no inference. This is what a live preview re-runs
/// while the user is dragging.
pub fn preview_compose(src: &PreviewSource, profile: &Profile) -> Result<PreviewPair, JobError> {
    let original = src.frame.clone();
    let mut censored = src.frame.clone();
    let regions = match &src.detect_error {
        None => {
            let selected = profile.filter.select_owned(&src.detections);
            apply_censor(&mut censored, &selected, &profile.censor)?;
            selected.len()
        }
        Some(e) => apply_detect_failure(&mut censored, profile.on_detect_failure, e)?,
    };
    Ok(PreviewPair {
        original,
        censored,
        regions,
    })
}

/// Decode the first frame of `input` and return it both before and after
/// censoring.
///
/// Two frames rather than one so the GUI can offer an A/B comparison: judging
/// whether the padding is large enough, or whether a region was missed
/// entirely, is guesswork without the original next to it. The source is
/// decoded once and cloned, so this costs one extra frame of memory and no
/// extra I/O or inference.
pub fn preview_pair(
    input: &Path,
    detector: &dyn Detector,
    profile: &Profile,
) -> Result<PreviewPair, JobError> {
    preview_compose(&preview_detect(input, detector)?, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::geometry::{BBox, Detection};
    use ob_core::taxonomy::cat;

    struct FakeDetector {
        dets: Vec<Detection>,
        fail: bool,
    }
    impl Detector for FakeDetector {
        fn detect(&self, _f: &Frame) -> Result<Vec<Detection>, ob_detect::DetectError> {
            if self.fail {
                Err(ob_detect::DetectError::Inference("boom".into()))
            } else {
                Ok(self.dets.clone())
            }
        }
    }

    fn solid_frame() -> Frame {
        Frame::new(8, 8, vec![200u8; 8 * 8 * 3]).unwrap()
    }

    /// Pixelating a flat field is a no-op (the cell average is the flat value),
    /// so censoring has to be proven on a non-uniform frame.
    fn checker_frame() -> Frame {
        let mut data = vec![0u8; 8 * 8 * 3];
        for y in 0..8usize {
            for x in 0..8usize {
                if (x + y) % 2 == 0 {
                    let p = (y * 8 + x) * 3;
                    data[p..p + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
        }
        Frame::new(8, 8, data).unwrap()
    }

    #[test]
    fn censor_frame_censors_selected_region() {
        let det = Detection {
            bbox: BBox::new(0.0, 0.0, 4.0, 4.0),
            category: cat::FEMALE_GENITALIA_EXPOSED,
            score: 0.9,
        };
        let d = FakeDetector {
            dets: vec![det],
            fail: false,
        };
        let mut frame = checker_frame();
        let before = frame.data.clone();
        let n = censor_frame(&mut frame, &d, &Profile::default()).unwrap();
        assert_eq!(n, 1);
        // The default style pixelates: the covered cell is now flat, and the
        // top-left pixel has moved off its original checker value.
        assert_ne!(frame.data[0], before[0]);
        assert_eq!(frame.data[0..3], frame.data[3..6]);
    }

    /// Write one PNG and return its path, for the preview tests.
    fn one_image(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ob-job-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.png");
        ob_media::save_image(&checker_frame(), &path).unwrap();
        path
    }

    fn genitalia_det() -> Detection {
        Detection {
            bbox: BBox::new(0.0, 0.0, 4.0, 4.0),
            category: cat::FEMALE_GENITALIA_EXPOSED,
            score: 0.9,
        }
    }

    #[test]
    fn composing_from_a_cached_source_matches_a_full_preview() {
        // The whole point of the split: the GUI re-composes without touching
        // the detector, so the two paths must agree pixel for pixel or the
        // live preview would be showing something the batch will not produce.
        let path = one_image("compose-matches");
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let profile = Profile::default();

        let full = preview_pair(&path, &d, &profile).unwrap();
        let src = preview_detect(&path, &d).unwrap();
        let composed = preview_compose(&src, &profile).unwrap();

        assert_eq!(full.regions, composed.regions);
        assert_eq!(full.censored.data, composed.censored.data);
        assert_eq!(full.original.data, composed.original.data);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn one_detect_serves_many_different_profiles() {
        // Dragging a censor-style control must not re-run inference, so the
        // same source has to answer for profiles it was not detected under.
        let path = one_image("compose-varies");
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let src = preview_detect(&path, &d).unwrap();

        let solid = Profile {
            censor: ob_core::censor::CensorConfig {
                default_style: ob_core::censor::CensorStyle::SolidFill {
                    color: [255, 0, 0, 255],
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let a = preview_compose(&src, &solid).unwrap();
        assert_eq!(a.regions, 1);
        assert_eq!(a.censored.data[0..3], [255, 0, 0]);

        // Deselecting the category censors nothing — from the same detections.
        let none = Profile {
            filter: ob_core::filter::FilterSet {
                rules: Vec::new(),
                ..Default::default()
            },
            ..Default::default()
        };
        let b = preview_compose(&src, &none).unwrap();
        assert_eq!(b.regions, 0);
        assert_eq!(b.censored.data, b.original.data);

        // And the source is untouched by either, so it stays reusable.
        assert_eq!(src.detections.len(), 1);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_detector_failure_is_carried_into_the_compose_stage() {
        // A failing detector must not abort the preview: the fail-closed policy
        // is itself something the user tunes, and they need to see what each
        // choice does without the failure having to happen again.
        let path = one_image("compose-failure");
        let d = FakeDetector {
            dets: vec![],
            fail: true,
        };
        let src = preview_detect(&path, &d).unwrap();
        assert!(src.detect_error.is_some());

        let blank = Profile {
            on_detect_failure: OnDetectFailure::Blank,
            ..Default::default()
        };
        let out = preview_compose(&src, &blank).unwrap();
        assert!(out.censored.data.iter().all(|&b| b == 0));
        // The original is still intact next to it.
        assert!(out.original.data.iter().any(|&b| b != 0));

        let through = Profile {
            on_detect_failure: OnDetectFailure::PassThrough,
            ..Default::default()
        };
        let out = preview_compose(&src, &through).unwrap();
        assert_eq!(out.censored.data, out.original.data);

        let skip = Profile {
            on_detect_failure: OnDetectFailure::Skip,
            ..Default::default()
        };
        assert!(matches!(
            preview_compose(&src, &skip),
            Err(JobError::FailClosedSkip(_))
        ));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn fail_closed_blank_zeroes_frame() {
        let d = FakeDetector {
            dets: vec![],
            fail: true,
        };
        let mut frame = solid_frame();
        let p = Profile {
            on_detect_failure: OnDetectFailure::Blank,
            ..Default::default()
        };
        censor_frame(&mut frame, &d, &p).unwrap();
        assert!(frame.data.iter().all(|&b| b == 0));
    }

    /// Write `n` tiny PNGs into a fresh temp directory and return it.
    fn dir_of_images(name: &str, n: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ob-job-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..n {
            ob_media::save_image(&checker_frame(), &dir.join(format!("{i}.png"))).unwrap();
        }
        dir
    }

    fn job_cfg<'a>(profile: &'a Profile, input: &Path, out: &Path) -> JobConfig<'a> {
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

    #[test]
    fn an_uncancelled_run_processes_every_file() {
        let dir = dir_of_images("normal", 4);
        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &dir, &out);
        let d = FakeDetector {
            dets: vec![],
            fail: false,
        };

        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.ok, 4);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.skipped, 0);
        assert!(!summary.cancelled);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_cancelled_up_front_writes_nothing_and_does_not_fail() {
        let dir = dir_of_images("cancelled", 4);
        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &dir, &out);
        cfg.cancel.cancel();
        let d = FakeDetector {
            dets: vec![],
            fail: false,
        };

        let summary = run(&cfg, &d, &|_| {}).unwrap();
        // Every item is skipped, and — the point — none is counted as failed,
        // so the CLI exits 0 and the GUI shows no error.
        assert_eq!(summary.ok, 0);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.skipped, 4);
        assert!(summary.cancelled);
        // Output files are never created for skipped items.
        assert!(!out.exists() || std::fs::read_dir(&out).unwrap().count() == 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelling_emits_a_cancelled_event_before_finished() {
        let dir = dir_of_images("events", 2);
        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &dir, &out);
        cfg.cancel.cancel();
        let d = FakeDetector {
            dets: vec![],
            fail: false,
        };

        let seen = std::sync::Mutex::new(Vec::new());
        run(&cfg, &d, &|ev| seen.lock().unwrap().push(format!("{ev:?}"))).unwrap();
        let seen = seen.into_inner().unwrap();

        let cancelled = seen.iter().position(|e| e.starts_with("Cancelled"));
        let finished = seen.iter().position(|e| e.starts_with("Finished"));
        assert!(cancelled.is_some(), "no Cancelled event in {seen:?}");
        // Ordering matters: a UI that renders "Done" on Finished must have
        // already seen the cancellation, or it reports a cancelled run as a
        // clean completion.
        assert!(
            cancelled < finished,
            "Cancelled must precede Finished: {seen:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Render a short test clip with ffmpeg. `None` if ffmpeg is unavailable.
    fn a_test_clip(name: &str) -> Option<PathBuf> {
        if !ob_media::tools::is_available(ob_media::tools::Tool::Ffmpeg) {
            return None;
        }
        let dir = std::env::temp_dir().join(format!("ob-job-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.mp4");
        let ok = ob_media::tools::command(ob_media::tools::Tool::Ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=2:size=64x48:rate=10")
            .args(["-pix_fmt", "yuv420p"])
            .arg(&path)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if ok {
            Some(path)
        } else {
            let _ = std::fs::remove_dir_all(&dir);
            None
        }
    }

    #[test]
    fn a_dry_run_over_a_video_actually_detects() {
        // The regression this guards: `--dry-run` used to open a video, confirm
        // it was readable and report 0 regions. Because dry-run output feeds
        // --report and the miss summary, every clip counted as "had none" and
        // pushed the reported miss rate down — a folder of videos looked like a
        // broken detector when nothing had been inferred at all.
        let Some(clip) = a_test_clip("dryrun-video") else {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        };
        let dir = clip.parent().unwrap().to_path_buf();
        let out = dir.join("out");
        let profile = Profile::default();
        let mut cfg = job_cfg(&profile, &clip, &out);
        cfg.dry_run = true;
        cfg.dry_run_frames = 4;
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };

        let seen = std::sync::Mutex::new(Vec::new());
        let summary = run(&cfg, &d, &|ev| {
            if let ProgressEvent::FileDone { regions, .. } = ev {
                seen.lock().unwrap().push(regions);
            }
        })
        .unwrap();

        assert_eq!(summary.ok, 1);
        let regions = seen.into_inner().unwrap();
        // One detection per sampled frame, four frames sampled.
        assert_eq!(regions, vec![4], "a dry run must report what it saw");
        // And it is still a dry run: nothing was written.
        assert!(
            !out.exists() || std::fs::read_dir(&out).unwrap().count() == 0,
            "dry run wrote output"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dry_run_never_writes_a_video() {
        // Guards the other half: the sampler must not be wired to the encoder.
        let Some(clip) = a_test_clip("dryrun-nowrite") else {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        };
        let dir = clip.parent().unwrap().to_path_buf();
        let out = dir.join("out");
        let profile = Profile::default();
        let mut cfg = job_cfg(&profile, &clip, &out);
        cfg.dry_run = true;
        let d = FakeDetector {
            dets: vec![],
            fail: false,
        };
        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.ok, 1);
        assert_eq!(summary.failed, 0);
        assert!(!out.join("clip.mp4").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_animated_gif_survives_as_an_animation() {
        // The regression: a GIF was classified as a still, so the batch decoded
        // frame 1, censored it, and wrote a one-frame GIF back — every other
        // frame silently discarded. It must round-trip as an animation.
        if !ob_media::tools::is_available(ob_media::tools::Tool::Ffmpeg) {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        }
        let dir = std::env::temp_dir().join("ob-job-test-gif-anim");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("a.gif");
        let ok = ob_media::tools::command(ob_media::tools::Tool::Ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=1:size=64x48:rate=8")
            .arg(&src)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("skipped: could not author a test gif");
            return;
        }
        // Precondition: the source really is an animation.
        assert_eq!(
            ob_media::classify_resolved(&src),
            ob_media::MediaKind::Video
        );

        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &src, &out);
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.failed, 0, "the gif failed to process");
        assert_eq!(summary.ok, 1);

        let written = out.join("a.gif");
        assert!(written.exists(), "no output gif written");
        // The point of the test: still more than one frame.
        assert_eq!(
            ob_media::classify_resolved(&written),
            ob_media::MediaKind::Video,
            "the output gif was flattened to a single frame"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_gif_keeps_every_frame_and_leaves_no_intermediate() {
        // GIF encoding cannot stream: `palettegen` only emits its palette at
        // end of input, so the single-pass graph held the entire clip in
        // memory and a long GIF dragged the machine into swap. Frames now go
        // to a lossless intermediate that two palette passes read back.
        //
        // Two things that fix can get wrong, both pinned here: dropping or
        // duplicating frames across the extra hop, and leaving the
        // intermediate -- the largest file the job touches -- behind.
        if !ob_media::tools::is_available(ob_media::tools::Tool::Ffmpeg) {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        }
        let dir = std::env::temp_dir().join("ob-job-test-gif-twopass");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("a.gif");
        let ok = ob_media::tools::command(ob_media::tools::Tool::Ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=2:size=64x48:rate=10")
            .arg(&src)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("skipped: could not author a test gif");
            return;
        }
        let before = ob_media::video::probe(&src).unwrap().frame_count;

        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &src, &out);
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.failed, 0, "the gif failed to process");

        let written = out.join("a.gif");
        let after = ob_media::video::probe(&written).unwrap().frame_count;
        // ffprobe does report nb_frames for GIF, so this comparison does run;
        // it is guarded so a build that declines to count frames skips the
        // check rather than failing on something that is not our bug. The
        // leftover check below is the unconditional one.
        if let (Some(b), Some(a)) = (before, after) {
            assert_eq!(a, b, "frame count changed passing through the intermediate");
        }

        // Nothing but the GIF itself should survive in the output directory.
        let leftovers: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "a.gif")
            .collect();
        assert!(
            leftovers.is_empty(),
            "the GIF intermediate was left behind: {leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_webm_round_trips_instead_of_breaking_the_pipe() {
        // The regression: every `.webm` failed with `Broken pipe (os error
        // 32)`. The output kept the input's extension, but the encoder was
        // always handed libx264, which the WebM muxer refuses — so ffmpeg died
        // writing the header and the first frame we wrote hit a dead pipe. The
        // codec has to follow the container.
        if !ob_media::tools::is_available(ob_media::tools::Tool::Ffmpeg) {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        }
        let dir = std::env::temp_dir().join("ob-job-test-webm");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("a.webm");
        // VP8 + Vorbis: what a WebM in the wild actually carries, and what the
        // audio-copy path has to stay legal for.
        let ok = ob_media::tools::command(ob_media::tools::Tool::Ffmpeg)
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=1:size=64x48:rate=10")
            .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=1"])
            .args(["-c:v", "libvpx", "-b:v", "200k", "-c:a", "libvorbis"])
            .arg(&src)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_dir_all(&dir);
            eprintln!("skipped: could not author a test webm");
            return;
        }

        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &src, &out);
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.failed, 0, "the webm failed to process");
        assert_eq!(summary.ok, 1);

        let written = out.join("a.webm");
        assert!(written.exists(), "no output webm written");
        let info = ob_media::video::probe(&written).unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        // The audio was copied, not dropped: Vorbis is legal in WebM, so the
        // stream should have come through untouched.
        assert!(info.has_audio, "the source audio was lost");
        assert_eq!(info.audio_codec.as_deref(), Some("vorbis"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_pipeline_queue_is_bounded_by_bytes_not_by_frames() {
        // A depth that is comfortable at 720p is most of a gigabyte at 2160p.
        // Sizing the queue by frame count is how the GIF encoder used to walk
        // the machine into swap, and the fix must not reintroduce it.
        let depth_720 = pipeline_depth(1280, 720);
        let depth_4k = pipeline_depth(3840, 2160);
        let depth_8k = pipeline_depth(7680, 4320);

        assert!(
            depth_720 >= depth_4k,
            "a smaller frame should buffer no less"
        );
        assert!(depth_4k >= depth_8k);
        // Always at least one, or the pipeline cannot make progress at all.
        for (w, h) in [(7680u32, 4320u32), (16000, 16000), (1, 1)] {
            assert!(pipeline_depth(w, h) >= 1, "{w}x{h} produced a zero depth");
        }
        // And never unbounded: two queues at this depth must stay inside the
        // budget the function documents.
        let bytes = |w: u32, h: u32| pipeline_depth(w, h) * 2 * (w as usize * h as usize * 3);
        for (w, h) in [(1280u32, 720u32), (1920, 1080), (3840, 2160)] {
            assert!(
                bytes(w, h) <= 128 * 1024 * 1024,
                "{w}x{h} would hold {} bytes in flight",
                bytes(w, h)
            );
        }
    }

    /// A detector that counts calls, and can cancel or fail partway through.
    struct ScriptedDetector {
        seen: std::sync::Arc<AtomicUsize>,
        /// Cancel this token once `trigger_after` frames have been seen.
        cancel: Option<CancelToken>,
        /// Return an error once `trigger_after` frames have been seen.
        fail: bool,
        trigger_after: usize,
    }

    impl Detector for ScriptedDetector {
        fn detect(&self, _f: &Frame) -> Result<Vec<Detection>, ob_detect::DetectError> {
            let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
            if n >= self.trigger_after {
                if let Some(c) = &self.cancel {
                    c.cancel();
                }
                if self.fail {
                    return Err(ob_detect::DetectError::Inference("scripted failure".into()));
                }
            }
            Ok(vec![genitalia_det()])
        }
    }

    #[test]
    fn a_video_keeps_every_frame_through_the_pipeline() {
        // Three stages and two queues between them is three places a frame can
        // be dropped or duplicated. The count has to survive the trip exactly.
        let Some(clip) = a_test_clip("pipeline-frames") else {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        };
        let dir = clip.parent().unwrap().to_path_buf();
        let before = ob_media::video::probe(&clip).unwrap().frame_count;

        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &clip, &out);
        let d = FakeDetector {
            dets: vec![genitalia_det()],
            fail: false,
        };
        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.failed, 0);

        let written = out.join("clip.mp4");
        assert!(written.exists(), "no output written");
        let after = ob_media::video::probe(&written).unwrap().frame_count;
        if let (Some(b), Some(a)) = (before, after) {
            assert_eq!(a, b, "frame count changed passing through the pipeline");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelling_mid_video_leaves_no_output_behind() {
        // The property the serial loop guaranteed and the pipeline must keep: a
        // cancelled encode never leaves a playable-looking truncated file that
        // could be mistaken for a finished censored copy. Now that the stages
        // run concurrently, the cancel is observed on a different thread from
        // the one decoding, which is exactly where this could regress.
        let Some(clip) = a_test_clip("pipeline-cancel") else {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        };
        let dir = clip.parent().unwrap().to_path_buf();
        let out = dir.join("out");
        let profile = Profile::default();
        let cfg = job_cfg(&profile, &clip, &out);
        let d = ScriptedDetector {
            seen: std::sync::Arc::new(AtomicUsize::new(0)),
            cancel: Some(cfg.cancel.clone()),
            fail: false,
            trigger_after: 3,
        };

        let summary = run(&cfg, &d, &|_| {}).unwrap();
        // A cancelled run is not a failed one.
        assert_eq!(summary.failed, 0, "cancellation reported as a failure");
        assert!(summary.cancelled);
        assert!(
            !out.join("clip.mp4").exists(),
            "a cancelled encode left a partial file behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_detector_failure_mid_video_removes_the_partial_output() {
        // Same guarantee on the error path. The failure now surfaces from the
        // detect stage through a channel rather than from a `?` in the loop, so
        // it is worth pinning that it still reaches the caller and still takes
        // the half-written file with it.
        let Some(clip) = a_test_clip("pipeline-detect-fail") else {
            eprintln!("skipped: ffmpeg unavailable");
            return;
        };
        let dir = clip.parent().unwrap().to_path_buf();
        let out = dir.join("out");
        let profile = Profile {
            // PassThrough would swallow it; this test is about the error path.
            on_detect_failure: OnDetectFailure::PassThrough,
            ..Default::default()
        };
        let cfg = job_cfg(&profile, &clip, &out);
        let d = ScriptedDetector {
            seen: std::sync::Arc::new(AtomicUsize::new(0)),
            cancel: None,
            fail: true,
            trigger_after: 3,
        };

        let summary = run(&cfg, &d, &|_| {}).unwrap();
        assert_eq!(summary.failed, 1, "the detector failure did not surface");
        assert!(
            !out.join("clip.mp4").exists(),
            "a failed encode left a partial file behind"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fail_closed_skip_errors() {
        let d = FakeDetector {
            dets: vec![],
            fail: true,
        };
        let mut frame = solid_frame();
        let p = Profile {
            on_detect_failure: OnDetectFailure::Skip,
            ..Default::default()
        };
        assert!(censor_frame(&mut frame, &d, &p).is_err());
    }
}
