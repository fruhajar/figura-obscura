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
use rayon::prelude::*;
use ob_censor::apply as apply_censor;
use ob_core::cancel::CancelToken;
use ob_core::geometry::{Detection, Frame};
use ob_core::profile::{OnDetectFailure, Profile};
use ob_detect::Detector;
use ob_media::video::{FfmpegSink, FfmpegSource, VideoEncodeOpts};
use ob_media::{classify, load_image, save_image, FrameSink, FrameSource, MediaKind};
use ob_track::{TrackConfig, Tracker};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything a run needs besides the detector (which the caller builds from the
/// chosen model + `ob-models` cache path).
pub struct JobConfig<'a> {
    pub profile: &'a Profile,
    pub input: InputSpec,
    pub output_dir: PathBuf,
    /// Log what would happen without writing any output.
    pub dry_run: bool,
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
            let selected: Vec<_> = profile
                .filter
                .select_all(&dets)
                .into_iter()
                .copied()
                .collect();
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
            for b in frame.data.iter_mut() {
                *b = 0;
            }
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
        save_image(&frame, output)?;
    }
    Ok(regions)
}

fn process_video(
    input: &Path,
    output: &Path,
    cfg: &JobConfig,
    detector: &dyn Detector,
) -> Result<usize, JobError> {
    if cfg.dry_run {
        // Nothing written; just confirm the file is openable.
        let _ = FfmpegSource::open(input)?;
        return Ok(0);
    }
    if let Some(parent) = output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut source = FfmpegSource::open(input)?;
    let info = source.info().clone();
    let mut sink = Box::new(FfmpegSink::create(
        output,
        input,
        &info,
        cfg.video_opts.clone(),
    )?);
    let mut tracker = Tracker::new(cfg.track);

    let mut total = 0usize;
    let mut idx: u32 = 0;
    let every = cfg.detect_every.max(1);
    while let Some(mut frame) = source.next_frame()? {
        // A single long video would otherwise ignore cancellation entirely.
        // The partially written output is discarded by the caller's error
        // path, so a cancelled encode never leaves a playable-looking truncated
        // file that might be mistaken for a finished censored copy.
        if cfg.cancel.is_cancelled() {
            // Dropping the sink kills the encoder instead of letting it
            // finalise (see `FfmpegSink::drop`), then the partial file goes.
            drop(sink);
            let _ = std::fs::remove_file(output);
            return Err(JobError::Cancelled);
        }
        // Detect on every Nth frame; coast the tracker on the others.
        let raw = if idx % every == 0 {
            Some(detector.detect(&frame)?)
        } else {
            None
        };
        let smoothed = tracker.update(raw.as_deref());
        let selected: Vec<_> = cfg
            .profile
            .filter
            .select_all(&smoothed)
            .into_iter()
            .copied()
            .collect();
        apply_censor(&mut frame, &selected, &cfg.profile.censor)?;
        total += selected.len();
        sink.put_frame(&frame)?;
        idx += 1;
    }
    sink.finish()?;
    Ok(total)
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
            let selected: Vec<_> = profile
                .filter
                .select_all(&src.detections)
                .into_iter()
                .copied()
                .collect();
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
