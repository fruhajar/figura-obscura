//! # ob-media
//!
//! All pixel I/O for Obscura. Images convert between disk formats and [`Frame`] via
//! the `image` crate (complete here). Video demux/decode/encode with **audio
//! passthrough** goes through ffmpeg (invoked as a child process) and is scoped
//! in [`video`]. The [`FrameSource`]/[`FrameSink`] traits are the seam the
//! real-time screen tool reimplements (R10).

pub mod tools;
pub mod video;

use image::{ImageReader, RgbImage};
use ob_core::geometry::Frame;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    #[error("image decode failed for `{0}`: {1}")]
    Decode(String, String),
    #[error("image encode failed for `{0}`: {1}")]
    Encode(String, String),
    #[error(transparent)]
    Frame(#[from] ob_core::geometry::FrameError),
    #[error("video error: {0}")]
    Video(String),
}

/// A producer of frames (a decoded video, a directory of images, or later, a
/// screen capturer). `next_frame` returns `None` at end of stream.
pub trait FrameSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, MediaError>;
}

/// A consumer of censored frames (an encoder, an image writer, or later, a
/// screen overlay).
pub trait FrameSink {
    fn put_frame(&mut self, frame: &Frame) -> Result<(), MediaError>;
    /// Flush and finalize (e.g. close the encoder, mux audio).
    fn finish(self: Box<Self>) -> Result<(), MediaError>;
}

/// Decode an image file into an RGB8 [`Frame`].
pub fn load_image(path: &Path) -> Result<Frame, MediaError> {
    let img = ImageReader::open(path)
        .map_err(|e| MediaError::Decode(path.display().to_string(), e.to_string()))?
        .with_guessed_format()
        .map_err(|e| MediaError::Decode(path.display().to_string(), e.to_string()))?
        .decode()
        .map_err(|e| MediaError::Decode(path.display().to_string(), e.to_string()))?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    Frame::new(w, h, img.into_raw()).map_err(Into::into)
}

/// Encode an RGB8 [`Frame`] to an image file (format inferred from extension).
pub fn save_image(frame: &Frame, path: &Path) -> Result<(), MediaError> {
    let img = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
        .ok_or_else(|| MediaError::Encode(path.display().to_string(), "buffer size".into()))?;
    img.save(path)
        .map_err(|e| MediaError::Encode(path.display().to_string(), e.to_string()))
}

/// Recognized image extensions (lowercase, no dot).
pub const IMAGE_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif", "avif", "qoi", "tga", "ff",
];

/// Recognized video extensions (lowercase, no dot).
///
/// Deliberately broad. Every one of these is a container ffmpeg demuxes without
/// special handling, and the previous six-entry list was rejecting perfectly
/// ordinary files — a `.wmv` or a camcorder's `.mts` was classified `Unknown`
/// and skipped, which reads to the user as the tool being broken rather than as
/// the tool declining. The cost of a wrong guess here is one clear per-file
/// error; the cost of omitting a format is a file silently not processed.
pub const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm", "m4v", "flv", "wmv", "asf", "mpg", "mpeg", "mpe", "m2v",
    "ts", "m2ts", "mts", "vob", "ogv", "ogm", "3gp", "3g2", "f4v", "divx", "mxf", "rm", "rmvb",
    "dv", "y4m",
];

/// Classify a path by extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Unknown,
}

pub fn classify(path: &Path) -> MediaKind {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
    {
        Some(ext) if IMAGE_EXTS.contains(&ext.as_str()) => MediaKind::Image,
        Some(ext) if VIDEO_EXTS.contains(&ext.as_str()) => MediaKind::Video,
        _ => MediaKind::Unknown,
    }
}

/// True if `path` is a GIF holding more than one frame.
///
/// Only the first two frames are pulled, and `into_frames` is lazy, so this
/// reads a header and a little data rather than decoding the animation.
fn gif_is_animated(path: &Path) -> bool {
    use image::codecs::gif::GifDecoder;
    use image::AnimationDecoder;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(decoder) = GifDecoder::new(std::io::BufReader::new(file)) else {
        return false;
    };
    let mut frames = decoder.into_frames();
    frames.next();
    frames.next().is_some()
}

/// Classify `path`, reading the file in the one case the extension cannot
/// settle on its own.
///
/// `.gif` is two formats wearing one extension: a still image, and an animation
/// container. Treating every GIF as a still meant an animated one was censored
/// on frame 1 and written back with every other frame discarded — the animation
/// destroyed, and the user given no indication of it. So an animated GIF is a
/// video, and a single-frame GIF stays an image and keeps working with no
/// ffmpeg installed.
///
/// Everything else is decided by extension alone, so this stays cheap enough to
/// call while walking a directory.
pub fn classify_resolved(path: &Path) -> MediaKind {
    match classify(path) {
        MediaKind::Image if has_ext(path, "gif") && gif_is_animated(path) => MediaKind::Video,
        other => other,
    }
}

fn has_ext(path: &Path, want: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(want))
}

/// Ask ffprobe what `path` is, for files whose extension says nothing.
///
/// This costs a child process, so it is reserved for files the user named
/// explicitly — naming a file is a statement of intent, where walking a
/// directory is a search and must not spawn an ffprobe per `.txt`.
///
/// Returns `Unknown` when ffprobe is absent or the file has no video stream, so
/// a missing ffmpeg degrades to the old extension-only behaviour.
pub fn probe_kind(path: &Path) -> MediaKind {
    match video::probe(path) {
        // A single-frame "video" is a still in a container ffmpeg understands.
        Ok(info) if info.frame_count == Some(1) => MediaKind::Image,
        Ok(_) => MediaKind::Video,
        Err(_) => MediaKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classify_by_extension() {
        assert_eq!(classify(&PathBuf::from("a/b.PNG")), MediaKind::Image);
        assert_eq!(classify(&PathBuf::from("a/b.mkv")), MediaKind::Video);
        assert_eq!(classify(&PathBuf::from("a/b.txt")), MediaKind::Unknown);
    }

    #[test]
    fn ordinary_containers_are_recognised() {
        // Each of these was `Unknown` before, so the file was skipped without
        // ever being opened — which users read as the tool being broken.
        for ext in [
            "wmv", "flv", "mts", "m2ts", "mpg", "ts", "3gp", "vob", "ogv",
        ] {
            assert_eq!(
                classify(&PathBuf::from(format!("clip.{ext}"))),
                MediaKind::Video,
                ".{ext} should be recognised as video"
            );
        }
        // Case is irrelevant: cameras and Windows both like upper case.
        assert_eq!(classify(&PathBuf::from("CLIP.MTS")), MediaKind::Video);
    }

    fn gif_at(name: &str, frames: usize) -> PathBuf {
        use image::codecs::gif::GifEncoder;
        use image::{Frame as ImgFrame, RgbaImage};
        let dir = std::env::temp_dir().join(format!("ob-media-gif-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.gif");
        let img = RgbaImage::from_pixel(8, 8, image::Rgba([200, 30, 40, 255]));
        let mut file = std::fs::File::create(&path).unwrap();
        let mut enc = GifEncoder::new(&mut file);
        enc.encode_frames((0..frames).map(|_| ImgFrame::new(img.clone())))
            .unwrap();
        drop(enc);
        path
    }

    #[test]
    fn an_animated_gif_is_a_video_and_a_still_one_is_not() {
        // The bug this pins: every GIF was an Image, so an animated one was
        // censored on frame 1 and rewritten with all its other frames dropped.
        let animated = gif_at("animated", 3);
        assert_eq!(classify(&animated), MediaKind::Image, "extension alone");
        assert_eq!(
            classify_resolved(&animated),
            MediaKind::Video,
            "an animated GIF must take the video path"
        );

        // A one-frame GIF stays an image, so it still works with no ffmpeg.
        let still = gif_at("still", 1);
        assert_eq!(classify_resolved(&still), MediaKind::Image);

        let _ = std::fs::remove_dir_all(animated.parent().unwrap());
        let _ = std::fs::remove_dir_all(still.parent().unwrap());
    }

    #[test]
    fn resolving_a_non_gif_costs_nothing_and_changes_nothing() {
        // classify_resolved runs during directory walks, so it must not start
        // reading files that the extension already settled.
        for p in ["a/b.png", "a/b.mkv", "a/b.txt", "/nonexistent/x.jpg"] {
            let path = PathBuf::from(p);
            assert_eq!(classify_resolved(&path), classify(&path));
        }
    }
}
