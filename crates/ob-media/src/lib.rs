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
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"];
/// Recognized video extensions (lowercase, no dot).
pub const VIDEO_EXTS: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "m4v"];

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
}
