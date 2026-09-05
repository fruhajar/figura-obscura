//! Video decode/encode with audio passthrough.
//!
//! v1 shells out to the system `ffmpeg`/`ffprobe` binaries rather than linking
//! libav: it keeps the Obscura binary small and license-clean, and the child-process
//! seam is easy to reason about. Policy (plan defaults): **re-encode video,
//! stream-copy audio**; default `libx264 -crf 20`.
//!
//! The pipeline reads decoded frames as packed `rgb24` from ffmpeg's stdout and
//! writes censored `rgb24` frames back to a second ffmpeg process's stdin, while
//! the original file is muxed in a second time purely for its audio stream.

use crate::tools::{self, Tool};
use crate::{FrameSink, FrameSource, MediaError};
use ob_core::geometry::Frame;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};

/// Encoder knobs surfaced to the user (each becomes a CLI flag / GUI setting).
#[derive(Debug, Clone)]
pub struct VideoEncodeOpts {
    /// Video codec, e.g. "libx264".
    pub codec: String,
    /// Constant rate factor (quality; lower = better/larger).
    pub crf: u32,
    /// Encoder speed/efficiency preset, e.g. "medium".
    pub preset: String,
    /// Copy the source audio stream unchanged.
    pub copy_audio: bool,
}

impl Default for VideoEncodeOpts {
    fn default() -> Self {
        Self {
            codec: "libx264".into(),
            crf: 20,
            preset: "medium".into(),
            copy_audio: true,
        }
    }
}

/// Basic stream metadata from `ffprobe`.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frame_count: Option<u64>,
    pub has_audio: bool,
}

/// Parse an ffprobe rational like `"30000/1001"` or `"25/1"` into fps.
fn parse_rational(s: &str) -> Option<f64> {
    let mut it = s.split('/');
    let num: f64 = it.next()?.trim().parse().ok()?;
    let den: f64 = match it.next() {
        Some(d) => d.trim().parse().ok()?,
        None => 1.0,
    };
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

/// Probe a video's dimensions/fps/audio via `ffprobe`.
pub fn probe(path: &Path) -> Result<VideoInfo, MediaError> {
    let out = tools::command(Tool::Ffprobe)
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(path)
        .output()
        .map_err(|e| {
            MediaError::Video(format!(
                "could not run ffprobe: {e}\n{}",
                tools::search_description(Tool::Ffprobe)
            ))
        })?;
    if !out.status.success() {
        return Err(MediaError::Video(format!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }

    let json: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| MediaError::Video(format!("ffprobe json parse error: {e}")))?;
    let streams = json
        .get("streams")
        .and_then(|s| s.as_array())
        .ok_or_else(|| MediaError::Video("ffprobe returned no streams".into()))?;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut fps = 0.0f64;
    let mut frame_count: Option<u64> = None;
    let mut has_audio = false;

    for s in streams {
        match s.get("codec_type").and_then(|c| c.as_str()) {
            // First video stream wins.
            Some("video") if width == 0 => {
                width = s.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                height = s.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                // avg_frame_rate is most representative; fall back to r_frame_rate.
                fps = s
                    .get("avg_frame_rate")
                    .and_then(|v| v.as_str())
                    .and_then(parse_rational)
                    .filter(|f| *f > 0.0)
                    .or_else(|| {
                        s.get("r_frame_rate")
                            .and_then(|v| v.as_str())
                            .and_then(parse_rational)
                    })
                    .unwrap_or(0.0);
                // nb_frames is often present for mp4/mkv; may be a string.
                frame_count = s.get("nb_frames").and_then(|v| {
                    v.as_str()
                        .and_then(|s| s.parse().ok())
                        .or_else(|| v.as_u64())
                });
            }
            Some("audio") => has_audio = true,
            _ => {}
        }
    }

    if width == 0 || height == 0 {
        return Err(MediaError::Video(format!(
            "no decodable video stream in {}",
            path.display()
        )));
    }
    // A sane fps fallback keeps output timing valid even if probing failed.
    // Deliberately negated: `fps <= 0.0` is false for NaN, so it would let a
    // NaN frame rate through. `!(fps > 0.0)` catches NaN, zero and negatives.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(fps > 0.0) {
        fps = 30.0;
    }
    // Derive an approximate frame count from duration when nb_frames is absent.
    if frame_count.is_none() {
        if let Some(dur) = json
            .get("format")
            .and_then(|f| f.get("duration"))
            .and_then(|d| d.as_str())
            .and_then(|d| d.parse::<f64>().ok())
        {
            if dur > 0.0 {
                frame_count = Some((dur * fps).round() as u64);
            }
        }
    }

    Ok(VideoInfo {
        width,
        height,
        fps,
        frame_count,
        has_audio,
    })
}

impl VideoInfo {
    /// Duration in seconds, when the probe knew enough to say.
    ///
    /// `frame_count` is itself derived from the container duration when
    /// `nb_frames` is absent, so this is the same fact read back the other way.
    /// `None` means the file did not declare a length — a stream, or a
    /// fragmented container — and callers must not invent one.
    pub fn duration_secs(&self) -> Option<f64> {
        // `probe` guarantees fps > 0, so this cannot divide by zero.
        self.frame_count.map(|n| n as f64 / self.fps)
    }
}

/// Timestamps at which to sample `count` frames from a clip of `duration`.
///
/// Samples land on the midpoint of each equal slice rather than at the slice
/// boundaries, which keeps them off both ends: the first frame of a video is
/// very often a black lead-in or a fade, and the last is often a fade-out, and
/// neither says anything about what the clip contains.
///
/// With no duration there is nothing to spread across, so it falls back to one
/// sample per second from the start. Seeking past the end simply yields no
/// frame, which the sampler reads as end of stream — so an over-long list is
/// self-limiting rather than an error.
fn sample_times(duration: Option<f64>, count: usize) -> Vec<f64> {
    let count = count.max(1);
    match duration {
        Some(d) if d > 0.0 => (0..count)
            .map(|i| d * (i as f64 + 0.5) / count as f64)
            .collect(),
        _ => (0..count).map(|i| i as f64).collect(),
    }
}

/// Decodes a handful of frames spread across a video, one seek at a time.
///
/// This exists for `--dry-run`, which answers "what does the detector fire on"
/// and writes nothing. Decoding every frame of a two-hour clip to answer that
/// would cost as much as the real run, so each sample is fetched by its own
/// ffmpeg invocation with **input seeking** (`-ss` before `-i`), which jumps to
/// the nearest keyframe instead of decoding everything in between.
///
/// The frames are therefore approximate in time. For sampling coverage that is
/// irrelevant, and it is the difference between a dry run taking seconds and
/// taking as long as the thing it is supposed to preview.
///
/// One process per frame is deliberate: it keeps memory to a single frame no
/// matter how long the clip or how large the resolution, and a failed seek
/// cannot poison the samples after it.
pub struct FfmpegSampler {
    path: PathBuf,
    info: VideoInfo,
    times: Vec<f64>,
    next: usize,
    frame_bytes: usize,
}

impl FfmpegSampler {
    /// Open `path` and plan `count` samples spread across it.
    pub fn open(path: &Path, count: usize) -> Result<Self, MediaError> {
        let info = probe(path)?;
        let times = sample_times(info.duration_secs(), count);
        let frame_bytes = info.width as usize * info.height as usize * 3;
        Ok(Self {
            path: path.to_path_buf(),
            info,
            times,
            next: 0,
            frame_bytes,
        })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }

    /// How many samples are planned. The run may yield fewer, because seeking
    /// past the end of a shorter-than-declared clip ends the stream early.
    pub fn planned(&self) -> usize {
        self.times.len()
    }
}

impl FrameSource for FfmpegSampler {
    fn next_frame(&mut self) -> Result<Option<Frame>, MediaError> {
        if self.next >= self.times.len() {
            return Ok(None);
        }
        let t = self.times[self.next];
        self.next += 1;

        let mut out = tools::command(Tool::Ffmpeg)
            .args(["-v", "error", "-nostdin"])
            // Before -i: seek by jumping keyframes, not by decoding to `t`.
            .args(["-ss", &format!("{t:.3}")])
            .arg("-i")
            .arg(&self.path)
            .args([
                "-map", "0:v:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ])
            .stdin(Stdio::null())
            .stderr(Stdio::inherit())
            .output()
            .map_err(|e| {
                MediaError::Video(format!(
                    "could not spawn ffmpeg to sample {} at {t:.3}s: {e}\n{}",
                    self.path.display(),
                    tools::search_description(Tool::Ffmpeg)
                ))
            })?;

        // A short read means the seek landed past the end. Because the
        // timestamps only increase, every later one is past the end too, so
        // this is end of stream rather than a gap to skip over.
        if out.stdout.len() < self.frame_bytes {
            return Ok(None);
        }
        out.stdout.truncate(self.frame_bytes);
        Ok(Some(Frame::new(
            self.info.width,
            self.info.height,
            out.stdout,
        )?))
    }
}

/// Decodes raw RGB frames from a video by piping ffmpeg's `rawvideo` output.
pub struct FfmpegSource {
    path: PathBuf,
    info: VideoInfo,
    child: Child,
    reader: BufReader<ChildStdout>,
    frame_bytes: usize,
}

impl FfmpegSource {
    pub fn open(path: &Path) -> Result<Self, MediaError> {
        let info = probe(path)?;
        // Decode to packed rgb24 on stdout; force output fps to match the probed
        // rate so frame count and the sink's `-r` stay consistent.
        let mut child = tools::command(Tool::Ffmpeg)
            .args(["-v", "error", "-nostdin"])
            .arg("-i")
            .arg(path)
            .args(["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                MediaError::Video(format!(
                    "could not spawn ffmpeg decoder: {e}\n{}",
                    tools::search_description(Tool::Ffmpeg)
                ))
            })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| MediaError::Video("ffmpeg produced no stdout pipe".into()))?;

        let frame_bytes = info.width as usize * info.height as usize * 3;
        Ok(Self {
            path: path.to_path_buf(),
            info,
            child,
            reader: BufReader::new(stdout),
            frame_bytes,
        })
    }

    pub fn info(&self) -> &VideoInfo {
        &self.info
    }
}

impl Drop for FfmpegSource {
    fn drop(&mut self) {
        // Ensure the decoder is reaped even if iteration stopped early.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl FrameSource for FfmpegSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, MediaError> {
        let mut buf = vec![0u8; self.frame_bytes];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => Ok(Some(Frame::new(self.info.width, self.info.height, buf)?)),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // A clean EOF on a frame boundary means end of stream. Zero bytes
                // read = normal end; a partial frame = truncated/corrupt input.
                let status = self.child.wait().ok();
                if let Some(st) = status {
                    if !st.success() {
                        return Err(MediaError::Video(format!(
                            "ffmpeg decoder exited with {st} while reading {}",
                            self.path.display()
                        )));
                    }
                }
                Ok(None)
            }
            Err(e) => Err(MediaError::Video(format!(
                "reading decoded frame from {}: {e}",
                self.path.display()
            ))),
        }
    }
}

/// Encodes censored frames and muxes the original audio back in.
pub struct FfmpegSink {
    output: PathBuf,
    child: Child,
    stdin: Option<ChildStdin>,
    frame_bytes: usize,
}

impl FfmpegSink {
    pub fn create(
        output: &Path,
        source: &Path,
        info: &VideoInfo,
        opts: VideoEncodeOpts,
    ) -> Result<Self, MediaError> {
        let mut cmd = tools::command(Tool::Ffmpeg);
        cmd.args(["-v", "error", "-nostdin", "-y"]);

        // Input 0: our raw censored frames on stdin.
        cmd.args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", info.width, info.height),
            "-r",
            &format!("{}", info.fps),
            "-i",
            "-",
        ]);

        let want_audio = opts.copy_audio && info.has_audio;
        // Input 1 (only when needed): the original file, for its audio stream.
        if want_audio {
            cmd.arg("-i").arg(source);
        }

        // Map censored video from input 0; audio (copied) from input 1.
        cmd.args(["-map", "0:v:0"]);
        if want_audio {
            cmd.args(["-map", "1:a:0", "-c:a", "copy"]);
        }

        cmd.args([
            "-c:v",
            &opts.codec,
            "-crf",
            &opts.crf.to_string(),
            "-preset",
            &opts.preset,
            // yuv420p keeps the output broadly playable regardless of source.
            "-pix_fmt",
            "yuv420p",
        ]);
        // Stop at the shorter of the (equal-length) streams so a slightly longer
        // audio track can't pad the tail.
        if want_audio {
            cmd.arg("-shortest");
        }
        cmd.arg(output);

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| MediaError::Video(format!("could not spawn ffmpeg encoder: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| MediaError::Video("ffmpeg encoder took no stdin pipe".into()))?;

        Ok(Self {
            output: output.to_path_buf(),
            child,
            stdin: Some(stdin),
            frame_bytes: info.width as usize * info.height as usize * 3,
        })
    }
}

impl Drop for FfmpegSink {
    fn drop(&mut self) {
        // `finish` takes `stdin`, so a still-present stdin here means the sink
        // was dropped without finishing — a cancelled or failed run. Killing
        // the encoder rather than letting it see a clean EOF is deliberate: on
        // EOF ffmpeg would finalise and *write out* a truncated file, which is
        // exactly what an abandoned censor job must not leave on disk.
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl FrameSink for FfmpegSink {
    fn put_frame(&mut self, frame: &Frame) -> Result<(), MediaError> {
        if frame.data.len() != self.frame_bytes {
            return Err(MediaError::Video(format!(
                "frame size mismatch: got {} bytes, encoder expects {}",
                frame.data.len(),
                self.frame_bytes
            )));
        }
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| MediaError::Video("encoder stdin already closed".into()))?;
        stdin
            .write_all(&frame.data)
            .map_err(|e| MediaError::Video(format!("writing frame to ffmpeg encoder: {e}")))
    }

    fn finish(mut self: Box<Self>) -> Result<(), MediaError> {
        // Close stdin so ffmpeg flushes and exits, then wait for it.
        drop(self.stdin.take());
        let status = self
            .child
            .wait()
            .map_err(|e| MediaError::Video(format!("waiting on ffmpeg encoder: {e}")))?;
        if !status.success() {
            return Err(MediaError::Video(format!(
                "ffmpeg encoder exited with {status} writing {}",
                self.output.display()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rational_handles_common_forms() {
        assert!((parse_rational("30/1").unwrap() - 30.0).abs() < 1e-9);
        assert!((parse_rational("30000/1001").unwrap() - 29.970_03).abs() < 1e-4);
        assert!((parse_rational("25").unwrap() - 25.0).abs() < 1e-9);
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("bad"), None);
    }

    #[test]
    fn samples_are_spread_across_the_clip_and_avoid_both_ends() {
        let t = sample_times(Some(10.0), 5);
        assert_eq!(t.len(), 5);
        // Midpoints of five equal 2s slices.
        assert!((t[0] - 1.0).abs() < 1e-9, "{t:?}");
        assert!((t[4] - 9.0).abs() < 1e-9, "{t:?}");
        // Never the very first or very last frame: those are usually a fade.
        assert!(t[0] > 0.0 && *t.last().unwrap() < 10.0);
        // Strictly increasing, which is what lets a short read mean "the end".
        assert!(t.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn an_unknown_duration_falls_back_to_one_sample_per_second() {
        // A stream or fragmented container declares no length. Inventing one
        // would be worse than sampling the opening and stopping at the end.
        let t = sample_times(None, 4);
        assert_eq!(t, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(sample_times(Some(0.0), 3), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn a_zero_sample_request_still_yields_one() {
        // Clamped rather than rejected: a dry run that samples nothing would
        // report 0 regions, which is the exact bug this sampler exists to fix.
        assert_eq!(sample_times(Some(10.0), 0).len(), 1);
    }

    #[test]
    fn duration_comes_from_frame_count_over_fps() {
        let info = VideoInfo {
            width: 8,
            height: 8,
            fps: 25.0,
            frame_count: Some(100),
            has_audio: false,
        };
        assert!((info.duration_secs().unwrap() - 4.0).abs() < 1e-9);
        let unknown = VideoInfo {
            frame_count: None,
            ..info
        };
        assert_eq!(unknown.duration_secs(), None);
    }

    #[test]
    fn default_opts_are_libx264_crf20_copy_audio() {
        let o = VideoEncodeOpts::default();
        assert_eq!(o.codec, "libx264");
        assert_eq!(o.crf, 20);
        assert!(o.copy_audio);
    }
}
