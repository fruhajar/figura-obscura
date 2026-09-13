//! Video decode/encode with audio passthrough.
//!
//! v1 shells out to the system `ffmpeg`/`ffprobe` binaries rather than linking
//! libav: it keeps the Obscura binary small and license-clean, and the child-process
//! seam is easy to reason about. Policy (plan defaults): **re-encode video,
//! stream-copy audio**; default `libx264 -crf 20`.
//!
//! The codec is not free to choose, though — the *output container* constrains
//! it, and the output keeps the input's extension. A WebM file takes only
//! VP8/VP9/AV1 and Vorbis/Opus, so h.264 into one is rejected by the muxer
//! before a single frame is read. [`Container`] is where that is decided.
//!
//! The pipeline reads decoded frames as packed `rgb24` from ffmpeg's stdout and
//! writes censored `rgb24` frames back to a second ffmpeg process's stdin, while
//! the original file is muxed in a second time purely for its audio stream.

use crate::tools::{self, Tool};
use crate::{FrameSink, FrameSource, MediaError};
use ob_core::geometry::Frame;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Stdio};
use std::sync::{Arc, Mutex};

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
    /// ffprobe's `codec_name` for the first audio stream, e.g. `"vorbis"`.
    ///
    /// Needed because "copy the audio" is not always legal: a container only
    /// accepts certain audio codecs, so the sink has to know what it is being
    /// asked to copy before it decides whether it can.
    pub audio_codec: Option<String>,
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
    let mut audio_codec: Option<String> = None;

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
            Some("audio") => {
                // First audio stream wins, matching the `0:a:0` the sink maps.
                if !has_audio {
                    audio_codec = s
                        .get("codec_name")
                        .and_then(|c| c.as_str())
                        .map(|c| c.to_ascii_lowercase());
                }
                has_audio = true;
            }
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
        audio_codec,
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

/// How much of a child's stderr to keep, in bytes.
///
/// ffmpeg's complaint is the last thing it writes before exiting, so the tail
/// is the useful end. A few kilobytes holds the whole failure message of even a
/// chatty invocation without letting a warning-per-frame run grow unbounded.
const STDERR_KEEP: usize = 8 * 1024;

/// The tail of a child process's stderr, collected on a thread.
///
/// ffmpeg says *why* it failed on stderr and then exits; everything we notice
/// afterwards is a secondary symptom. Letting it inherit our stderr put that
/// sentence wherever the process happened to be launched from — for a GUI
/// started from a desktop entry, nowhere at all — and left the user with
/// `Broken pipe (os error 32)`, which names no cause.
///
/// It has to be drained *while* the child runs rather than read after waiting
/// on it. A pipe nobody reads fills at 64 KiB and blocks the writer, and an
/// ffmpeg blocked writing stderr never reads its stdin again: the encode would
/// hang instead of failing. Hence the thread, which ends by itself at EOF.
struct StderrTail {
    buf: Arc<Mutex<String>>,
}

impl StderrTail {
    fn spawn(stderr: ChildStderr) -> Self {
        let buf = Arc::new(Mutex::new(String::new()));
        let sink = Arc::clone(&buf);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            // Line at a time, decoded lossily: ffmpeg can emit a file name in
            // whatever encoding the filesystem used, and a mangled path in a
            // diagnostic beats dropping the diagnostic.
            while matches!(reader.read_until(b'\n', &mut line), Ok(n) if n > 0) {
                let mut b = sink.lock().unwrap_or_else(|e| e.into_inner());
                b.push_str(&String::from_utf8_lossy(&line));
                if b.len() > STDERR_KEEP {
                    let cut = b.len() - STDERR_KEEP;
                    let cut = (cut..=b.len())
                        .find(|i| b.is_char_boundary(*i))
                        .unwrap_or(b.len());
                    b.replace_range(..cut, "");
                }
                line.clear();
            }
        });
        Self { buf }
    }

    /// What the child said, as a single line, or `None` if it said nothing.
    ///
    /// Flattened to one line because this ends up inside a per-file error in a
    /// progress log and a GUI list, neither of which has room for a paragraph.
    fn message(&self) -> Option<String> {
        let buf = self.buf.lock().unwrap_or_else(|e| e.into_inner());
        let joined = buf
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        (!joined.is_empty()).then_some(joined)
    }
}

/// The output container, which decides what may legally go inside it.
///
/// This exists because of a failure that looked like a bug in our own pipe
/// handling. ffmpeg will not write h.264 into a WebM file: the muxer rejects it
/// while writing the header, before reading a single frame, so the encoder was
/// already dead when the first `put_frame` ran and every `.webm` came back as
/// `Broken pipe (os error 32)`. The codec is not a free choice — the container
/// constrains it — so the container picks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    /// Animated GIF: a different encoder and a palette pass, not a codec choice.
    Gif,
    /// WebM: VP8/VP9/AV1 video and Vorbis/Opus audio, and nothing else.
    WebM,
    /// mp4/mkv/mov/avi and friends — everything that takes h.264 happily.
    ///
    /// Also the landing place for the handful of exotic containers in
    /// `VIDEO_EXTS` that take neither h.264 nor VP9 (`.ogv`, `.dv`, `.vob`).
    /// Guessing an encoder for each of those is not worth it while nobody has
    /// asked; what matters is that they now fail saying which codec the muxer
    /// refused, instead of saying `Broken pipe`.
    H264,
}

impl Container {
    fn of(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("gif") => Container::Gif,
            Some("webm") => Container::WebM,
            _ => Container::H264,
        }
    }
}

/// Audio codecs a WebM file is allowed to carry.
const WEBM_AUDIO: &[&str] = &["opus", "vorbis"];

/// Re-scale the x264-shaped `crf` knob onto VP9's range.
///
/// Both scales run "lower is better" and both start at 0, but x264 stops at 51
/// and VP9 at 63, so the same number is not the same picture. Passing the
/// default 20 through unchanged would ask VP9 for something markedly sharper —
/// and larger — than the user picked on the scale the knob is documented in, so
/// the position on the scale is preserved rather than the number: 20 becomes
/// 25, and the far end, 51, becomes 63.
fn vp9_crf(crf: u32) -> u32 {
    ((crf.min(51) as f64) * 63.0 / 51.0).round() as u32
}

/// Map an x264 preset name onto libvpx's `-cpu-used` speed knob.
///
/// libvpx has no `-preset`; its equivalent is a 0–5 speed/quality dial where
/// *higher* is faster. Without this the preset would silently do nothing on
/// WebM output and every VP9 encode would run at libvpx's own default speed,
/// which is slow enough on a batch of clips to look like a hang.
fn vpx_cpu_used(preset: &str) -> &'static str {
    match preset {
        "ultrafast" | "superfast" | "veryfast" => "5",
        "faster" | "fast" => "3",
        "slow" | "slower" | "veryslow" | "placebo" => "1",
        // "medium" and anything unrecognised: near real-time at 720p, which is
        // the right default for a tool that encodes whole folders.
        _ => "2",
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
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
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
    stderr: StderrTail,
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
            .stderr(Stdio::piped())
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
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| MediaError::Video("ffmpeg produced no stderr pipe".into()))?;

        let frame_bytes = info.width as usize * info.height as usize * 3;
        Ok(Self {
            path: path.to_path_buf(),
            info,
            child,
            reader: BufReader::new(stdout),
            stderr: StderrTail::spawn(stderr),
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
                            "ffmpeg decoder exited with {st} while reading {}{}",
                            self.path.display(),
                            detail(self.stderr.message())
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
    stderr: StderrTail,
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

        let container = Container::of(output);

        // GIF is not a codec choice, it is a different encoder entirely: libx264
        // cannot write one, and a naive `-f gif` produces a 256-colour mess
        // because ffmpeg falls back to a fixed palette. palettegen/paletteuse
        // over a `split` builds a palette from this clip's own colours in a
        // single pass, which matters here — censor boxes are flat blocks of one
        // colour and a generic palette bands them visibly.
        if container == Container::Gif {
            cmd.args([
                "-filter_complex",
                "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
                // 0 = loop forever, matching how animated GIFs are normally authored.
                "-loop",
                "0",
                "-f",
                "gif",
            ]);
            cmd.arg(output);
            return Self::spawn(cmd, output, info);
        }

        let want_audio = opts.copy_audio && info.has_audio;
        // Input 1 (only when needed): the original file, for its audio stream.
        if want_audio {
            cmd.arg("-i").arg(source);
        }

        // Map censored video from input 0; audio from input 1.
        cmd.args(["-map", "0:v:0"]);
        if want_audio {
            cmd.args(["-map", "1:a:0"]);
            cmd.args(["-c:a", audio_codec_for(container, info)]);
        }

        match container {
            // libvpx needs `-b:v 0` for `-crf` to mean constant quality at all:
            // without it the CRF is only a *ceiling* on quality and the encode
            // targets libvpx's 256 kbit/s default bitrate, which at any real
            // resolution is a smear. `-row-mt` puts the encode on every core;
            // together with `-cpu-used` it is the difference between a clip
            // encoding at roughly real time and slowly enough to look hung.
            Container::WebM => cmd.args([
                "-c:v",
                "libvpx-vp9",
                "-crf",
                &vp9_crf(opts.crf).to_string(),
                "-b:v",
                "0",
                "-row-mt",
                "1",
                "-deadline",
                "good",
                "-cpu-used",
                vpx_cpu_used(&opts.preset),
                "-pix_fmt",
                "yuv420p",
            ]),
            _ => cmd.args([
                "-c:v",
                &opts.codec,
                "-crf",
                &opts.crf.to_string(),
                "-preset",
                &opts.preset,
                // yuv420p keeps the output broadly playable regardless of source.
                "-pix_fmt",
                "yuv420p",
            ]),
            // Gif returned above.
        };
        // Stop at the shorter of the (equal-length) streams so a slightly longer
        // audio track can't pad the tail.
        if want_audio {
            cmd.arg("-shortest");
        }
        cmd.arg(output);

        Self::spawn(cmd, output, info)
    }

    /// Start the configured encoder and take its stdin. Shared by the GIF and
    /// the ordinary video paths so both get identical process handling.
    fn spawn(
        mut cmd: std::process::Command,
        output: &Path,
        info: &VideoInfo,
    ) -> Result<Self, MediaError> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| MediaError::Video(format!("could not spawn ffmpeg encoder: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| MediaError::Video("ffmpeg encoder took no stdin pipe".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| MediaError::Video("ffmpeg encoder took no stderr pipe".into()))?;

        Ok(Self {
            output: output.to_path_buf(),
            child,
            stdin: Some(stdin),
            stderr: StderrTail::spawn(stderr),
            frame_bytes: info.width as usize * info.height as usize * 3,
        })
    }
}

impl FfmpegSink {
    /// Turn a failed write to the encoder into an error that says why.
    ///
    /// A write to the encoder essentially only fails one way: ffmpeg is already
    /// gone, and the pipe with it. `Broken pipe (os error 32)` is the symptom
    /// of that, never the cause — the cause is on the child's stderr and in its
    /// exit status, both of which are available right here. Reporting the EPIPE
    /// alone sent users looking at our pipe handling for a fault that was
    /// always in the arguments we had handed ffmpeg.
    fn write_failed(&mut self, e: std::io::Error) -> MediaError {
        // Kill rather than close stdin, for the reason `Drop` gives: an ffmpeg
        // that is still alive would take EOF as "that was the whole video" and
        // finalise a truncated file, which is the one thing a half-censored run
        // must never leave behind. Killing also guarantees `wait` returns
        // instead of blocking on a child that is still waiting to be fed.
        //
        // It costs nothing in the usual case: ffmpeg has already exited and is
        // a zombie holding its real exit status, and a signal to a zombie does
        // nothing — so the status below is still the one ffmpeg chose.
        let _ = self.child.kill();
        drop(self.stdin.take());
        let status = match self.child.wait() {
            Ok(st) if !st.success() => format!(" (ffmpeg exited with {st})"),
            // Exited cleanly, or unwaitable: the io error is all we have.
            _ => format!(" ({e})"),
        };
        MediaError::Video(format!(
            "ffmpeg encoder failed writing {}{status}{}",
            self.output.display(),
            detail(self.stderr.message())
        ))
    }
}

/// Append a child's own words to an error message, when it said anything.
fn detail(msg: Option<String>) -> String {
    msg.map(|m| format!(": {m}")).unwrap_or_default()
}

/// How to carry the source's audio into `container`.
///
/// The policy is still "keep the user's audio, don't re-encode it" — `copy`
/// wherever the container will take the stream as it stands. WebM is the case
/// where it won't: it accepts only Vorbis and Opus, so an AAC or MP3 track has
/// to be converted rather than copied. Converting is the least-bad of the three
/// options, because the other two are dropping the audio silently and failing
/// the file outright, and neither is what "copy the audio" was asking for.
fn audio_codec_for(container: Container, info: &VideoInfo) -> &'static str {
    let copyable = match container {
        Container::WebM => info
            .audio_codec
            .as_deref()
            .is_some_and(|c| WEBM_AUDIO.contains(&c)),
        _ => true,
    };
    match (container, copyable) {
        (_, true) => "copy",
        (Container::WebM, false) => "libopus",
        // Unreachable: every other container reports `copyable`.
        (_, false) => "copy",
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
        if let Err(e) = stdin.write_all(&frame.data) {
            return Err(self.write_failed(e));
        }
        Ok(())
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
                "ffmpeg encoder exited with {status} writing {}{}",
                self.output.display(),
                detail(self.stderr.message())
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
            audio_codec: None,
        };
        assert!((info.duration_secs().unwrap() - 4.0).abs() < 1e-9);
        let unknown = VideoInfo {
            frame_count: None,
            ..info
        };
        assert_eq!(unknown.duration_secs(), None);
    }

    fn info_with_audio(codec: Option<&str>) -> VideoInfo {
        VideoInfo {
            width: 8,
            height: 8,
            fps: 25.0,
            frame_count: Some(100),
            has_audio: codec.is_some(),
            audio_codec: codec.map(str::to_string),
        }
    }

    #[test]
    fn the_container_picks_the_codec_not_the_other_way_round() {
        // The whole bug in one assertion: a `.webm` output used to be handed
        // libx264, which the WebM muxer rejects while writing the header — so
        // the encoder died before the first frame and every write came back
        // `Broken pipe (os error 32)`.
        assert_eq!(Container::of(Path::new("clip.webm")), Container::WebM);
        assert_eq!(Container::of(Path::new("CLIP.WEBM")), Container::WebM);
        assert_eq!(Container::of(Path::new("clip.gif")), Container::Gif);
        assert_eq!(Container::of(Path::new("clip.mp4")), Container::H264);
        assert_eq!(Container::of(Path::new("clip.mkv")), Container::H264);
        // No extension at all is not a reason to refuse: ffmpeg guesses from
        // the name, and h.264 is the right guess to hand it.
        assert_eq!(Container::of(Path::new("clip")), Container::H264);
    }

    #[test]
    fn webm_copies_audio_it_can_hold_and_converts_audio_it_cannot() {
        // Vorbis and Opus go through untouched — this is the ordinary case,
        // since output keeps the input's extension and WebM sources carry one
        // of the two.
        for ok in ["vorbis", "opus"] {
            assert_eq!(
                audio_codec_for(Container::WebM, &info_with_audio(Some(ok))),
                "copy"
            );
        }
        // AAC in a WebM file is not legal, so copying it is the same header
        // failure by another route.
        assert_eq!(
            audio_codec_for(Container::WebM, &info_with_audio(Some("aac"))),
            "libopus"
        );
        // An unprobeable codec is treated as not copyable: converting costs
        // quality, guessing wrong costs the whole file.
        assert_eq!(
            audio_codec_for(Container::WebM, &info_with_audio(None)),
            "libopus"
        );
        // Everything else still copies, which is the documented policy.
        assert_eq!(
            audio_codec_for(Container::H264, &info_with_audio(Some("aac"))),
            "copy"
        );
    }

    #[test]
    fn the_quality_knob_keeps_its_position_on_vp9s_longer_scale() {
        // Same fraction of the scale, not the same number: VP9 runs to 63.
        assert_eq!(vp9_crf(0), 0);
        assert_eq!(vp9_crf(20), 25);
        assert_eq!(vp9_crf(51), 63);
        // Out-of-range input is clamped rather than producing an illegal CRF
        // that ffmpeg would reject at spawn time.
        assert_eq!(vp9_crf(99), 63);
    }

    #[test]
    fn presets_map_onto_libvpxs_inverted_speed_dial() {
        // Higher cpu-used is *faster*, so the mapping inverts the x264 sense.
        assert_eq!(vpx_cpu_used("veryfast"), "5");
        assert_eq!(vpx_cpu_used("medium"), "2");
        assert_eq!(vpx_cpu_used("veryslow"), "1");
        // An unknown preset must not leave libvpx on its own default speed,
        // which is slow enough over a folder of clips to look like a hang.
        assert_eq!(vpx_cpu_used("nonsense"), "2");
    }

    #[test]
    fn default_opts_are_libx264_crf20_copy_audio() {
        let o = VideoEncodeOpts::default();
        assert_eq!(o.codec, "libx264");
        assert_eq!(o.crf, 20);
        assert!(o.copy_audio);
    }
}
