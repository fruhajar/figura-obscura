//! `cargo xtask bench` — time the real pipeline over a synthetic corpus.
//!
//! Micro-benchmarks tell you a function got faster; they do not tell you a run
//! got shorter. This drives the shipped `obscura` binary over generated media
//! and reports wall time, which is the number that decides whether a change was
//! worth making.
//!
//! The corpus is generated once and cached in `target/bench-corpus/`, because
//! the comparison is only meaningful if both sides see identical bytes. Delete
//! that directory to regenerate it.
//!
//! There are two halves. `pipeline` drives the binary end to end. `stages`
//! calls a handful of library functions directly, because the end-to-end run
//! cannot reach all of them: the detector finds nothing in procedurally
//! generated images, so the censor renderers never execute during a pipeline
//! run however the corpus is built. Timing them needs calling them.
//!
//! The stage half deliberately uses only functions that exist on **both** sides
//! of the optimisation work, so the same source file can be dropped into an
//! older checkout and produce comparable numbers. That constraint is why it
//! measures `letterbox_chw_with` rather than the batching entry point: batching
//! has no pre-existing counterpart to be compared against, and shows up in the
//! end-to-end tiling figure instead.
//!
//! This reports nothing about GPU execution providers. It measures whatever the
//! binary it is handed was built with — read the `--version` banner it prints
//! before believing any figure here is about a GPU.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Image sizes in the corpus, with how many of each.
///
/// The 4K entries are the case the tiling work exists for — they are what
/// actually tiles — and the 720p ones are there so the total is not dominated
/// by a handful of enormous files.
const IMAGES: &[(u32, u32, usize)] = &[(3840, 2160, 4), (1920, 1080, 6), (1280, 720, 10)];

/// One scenario to time.
struct Case {
    name: &'static str,
    /// Extra flags after the common ones.
    args: &'static [&'static str],
    /// Which corpus subdirectory to run over.
    input: &'static str,
}

const CASES: &[Case] = &[
    Case {
        name: "images, tiling auto (a 4K frame becomes a grid)",
        args: &["--set", "tiling=auto"],
        input: "images",
    },
    Case {
        name: "images, tiling off (one whole-frame pass each)",
        args: &["--set", "tiling=off"],
        input: "images",
    },
    Case {
        name: "video 1080p, detect every 3rd frame",
        args: &["--set", "tiling=off", "--detect-every", "3"],
        input: "video",
    },
];

pub fn run(root: &Path, binary: Option<String>) -> Result<()> {
    stages()?;
    println!();
    pipeline(root, binary)
}

/// How many times each stage is repeated. Enough that process noise averages
/// out at this granularity; not so many that the whole run stops being a thing
/// anyone will wait for.
const STAGE_ITERS: usize = 20;

/// Time the library functions the pipeline run cannot reach.
fn stages() -> Result<()> {
    use ob_core::censor::{CensorConfig, CensorStyle};
    use ob_core::geometry::{BBox, Detection};
    use ob_core::taxonomy::cat;
    use ob_detect::preprocess::Resampler;

    println!("stages (best of {STAGE_ITERS}, single-threaded caller)");

    let frame4k = frame(3840, 2160, 1);
    let frame1080 = frame(1920, 1080, 2);
    let tile = frame(320, 320, 3);

    let time = |name: &str, mut f: Box<dyn FnMut()>| {
        // Best-of rather than mean: the floor is the measurement, everything
        // above it is the machine doing something else at the time.
        let mut best = Duration::MAX;
        for _ in 0..STAGE_ITERS {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed());
        }
        println!("  {name:<44} {:>9.3} ms", best.as_secs_f64() * 1000.0);
    };

    time(
        "letterbox 4K -> 320, triangle",
        Box::new(move || {
            std::hint::black_box(ob_detect::preprocess::letterbox_chw_with(
                &frame4k,
                320,
                Resampler::Triangle,
            ));
        }),
    );
    let f = frame1080.clone();
    time(
        "letterbox 1080p -> 320, triangle",
        Box::new(move || {
            std::hint::black_box(ob_detect::preprocess::letterbox_chw_with(
                &f,
                320,
                Resampler::Triangle,
            ));
        }),
    );
    time(
        "letterbox 320 tile -> 320 (identity resize)",
        Box::new(move || {
            std::hint::black_box(ob_detect::preprocess::letterbox_chw_with(
                &tile,
                320,
                Resampler::Triangle,
            ));
        }),
    );

    // A region covering a good fraction of a 1080p frame: what censoring a
    // torso in a photograph actually costs.
    let region = [Detection {
        bbox: BBox::new(300.0, 150.0, 1500.0, 950.0),
        category: cat::FEMALE_BREAST_EXPOSED,
        score: 0.99,
    }];
    let styles: [(&str, CensorStyle, f32); 5] = [
        (
            "censor solid fill, 1200x800",
            CensorStyle::SolidFill {
                color: [0, 0, 0, 255],
            },
            0.0,
        ),
        (
            "censor pixelate block 16",
            CensorStyle::Pixelate { block: 16 },
            0.0,
        ),
        (
            "censor pixelate block 4",
            CensorStyle::Pixelate { block: 4 },
            0.0,
        ),
        (
            "censor blur sigma 12",
            CensorStyle::Blur { sigma: 12.0 },
            0.0,
        ),
        (
            "censor solid fill + rounding 0.4",
            CensorStyle::SolidFill {
                color: [0, 0, 0, 255],
            },
            0.4,
        ),
    ];
    for (name, style, rounding) in styles {
        let mut cfg = CensorConfig::default();
        cfg.shape.padding = 0.0;
        cfg.shape.rounding = rounding;
        cfg.default_style = style;
        let base = frame1080.clone();
        let dets = region;
        time(
            name,
            Box::new(move || {
                let mut f = base.clone();
                ob_censor::apply(&mut f, &dets, &cfg).unwrap();
                std::hint::black_box(&f);
            }),
        );
    }
    Ok(())
}

/// A textured frame as an `ob-core` `Frame`.
fn frame(w: u32, h: u32, seed: u64) -> ob_core::geometry::Frame {
    ob_core::geometry::Frame::new(w, h, textured(w, h, seed).into_raw()).unwrap()
}

/// Drive the shipped binary over the generated corpus.
fn pipeline(root: &Path, binary: Option<String>) -> Result<()> {
    let bin = match binary {
        Some(b) => PathBuf::from(b),
        None => root.join("target").join("release").join("obscura"),
    };
    if !bin.exists() {
        bail!(
            "no binary at {}\nbuild one first (cargo build --release -p ob-cli), \
             or pass a path: cargo xtask bench <path-to-obscura>",
            bin.display()
        );
    }

    let corpus = root.join("target").join("bench-corpus");
    ensure_corpus(&corpus)?;

    println!("pipeline");
    println!("  binary: {}", bin.display());
    let version = Command::new(&bin).arg("--version").output();
    if let Ok(v) = version {
        for line in String::from_utf8_lossy(&v.stdout).lines().take(4) {
            println!("    {line}");
        }
    }
    println!();

    let out = root.join("target").join("bench-out");
    for case in CASES {
        let input = corpus.join(case.input);
        if !input.exists() {
            println!("  {:<46} skipped (no corpus)", case.name);
            continue;
        }
        let _ = std::fs::remove_dir_all(&out);

        let start = Instant::now();
        let status = Command::new(&bin)
            .arg("process")
            .arg(&input)
            .arg("-o")
            .arg(&out)
            // Never reach for the network mid-measurement: a surprise model
            // download in the middle of a timed run would be recorded as
            // pipeline time.
            .arg("--no-auto-fetch")
            .args(case.args)
            // The per-run detection summary is useful output for a user and
            // noise in a results table.
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .with_context(|| format!("running {}", bin.display()))?;
        let elapsed = start.elapsed();

        if status.success() {
            println!("  {:<46} {}", case.name, fmt(elapsed));
        } else {
            println!("  {:<46} FAILED ({status})", case.name);
        }
    }
    let _ = std::fs::remove_dir_all(&out);
    Ok(())
}

fn fmt(d: Duration) -> String {
    format!("{:>8.2} s", d.as_secs_f64())
}

/// Generate the corpus if it is not already there.
fn ensure_corpus(dir: &Path) -> Result<()> {
    let images = dir.join("images");
    if !images.exists() {
        std::fs::create_dir_all(&images)?;
        println!("generating image corpus in {} ...", images.display());
        for &(w, h, n) in IMAGES {
            for i in 0..n {
                let path = images.join(format!("{w}x{h}-{i}.png"));
                textured(w, h, i as u64).save(&path)?;
            }
        }
    }

    let video = dir.join("video");
    if !video.exists() {
        std::fs::create_dir_all(&video)?;
        println!("generating video corpus in {} ...", video.display());
        // Authored with ffmpeg rather than encoded from our own frames: the
        // point is a file the decode path reads the way a real one would.
        let clip = video.join("clip.mp4");
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=10:size=1920x1080:rate=25")
            .args([
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast",
            ])
            .arg(&clip)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_dir_all(&video);
            println!("  (ffmpeg unavailable — the video case will be skipped)");
        }
    }
    Ok(())
}

/// A frame with structure at several scales.
///
/// Content matters more than it looks like it should: a flat field costs the
/// resampler the same taps but gives the encoder almost nothing to do, so a
/// corpus of grey squares would quietly measure a different pipeline.
fn textured(w: u32, h: u32, seed: u64) -> image::RgbImage {
    let mut img = image::RgbImage::new(w, h);
    let s = seed.wrapping_mul(2654435761) as u32;
    for (x, y, px) in img.enumerate_pixels_mut() {
        let a = ((x.wrapping_add(s)).wrapping_mul(7) ^ y.wrapping_mul(13)) as u8;
        let b = ((x / 3).wrapping_add(y / 5) % 251) as u8;
        let c = if ((x / 2 + y / 2 + s) % 3) == 0 {
            240
        } else {
            12
        };
        *px = image::Rgb([a, b, c]);
    }
    img
}
