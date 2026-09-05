//! `cargo xtask` — build-time asset generation for Figura Obscura.
//!
//! Currently one task: `icons`, which renders the app icon set into
//! `packaging/assets/`. Those files are committed, so a normal build never
//! needs to run this; re-run it only when the mark itself changes.

mod icns;
mod icon;

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Sizes emitted as standalone PNGs. Covers Linux hicolor theme directories,
/// the itch.io page assets and the Windows/macOS source art.
const PNG_SIZES: &[u32] = &[16, 32, 48, 64, 128, 256, 512, 1024];

/// Sizes packed into the Windows `.ico`. 256 is the format's maximum.
const ICO_SIZES: &[u32] = &[16, 32, 48, 64, 128, 256];

/// The size the GUI embeds as a raw RGBA window icon.
const WINDOW_ICON_SIZE: u32 = 256;

fn main() -> Result<()> {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("icons") => generate_icons(&repo_root()?),
        Some(other) => bail!("unknown task `{other}` (known tasks: icons)"),
        None => {
            eprintln!("usage: cargo xtask <task>\n\ntasks:\n  icons   regenerate packaging/assets/ app icons");
            Ok(())
        }
    }
}

/// The workspace root, derived from this crate's manifest location rather than
/// the current directory, so `cargo xtask` works from any subdirectory.
fn repo_root() -> Result<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(Path::to_path_buf)
        .context("xtask has no parent directory")
}

fn generate_icons(root: &Path) -> Result<()> {
    let out = root.join("packaging").join("assets");
    std::fs::create_dir_all(&out)?;

    // Render each size independently rather than downscaling one master: the
    // shapes are analytic, so a 16px render places the bar on whole pixels
    // instead of inheriting a blurred 1024px edge.
    for &size in PNG_SIZES {
        let img = icon::render(size);
        let path = out.join(format!("icon-{size}.png"));
        img.save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        println!("  {}", path.display());
    }

    // --- Windows .ico ----------------------------------------------------
    let ico_path = out.join("figura-obscura.ico");
    write_ico(&ico_path, ICO_SIZES)?;
    println!("  {}", ico_path.display());

    // --- macOS .icns -----------------------------------------------------
    let icns_path = out.join("figura-obscura.icns");
    icns::write(&icns_path)?;
    println!("  {}", icns_path.display());

    // --- raw RGBA for the eframe window icon -----------------------------
    // Raw rather than PNG so the GUI needs no image decoder at startup just to
    // put a picture on its own title bar.
    let rgba_path = out.join(format!("window-icon-{WINDOW_ICON_SIZE}.rgba"));
    std::fs::write(&rgba_path, icon::render(WINDOW_ICON_SIZE).into_raw())?;
    println!("  {}", rgba_path.display());

    println!("\nicons written to {}", out.display());
    Ok(())
}

/// Write a multi-resolution Windows icon.
///
/// The `image` crate's ICO encoder stores each frame as PNG, which every
/// supported Windows version reads.
fn write_ico(path: &Path, sizes: &[u32]) -> Result<()> {
    use image::codecs::ico::{IcoEncoder, IcoFrame};

    let mut pngs: Vec<Vec<u8>> = Vec::new();
    for &size in sizes {
        let img = icon::render(size);
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png).write_image(
            img.as_raw(),
            size,
            size,
            image::ExtendedColorType::Rgba8,
        )?;
        pngs.push(png);
    }
    let frames: Vec<IcoFrame> = pngs
        .iter()
        .zip(sizes)
        .map(|(png, &size)| {
            IcoFrame::with_encoded(png, size, size, image::ExtendedColorType::Rgba8)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let file = std::fs::File::create(path)?;
    IcoEncoder::new(std::io::BufWriter::new(file)).encode_images(&frames)?;
    Ok(())
}

// `write_image` lives on this trait.
use image::ImageEncoder;
