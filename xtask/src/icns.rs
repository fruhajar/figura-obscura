//! Minimal writer for the macOS `.icns` icon container.
//!
//! Written by hand because the build container has no `iconutil` (that is a
//! macOS tool) and no `png2icns`. The format is simple enough not to warrant a
//! dependency: a magic word, a total length, then a flat sequence of
//! `[OSType][u32 length][payload]` entries, all big-endian. Modern macOS reads
//! PNG payloads directly, so every entry here is just a PNG.

use crate::icon;
use anyhow::{Context, Result};
use image::ImageEncoder;
use std::path::Path;

/// The icon types Finder, the Dock and Launchpad actually consult, paired with
/// the pixel dimensions each one must contain. `ic11`–`ic14` are the Retina
/// (`@2x`) variants; omitting them makes the icon look soft on every Mac sold
/// in the last decade.
const ENTRIES: &[(&[u8; 4], u32)] = &[
    (b"ic11", 32),   // 16x16@2x
    (b"ic12", 64),   // 32x32@2x
    (b"ic07", 128),  // 128x128
    (b"ic13", 256),  // 128x128@2x
    (b"ic08", 256),  // 256x256
    (b"ic14", 512),  // 256x256@2x
    (b"ic09", 512),  // 512x512
    (b"ic10", 1024), // 512x512@2x
];

/// Render and write the complete icns file.
pub fn write(path: &Path) -> Result<()> {
    let mut body: Vec<u8> = Vec::new();

    for (ostype, size) in ENTRIES {
        let png = render_png(*size).with_context(|| format!("encoding the {size}px icns entry"))?;
        body.extend_from_slice(*ostype);
        // Length covers the 8-byte entry header as well as the payload.
        let entry_len = (png.len() + 8) as u32;
        body.extend_from_slice(&entry_len.to_be_bytes());
        body.extend_from_slice(&png);
    }

    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(b"icns");
    out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
    out.extend_from_slice(&body);

    std::fs::write(path, &out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn render_png(size: u32) -> Result<Vec<u8>> {
    let img = icon::render(size);
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png).write_image(
        img.as_raw(),
        size,
        size,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_container_is_well_formed() {
        let dir = std::env::temp_dir().join("ob-icns-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.icns");
        write(&path).unwrap();
        let data = std::fs::read(&path).unwrap();

        assert_eq!(&data[0..4], b"icns");
        // The declared length must match the file, or Finder rejects the icon
        // silently and falls back to a generic document glyph.
        let declared = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
        assert_eq!(declared, data.len());

        // Walk the entries and confirm they tile the body exactly.
        let mut off = 8;
        let mut seen = 0;
        while off < data.len() {
            let len = u32::from_be_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
            assert!(len >= 8, "entry length {len} does not cover its header");
            // Every payload is a PNG.
            assert_eq!(&data[off + 8..off + 12], b"\x89PNG");
            off += len;
            seen += 1;
        }
        assert_eq!(off, data.len(), "entries do not tile the body");
        assert_eq!(seen, ENTRIES.len());

        std::fs::remove_dir_all(&dir).ok();
    }
}
