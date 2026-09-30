//! Batching a tile grid must not change what the detector reports.
//!
//! The whole case for feeding several windows through the graph in one run is
//! that it is the same arithmetic, arranged better. That is a claim about a
//! real ONNX graph, not about our own code, so it is checked against a real
//! model rather than a mock — and skipped, loudly, where no model has been
//! downloaded.

use ob_core::geometry::{Frame, Region};
use ob_core::settings::SettingValues;
use ob_detect::session::OnnxDetector;
use ob_detect::Detector;

/// A frame with structure at several scales. A flat field would agree under
/// any implementation and prove nothing.
fn textured(w: u32, h: u32) -> Frame {
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let p = ((y * w + x) * 3) as usize;
            data[p] = (x.wrapping_mul(5) ^ y.wrapping_mul(11)) as u8;
            data[p + 1] = ((x / 7).wrapping_add(y / 3) % 253) as u8;
            data[p + 2] = (((x * x + y * y) / 64) % 256) as u8;
        }
    }
    Frame::new(w, h, data).unwrap()
}

/// Load the default model from the local cache, or `None` if it is not there.
fn detector() -> Option<OnnxDetector> {
    let entry = ob_core::registry::find("nudenet-320n")?;
    let path = ob_models::require(&entry).ok()?;
    let settings = ob_detect::resolve_settings(&entry, &SettingValues::new());
    OnnxDetector::load(&entry, &settings, path).ok()
}

#[test]
fn a_batched_grid_reports_exactly_what_window_by_window_reports() {
    let Some(d) = detector() else {
        eprintln!("skipped: nudenet-320n is not in the model cache");
        return;
    };

    let frame = textured(1500, 900);
    let windows = [
        Region::whole(&frame),
        Region::new(0, 0, 320, 320),
        Region::new(400, 200, 320, 320),
        Region::new(1180, 580, 320, 320),
        Region::new(760, 100, 320, 320),
    ];

    // One call: the windows are batched, in chunks of MAX_BATCH.
    let batched = d.detect_regions(&frame, &windows).unwrap();

    // The same windows asked about one at a time. A single-window call runs the
    // identical code path at n = 1, which is exactly what the unbatched
    // detector did, so any disagreement is the batch axis changing the answer.
    let mut separate = Vec::new();
    for w in &windows {
        separate.extend(d.detect_regions(&frame, std::slice::from_ref(w)).unwrap());
    }

    assert_eq!(
        batched.len(),
        separate.len(),
        "batched reported {} detections, window-by-window reported {}",
        batched.len(),
        separate.len()
    );
    for (i, (a, b)) in batched.iter().zip(&separate).enumerate() {
        assert_eq!(a.category, b.category, "detection {i}: category");
        assert_eq!(a.score, b.score, "detection {i}: score");
        assert_eq!(a.bbox, b.bbox, "detection {i}: box");
    }
}

#[test]
fn the_whole_frame_pass_is_unchanged_by_going_through_the_region_path() {
    // `detect` is now `detect_regions` over a single whole-frame window plus
    // NMS. It has to stay what it always was, because every caller that is not
    // tiling still goes through it.
    let Some(d) = detector() else {
        eprintln!("skipped: nudenet-320n is not in the model cache");
        return;
    };
    let frame = textured(800, 600);
    let a = d.detect(&frame).unwrap();
    let b = d.detect_regions(&frame, &[Region::whole(&frame)]).unwrap();
    // `detect` suppresses; the region call does not. Same boxes before NMS.
    assert!(
        a.len() <= b.len(),
        "suppression cannot invent detections: {} vs {}",
        a.len(),
        b.len()
    );
    for det in &a {
        assert!(
            b.iter().any(|o| o.bbox == det.bbox && o.score == det.score),
            "a detection survived NMS that the region pass never produced"
        );
    }
}
