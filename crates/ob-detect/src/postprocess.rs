//! Non-maximum suppression and native-label mapping — pure and testable.

use ob_core::geometry::Detection;
use ob_core::registry::LabelMap;

/// Per-class greedy non-maximum suppression.
///
/// Detections are grouped by category; within each group, boxes are kept in
/// descending score order and any later box with IoU above `iou_threshold`
/// against a kept box is discarded.
pub fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    // Sort by score descending; NaN scores sink to the bottom.
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::with_capacity(dets.len());
    for d in dets {
        let suppressed = kept
            .iter()
            .any(|k| k.category == d.category && k.bbox.iou(&d.bbox) > iou_threshold);
        if !suppressed {
            kept.push(d);
        }
    }
    kept
}

/// Map a native class index to its canonical category, or `None` if the model
/// emitted an out-of-range class (which is dropped by the caller).
pub fn map_label(label_map: &LabelMap, native_index: usize) -> Option<ob_core::Category> {
    label_map.get(native_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::geometry::BBox;
    use ob_core::taxonomy::cat;

    fn det(x1: f32, score: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, 0.0, x1 + 10.0, 10.0),
            category: cat::FEMALE_BREAST_EXPOSED,
            score,
        }
    }

    #[test]
    fn nms_suppresses_lower_scoring_overlap() {
        let out = nms(vec![det(0.0, 0.9), det(1.0, 0.5)], 0.45);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].score, 0.9);
    }

    #[test]
    fn nms_keeps_disjoint_boxes() {
        let out = nms(vec![det(0.0, 0.9), det(100.0, 0.5)], 0.45);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn nms_keeps_overlap_of_different_categories() {
        let mut a = det(0.0, 0.9);
        let mut b = det(1.0, 0.8);
        a.category = cat::FACE_FEMALE;
        b.category = cat::FEMALE_BREAST_EXPOSED;
        let out = nms(vec![a, b], 0.45);
        assert_eq!(out.len(), 2);
    }
}
