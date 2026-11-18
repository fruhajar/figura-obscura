//! The review session: which item and frame are on the canvas, and getting
//! edits back out of it.
//!
//! The canvas itself is the Edit page's, unchanged: same drag, resize,
//! selection and undo. All this adds is a cursor over the batch and the two
//! conversions at its edges: detections into a document on the way in, and the
//! document back into detections on the way out.
//!
//! The one rule the whole thing rests on: **nothing is read out of the canvas
//! except through [`ReviewSession::flush`], and nothing moves the cursor
//! without flushing first.** A navigation that forgot would silently discard
//! the edit the user just made, and they would not find out until they opened
//! the written file.

use crate::edit::{detections_from_doc, EditorState};
use ob_job::review::{Review, ReviewItem};

/// Where a review is in its own lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewPhase {
    /// The detect pass is running.
    Detecting,
    /// It finished; the batch is waiting to be looked at.
    Ready,
    /// The commit pass is running.
    Committing,
    /// The files are written.
    Done,
}

pub struct ReviewSession {
    pub review: Review,
    pub phase: ReviewPhase,
    /// Index into `review.items`.
    pub item: usize,
    /// Which of the current item's detection slots is on screen. Always 0 for
    /// an image; a frame ordinal for a video (see `ReviewItem::slot_index`).
    pub slot: usize,
    /// The `(item, slot)` the canvas currently holds, so the pixels are only
    /// re-read when the cursor actually moves.
    pub loaded: Option<(usize, usize)>,
    /// Why the current frame could not be loaded.
    pub error: Option<String>,
    /// Include files the detector found nothing in. On by default: those are
    /// the ones a miss hides in, and a review that hid them would only ever
    /// show you what already worked.
    pub show_clean: bool,
    /// Items whose geometry the user has actually changed, so the summary can
    /// say what the review did rather than only that it happened.
    pub edited: std::collections::HashSet<usize>,
}

impl ReviewSession {
    pub fn new(review: Review) -> Self {
        Self {
            review,
            phase: ReviewPhase::Ready,
            item: 0,
            slot: 0,
            loaded: None,
            error: None,
            show_clean: true,
            edited: std::collections::HashSet::new(),
        }
    }

    /// A session with nothing in it yet, while the detect pass runs.
    pub fn detecting() -> Self {
        Self {
            phase: ReviewPhase::Detecting,
            ..Self::new(Review::default())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.review.items.is_empty()
    }

    pub fn current(&self) -> Option<&ReviewItem> {
        self.review.items.get(self.item)
    }

    /// The indices of the items the list is showing, in order.
    pub fn visible(&self) -> Vec<usize> {
        self.review
            .items
            .iter()
            .enumerate()
            .filter(|(_, i)| self.show_clean || !i.is_clean())
            .map(|(n, _)| n)
            .collect()
    }

    /// Write the canvas back into the item and slot it came from.
    ///
    /// Called before every cursor move and before the commit. Comparing against
    /// what was there means merely *looking* at a file does not mark it edited,
    /// which is what makes the "n files changed" summary mean anything.
    pub fn flush(&mut self, editor: &EditorState) {
        let Some((item, slot)) = self.loaded else {
            return;
        };
        let Some(target) = self.review.items.get_mut(item) else {
            return;
        };
        let dets = detections_from_doc(editor.doc());
        if target.detections_at(slot) != dets.as_slice() {
            target.set_detections_at(slot, dets);
            self.edited.insert(item);
        }
    }

    /// Move to an item, keeping the canvas and the cursor in step.
    pub fn go_to(&mut self, editor: &EditorState, item: usize) {
        self.flush(editor);
        self.item = item.min(self.review.items.len().saturating_sub(1));
        self.slot = 0;
    }

    pub fn go_to_slot(&mut self, editor: &EditorState, slot: usize) {
        self.flush(editor);
        let max = self.current().map(|i| i.slot_count()).unwrap_or(1);
        self.slot = slot.min(max.saturating_sub(1));
    }

    /// Step to the next or previous listed item. Returns false at the ends.
    pub fn step_item(&mut self, editor: &EditorState, forward: bool) -> bool {
        let visible = self.visible();
        let Some(at) = visible.iter().position(|&i| i == self.item) else {
            // The current item is filtered out: land on the nearest listed one
            // rather than refusing to move.
            if let Some(&first) = visible.first() {
                self.go_to(editor, first);
                return true;
            }
            return false;
        };
        let next = if forward {
            at.checked_add(1).filter(|n| *n < visible.len())
        } else {
            at.checked_sub(1)
        };
        match next {
            Some(n) => {
                self.go_to(editor, visible[n]);
                true
            }
            None => false,
        }
    }

    /// Jump to the next slot in this video that has something on it.
    pub fn next_populated_slot(&self, from: usize) -> Option<usize> {
        let item = self.current()?;
        item.frames_with_regions()
            .map(|f| item.slot_of(f.index))
            .find(|s| *s > from)
    }

    pub fn prev_populated_slot(&self, from: usize) -> Option<usize> {
        let item = self.current()?;
        // Frames are stored in index order, so the last one before the cursor
        // is the nearest.
        item.frames_with_regions()
            .map(|f| item.slot_of(f.index))
            .filter(|s| *s < from)
            .last()
    }

    /// Whether the canvas is showing the cursor's item and slot.
    pub fn canvas_is_current(&self) -> bool {
        self.loaded == Some((self.item, self.slot))
    }

    /// What the page title says about progress through the batch.
    pub fn position_label(&self) -> String {
        let visible = self.visible();
        let at = visible.iter().position(|&i| i == self.item);
        match at {
            Some(n) => format!("{} of {}", n + 1, visible.len()),
            None => format!("{} items", visible.len()),
        }
    }

    pub fn edited_count(&self) -> usize {
        self.edited.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::EditorState;
    use ob_core::geometry::{BBox, Detection};
    use ob_core::taxonomy::cat;
    use ob_job::review::{ReviewFrame, ReviewItem};
    use ob_media::MediaKind;
    use std::path::PathBuf;

    fn det(x: f32) -> Detection {
        Detection {
            bbox: BBox::new(x, 10.0, x + 20.0, 40.0),
            category: cat::FEMALE_BREAST_EXPOSED,
            score: 0.9,
        }
    }

    fn item(name: &str, dets: Vec<Detection>) -> ReviewItem {
        ReviewItem {
            path: PathBuf::from(name),
            output: PathBuf::from("out").join(name),
            kind: MediaKind::Image,
            width: 200,
            height: 100,
            frames: vec![ReviewFrame {
                index: 0,
                detections: dets,
            }],
            detect_every: 3,
            frame_count: None,
            error: None,
        }
    }

    fn session() -> ReviewSession {
        ReviewSession::new(Review {
            items: vec![
                item("a.png", vec![det(10.0)]),
                item("b.png", vec![]),
                item("c.png", vec![det(50.0), det(80.0)]),
            ],
        })
    }

    /// An editor holding `dets` as though it had been loaded for `(item, slot)`.
    fn canvas(dets: &[Detection]) -> EditorState {
        let mut editor = EditorState::default();
        editor.load_review(
            PathBuf::from("x.png"),
            ob_core::geometry::Frame::new(4, 4, vec![0u8; 48]).unwrap(),
            dets,
        );
        editor
    }

    #[test]
    fn flushing_writes_the_canvas_back_into_the_item_it_came_from() {
        let mut s = session();
        s.loaded = Some((2, 0));
        s.item = 2;
        // One of the two boxes deleted.
        let editor = canvas(&[det(50.0)]);
        s.flush(&editor);
        assert_eq!(s.review.items[2].region_count(), 1);
        assert_eq!(s.edited_count(), 1);
    }

    #[test]
    fn merely_looking_at_an_item_does_not_mark_it_edited() {
        // The summary counts files the review actually changed, so a flush that
        // put back exactly what it found must not count.
        let mut s = session();
        s.loaded = Some((0, 0));
        s.item = 0;
        let editor = canvas(&[det(10.0)]);
        s.flush(&editor);
        assert_eq!(s.edited_count(), 0);
        assert_eq!(s.review.items[0].region_count(), 1);
    }

    #[test]
    fn moving_between_items_flushes_first() {
        // The failure this guards is silent: navigate away, the edit is gone,
        // and nobody finds out until the file is written.
        let mut s = session();
        s.loaded = Some((0, 0));
        s.item = 0;
        let editor = canvas(&[det(10.0), det(99.0)]);
        s.go_to(&editor, 2);
        assert_eq!(s.item, 2);
        assert_eq!(s.review.items[0].region_count(), 2, "the edit was dropped");
    }

    #[test]
    fn stepping_stops_at_the_ends_rather_than_wrapping() {
        let mut s = session();
        let editor = EditorState::default();
        assert!(!s.step_item(&editor, false), "stepped back past the first");
        assert!(s.step_item(&editor, true));
        assert!(s.step_item(&editor, true));
        assert_eq!(s.item, 2);
        assert!(!s.step_item(&editor, true), "stepped past the last");
    }

    #[test]
    fn hiding_clean_files_takes_them_out_of_the_walk() {
        let mut s = session();
        s.show_clean = false;
        // b.png had no detections, so it drops out between a and c.
        assert_eq!(s.visible(), vec![0, 2]);
        let editor = EditorState::default();
        s.step_item(&editor, true);
        assert_eq!(s.item, 2);
    }

    #[test]
    fn a_filtered_out_item_still_lets_the_walk_continue() {
        // Turning the filter off while sitting on a clean file would otherwise
        // strand the cursor outside the list with both arrows dead.
        let mut s = session();
        s.item = 1;
        s.show_clean = false;
        let editor = EditorState::default();
        assert!(s.step_item(&editor, true));
        assert!(s.visible().contains(&s.item));
    }

    #[test]
    fn slot_navigation_is_clamped_to_the_clip() {
        let mut s = ReviewSession::new(Review {
            items: vec![{
                let mut v = item("clip.mp4", vec![]);
                v.kind = MediaKind::Video;
                v.detect_every = 3;
                v.frame_count = Some(10);
                v.frames.clear();
                v
            }],
        });
        let editor = EditorState::default();
        s.go_to_slot(&editor, 999);
        // 10 frames at one in three gives slots 0..=3.
        assert_eq!(s.slot, 3);
    }

    #[test]
    fn populated_slots_are_what_the_jump_buttons_find() {
        let mut v = item("clip.mp4", vec![]);
        v.kind = MediaKind::Video;
        v.detect_every = 3;
        v.frame_count = Some(30);
        v.frames = vec![
            ReviewFrame {
                index: 3,
                detections: vec![det(1.0)],
            },
            ReviewFrame {
                index: 9,
                detections: vec![],
            },
            ReviewFrame {
                index: 15,
                detections: vec![det(2.0)],
            },
        ];
        let s = ReviewSession::new(Review { items: vec![v] });
        // Slot 3 is frame 9, which was reviewed down to nothing, so it is not
        // somewhere the "next region" button should stop.
        assert_eq!(s.next_populated_slot(0), Some(1));
        assert_eq!(s.next_populated_slot(1), Some(5));
        assert_eq!(s.next_populated_slot(5), None);
        assert_eq!(s.prev_populated_slot(5), Some(1));
        assert_eq!(s.prev_populated_slot(1), None);
    }
}
