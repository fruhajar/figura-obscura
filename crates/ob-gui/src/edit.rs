//! The manual editor's document: hand-drawn censor boxes, their undo history,
//! and the geometry that maps them onto the canvas.
//!
//! Everything in here is pure — no egui context, no I/O, no inference — so the
//! parts that are easy to get subtly wrong are unit-testable in a container
//! with no display server: what counts as *one* undo step, where a corner
//! handle sits, and what a drag that leaves the frame means.
//!
//! The pixels are not this module's business beyond [`render`], which hands
//! each box to `ob-censor` exactly as the batch pipeline would. A manual box
//! and a detected one are censored by the same code, so what the editor shows
//! is what a run would produce.

use egui::{Pos2, Rect, Vec2};
use ob_censor::CensorError;
use ob_core::censor::{CensorConfig, CensorStyle, RegionShape};
use ob_core::geometry::{BBox, Detection, Frame};
use ob_core::taxonomy::{Category, Part, Sex, State};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Smallest box a drag will leave behind, in image pixels.
///
/// Below this a drag was a click: someone clicking on empty canvas to deselect
/// should not litter the document with one-pixel boxes they then have to find
/// and delete.
pub const MIN_BOX: f32 = 4.0;

/// Size of the corner handle as drawn, in screen pixels.
pub const HANDLE_PX: f32 = 9.0;

/// How near a corner, in *screen* pixels, grabs the resize handle instead of
/// the box body.
///
/// Deliberately wider than the handle is drawn. Pressing near a corner is
/// almost always an attempt to resize — that is what corners are for — so the
/// target is the one worth being generous with, and asking for pixel precision
/// to hit a 9px square is the kind of thing that makes a tool feel stiff. The
/// cost of being wrong is small: a press meant as a move, a few pixels inside a
/// corner, resizes instead, which one drag undoes.
pub const GRAB_PX: f32 = 18.0;

/// A box's corner zones never take more than this share of its shorter side.
///
/// Without it, [`GRAB_PX`] would swallow a small box whole — every point in a
/// 30px box is within 18px of some corner — and it could then be resized but
/// never moved.
const GRAB_SHARE: f32 = 1.0 / 3.0;

/// Two edits sharing a coalescing key this close together are one undo step.
///
/// A drag or a slider sweep arrives as one change per frame; without this,
/// Ctrl-Z would walk back through sixty intermediate positions of a single
/// gesture. Re-armed on every touch, so a slow drag stays one step as long as
/// it keeps moving.
const COALESCE: Duration = Duration::from_millis(600);

/// Undo steps kept. The document is a handful of rectangles, so a snapshot is
/// cheap enough that there is no reason to be stingy.
const DEPTH: usize = 64;

/// One hand-drawn region.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ManualBox {
    /// Stable across undo and redo, so the selection survives a step.
    pub id: u64,
    pub rect: BBox,
    pub style: CensorStyle,
    /// Corner rounding, as a fraction of the shorter side (0 = square).
    ///
    /// Per box rather than per document, like the style it belongs with:
    /// a slider that silently reshaped every box already drawn was the odd one
    /// out among controls that all act on the selection.
    pub rounding: f32,
    /// Hidden boxes stay in the list but are not drawn — a layer's eye. Kept
    /// rather than deleted so a style can be compared against the original
    /// without losing the box.
    pub visible: bool,
}

impl ManualBox {
    pub fn new(id: u64, rect: BBox, style: CensorStyle, rounding: f32) -> Self {
        Self {
            id,
            rect,
            style,
            rounding,
            visible: true,
        }
    }

    /// What the list calls this box: its number and its style.
    pub fn label(&self) -> String {
        format!("Box {} · {}", self.id, style_name(&self.style))
    }
}

/// The whole editable state of one image: the boxes and the geometry shared by
/// all of them.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EditDoc {
    /// Drawn in order, so a later box paints over an earlier one. The list is
    /// bottom-to-top, like layers.
    pub boxes: Vec<ManualBox>,
}

impl EditDoc {
    pub fn find(&self, id: u64) -> Option<&ManualBox> {
        self.boxes.iter().find(|b| b.id == id)
    }

    pub fn find_mut(&mut self, id: u64) -> Option<&mut ManualBox> {
        self.boxes.iter_mut().find(|b| b.id == id)
    }

    pub fn position(&self, id: u64) -> Option<usize> {
        self.boxes.iter().position(|b| b.id == id)
    }

    /// Hash of everything that changes the rendered pixels.
    ///
    /// Drives the composite cache: the editor re-censors the frame when this
    /// moves and not otherwise, so selecting a box or scrolling the list costs
    /// nothing.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // Via JSON because `CensorStyle` carries floats, which are not `Hash`.
        // Only ever compared with itself, never stored, so the hasher's lack of
        // cross-build stability does not matter.
        serde_json::to_string(self).unwrap_or_default().hash(&mut h);
        h.finish()
    }
}

/// Snapshot undo/redo over an [`EditDoc`].
///
/// Snapshots rather than inverse commands: the document is a few rectangles, so
/// a full copy costs nothing, and every edit is then undoable by construction
/// instead of by remembering to write a matching inverse for each one.
#[derive(Default)]
pub struct History {
    current: EditDoc,
    past: Vec<EditDoc>,
    future: Vec<EditDoc>,
    /// The last coalescing key and when it was touched. See [`COALESCE`].
    last: Option<(String, Instant)>,
}

impl History {
    pub fn doc(&self) -> &EditDoc {
        &self.current
    }

    /// Begin a discrete change: adding, deleting, reordering, toggling.
    ///
    /// Always its own undo step, so two boxes drawn in quick succession are two
    /// presses of Ctrl-Z rather than one.
    pub fn step(&mut self) -> &mut EditDoc {
        self.push_snapshot();
        self.last = None;
        &mut self.current
    }

    /// Begin a change that is part of a continuous gesture — a drag, a slider
    /// sweep — identified by `key`.
    ///
    /// Consecutive calls with the same key fold into the step the first one
    /// opened, so the whole gesture undoes at once.
    pub fn stroke(&mut self, key: &str, now: Instant) -> &mut EditDoc {
        let continuing = matches!(
            &self.last,
            Some((k, at)) if k == key && now.saturating_duration_since(*at) < COALESCE
        );
        if !continuing {
            self.push_snapshot();
        }
        self.last = Some((key.to_string(), now));
        &mut self.current
    }

    fn push_snapshot(&mut self) {
        self.past.push(self.current.clone());
        if self.past.len() > DEPTH {
            self.past.remove(0);
        }
        // A new edit is a new branch; anything that was undone is gone.
        self.future.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.past.pop() else {
            return false;
        };
        self.future.push(std::mem::replace(&mut self.current, prev));
        // Whatever gesture was in progress is over; the next edit must open its
        // own step rather than folding into the one just undone.
        self.last = None;
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(next) = self.future.pop() else {
            return false;
        };
        self.past.push(std::mem::replace(&mut self.current, next));
        self.last = None;
        true
    }

    /// Replace the document and forget the history — a different image.
    pub fn reset(&mut self, doc: EditDoc) {
        self.current = doc;
        self.past.clear();
        self.future.clear();
        self.last = None;
    }
}

/// Where the image is drawn, and the conversion between image pixels and screen
/// pixels in both directions.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// The on-screen rectangle the image occupies.
    pub rect: Rect,
    /// Screen pixels per image pixel.
    pub scale: f32,
}

impl View {
    /// Fit a `w × h` image centred inside `avail`.
    pub fn fit(w: u32, h: u32, avail: Rect) -> View {
        let (iw, ih) = (w.max(1) as f32, h.max(1) as f32);
        let scale = (avail.width() / iw)
            .min(avail.height() / ih)
            // A zero-sized panel for one frame (a collapsed window, the first
            // layout pass) must not produce a degenerate transform that maps
            // every click to the same pixel.
            .max(f32::MIN_POSITIVE);
        let size = Vec2::new(iw * scale, ih * scale);
        View {
            rect: Rect::from_center_size(avail.center(), size),
            scale,
        }
    }

    pub fn to_screen(self, b: &BBox) -> Rect {
        Rect::from_min_max(
            self.rect.min + Vec2::new(b.x1, b.y1) * self.scale,
            self.rect.min + Vec2::new(b.x2, b.y2) * self.scale,
        )
    }

    /// Screen position to image pixels. Not clamped: a drag that leaves the
    /// image still has to say which way it went.
    pub fn to_image(self, p: Pos2) -> Pos2 {
        ((p - self.rect.min) / self.scale).to_pos2()
    }

    /// [`GRAB_PX`] expressed in image pixels at this zoom, so the corner target
    /// is the same physical size however far the image is scaled down.
    pub fn grab_in_image(self) -> f32 {
        GRAB_PX / self.scale
    }
}

/// Which corner of a box a drag has hold of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    pub const ALL: [Corner; 4] = [
        Corner::TopLeft,
        Corner::TopRight,
        Corner::BottomLeft,
        Corner::BottomRight,
    ];

    /// The corner's position, in whatever space `b` is in.
    pub fn of(self, b: &BBox) -> Pos2 {
        match self {
            Corner::TopLeft => Pos2::new(b.x1, b.y1),
            Corner::TopRight => Pos2::new(b.x2, b.y1),
            Corner::BottomLeft => Pos2::new(b.x1, b.y2),
            Corner::BottomRight => Pos2::new(b.x2, b.y2),
        }
    }
}

/// What a press on the canvas landed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grab {
    /// Inside the box: move it.
    Body,
    /// On a corner handle: resize from there.
    Corner(Corner),
}

/// A gesture in progress. Coordinates are image pixels throughout, so the
/// gesture survives the panel being resized under it.
#[derive(Clone, Debug, PartialEq)]
pub enum Drag {
    /// Drawing a new box, from a fixed anchor to the pointer.
    New { anchor: Pos2, current: Pos2 },
    /// Moving or resizing boxes that already exist. `origins` is where they
    /// were when the gesture began, so the result is always computed from the
    /// press position rather than accumulated frame by frame.
    ///
    /// A move carries the whole selection — grabbing one of several selected
    /// boxes and having the others stay behind would make the selection look
    /// like it had been forgotten. A resize carries only `id`: there is no one
    /// obvious thing "resize these four" should mean.
    Existing {
        id: u64,
        grab: Grab,
        anchor: Pos2,
        origins: Vec<(u64, BBox)>,
    },
}

/// What a press at `p` (image pixels) should take hold of.
///
/// Corner handles of the *selected* boxes win over every body, so a small box
/// sitting on top of a big one can still be resized; otherwise the topmost box
/// containing the point is moved. `None` means empty canvas: draw a new box
/// there.
///
/// `grab` is the corner radius in image pixels — see [`View::grab_in_image`].
pub fn hit(doc: &EditDoc, p: Pos2, selected: &[u64], grab: f32) -> Option<(u64, Grab)> {
    // Topmost first among the selected, so stacked selected boxes resolve the
    // same way bodies do.
    for sel in doc.boxes.iter().rev().filter(|b| selected.contains(&b.id)) {
        // Never let the corner zones meet in the middle of a small box, or it
        // could be resized but never moved.
        let near = grab
            .min(sel.rect.width() * GRAB_SHARE)
            .min(sel.rect.height() * GRAB_SHARE);
        for corner in Corner::ALL {
            let c = corner.of(&sel.rect);
            if (c.x - p.x).abs() <= near && (c.y - p.y).abs() <= near {
                return Some((sel.id, Grab::Corner(corner)));
            }
        }
    }
    // Reverse: the list is bottom-to-top, so the last match is the box a user
    // sees on top.
    doc.boxes
        .iter()
        .rev()
        .find(|b| contains(&b.rect, p))
        .map(|b| (b.id, Grab::Body))
}

fn contains(b: &BBox, p: Pos2) -> bool {
    p.x >= b.x1 && p.x <= b.x2 && p.y >= b.y1 && p.y <= b.y2
}

/// The box a from-scratch drag describes, or `None` when it was really a click.
pub fn from_drag(anchor: Pos2, current: Pos2, img: Vec2) -> Option<BBox> {
    let b = clamped(
        BBox::new(
            anchor.x.min(current.x),
            anchor.y.min(current.y),
            anchor.x.max(current.x),
            anchor.y.max(current.y),
        ),
        img,
    );
    (b.width() >= MIN_BOX && b.height() >= MIN_BOX).then_some(b)
}

/// Translate a box, keeping it whole and inside the image.
///
/// The box is slid back in rather than clipped: dragging a region off the edge
/// and getting a thinner one back is not what the gesture asked for.
pub fn moved(origin: &BBox, delta: Vec2, img: Vec2) -> BBox {
    let (w, h) = (origin.width(), origin.height());
    let x1 = (origin.x1 + delta.x).clamp(0.0, (img.x - w).max(0.0));
    let y1 = (origin.y1 + delta.y).clamp(0.0, (img.y - h).max(0.0));
    BBox::new(x1, y1, x1 + w, y1 + h)
}

/// Translate a group of boxes together, as one rigid shape.
///
/// The clamp applies to the group's bounding box rather than to each member, so
/// a selection dragged into a corner keeps its internal spacing instead of
/// collapsing as its members hit the edge one by one.
pub fn moved_group(origins: &[(u64, BBox)], delta: Vec2, img: Vec2) -> Vec<(u64, BBox)> {
    let Some(union) = union_of(origins) else {
        return Vec::new();
    };
    let slid = moved(&union, delta, img);
    let d = Vec2::new(slid.x1 - union.x1, slid.y1 - union.y1);
    origins
        .iter()
        .map(|(id, b)| {
            (
                *id,
                BBox::new(b.x1 + d.x, b.y1 + d.y, b.x2 + d.x, b.y2 + d.y),
            )
        })
        .collect()
}

fn union_of(boxes: &[(u64, BBox)]) -> Option<BBox> {
    boxes.iter().map(|(_, b)| *b).reduce(|a, b| {
        BBox::new(
            a.x1.min(b.x1),
            a.y1.min(b.y1),
            a.x2.max(b.x2),
            a.y2.max(b.y2),
        )
    })
}

/// Move one corner of a box to `to`, keeping the opposite corner fixed.
///
/// Dragging a corner past its opposite flips the box rather than collapsing it,
/// which is what every drawing tool does and what the hand expects.
pub fn resized(origin: &BBox, corner: Corner, to: Pos2, img: Vec2) -> BBox {
    let (mut x1, mut y1, mut x2, mut y2) = (origin.x1, origin.y1, origin.x2, origin.y2);
    match corner {
        Corner::TopLeft => {
            x1 = to.x;
            y1 = to.y;
        }
        Corner::TopRight => {
            x2 = to.x;
            y1 = to.y;
        }
        Corner::BottomLeft => {
            x1 = to.x;
            y2 = to.y;
        }
        Corner::BottomRight => {
            x2 = to.x;
            y2 = to.y;
        }
    }
    let b = BBox::new(x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2));
    let b = clamped(b, img);
    // A resize may not delete the box: hold the floor at MIN_BOX, growing from
    // whichever edge has room.
    let x2 = b.x2.max(b.x1 + MIN_BOX).min(img.x);
    let y2 = b.y2.max(b.y1 + MIN_BOX).min(img.y);
    BBox::new((x2 - MIN_BOX).min(b.x1), (y2 - MIN_BOX).min(b.y1), x2, y2)
}

fn clamped(b: BBox, img: Vec2) -> BBox {
    BBox::new(
        b.x1.clamp(0.0, img.x),
        b.y1.clamp(0.0, img.y),
        b.x2.clamp(0.0, img.x),
        b.y2.clamp(0.0, img.y),
    )
}

/// Draw every visible box over a copy of `source`.
///
/// Each box carries its own style, so each is a separate `ob_censor::apply`
/// call with that style as the default and no per-part overrides — the
/// taxonomy does not apply to a region someone drew by hand. Padding is zero,
/// unlike the batch default: the box *is* what the user drew, and quietly
/// growing it by 10% would make the drawn outline a lie.
pub fn render(source: &Frame, doc: &EditDoc) -> Result<Frame, CensorError> {
    let mut out = source.clone();
    for b in doc.boxes.iter().filter(|b| b.visible) {
        let cfg = CensorConfig {
            default_style: b.style.clone(),
            per_part: HashMap::new(),
            shape: RegionShape {
                padding: 0.0,
                rounding: b.rounding,
            },
        };
        ob_censor::apply(&mut out, &[manual_detection(b.rect)], &cfg)?;
    }
    Ok(out)
}

/// Dress a hand-drawn rectangle up as a detection so the shared censor path can
/// take it.
///
/// The category is never consulted — `per_part` is empty in [`render`], so
/// `style_for` falls through to the default every time — but one has to be
/// supplied, and an honest "nobody classified this" beats borrowing a real
/// body part's label.
fn manual_detection(rect: BBox) -> Detection {
    Detection {
        bbox: rect,
        category: Category::new(Sex::Unknown, Part::Face, State::Covered),
        score: 1.0,
    }
}

/// A censor style's name, as the picker and the box list show it.
pub fn style_name(s: &CensorStyle) -> &'static str {
    match s {
        CensorStyle::SolidFill { .. } => "Solid fill",
        CensorStyle::Pixelate { .. } => "Pixelate",
        CensorStyle::Blur { .. } => "Blur",
        CensorStyle::ImageOverlay { .. } => "Image overlay",
    }
}

/// Everything the Edit page owns: the loaded image, the document and its
/// history, the selection, and the cached composite on screen.
pub struct EditorState {
    pub path: Option<PathBuf>,
    /// The image as decoded. Kept so every re-censor starts from clean pixels
    /// rather than from the last render, which would compound every edit.
    pub source: Option<Frame>,
    pub history: History,
    /// The boxes under the controls, in the order they were picked. The last is
    /// the primary: the one the style editor reads its values from, and the one
    /// a resize acts on.
    ///
    /// A list rather than a single id so several boxes can be restyled at once
    /// (Ctrl-click to add one). Kept out of the document: selecting is not an
    /// edit, and undoing back through selections would be maddening.
    selection: Vec<u64>,
    /// The style and rounding the next drawn box gets. Not part of the
    /// document: drawing five boxes in one style should take one trip to the
    /// picker, and changing it is not an edit to undo.
    pub new_style: CensorStyle,
    pub new_rounding: f32,
    /// Why the last load or save failed.
    pub error: Option<String>,
    pub drag: Option<Drag>,
    /// The composite on screen, and the document fingerprint it was built from.
    pub texture: Option<egui::TextureHandle>,
    pub rendered: Option<u64>,
    next_id: u64,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            path: None,
            source: None,
            history: History::default(),
            selection: Vec::new(),
            new_style: CensorStyle::default(),
            new_rounding: 0.0,
            error: None,
            drag: None,
            texture: None,
            rendered: None,
            next_id: 1,
        }
    }
}

impl EditorState {
    /// Take on a freshly decoded image, discarding the previous document.
    ///
    /// Boxes are positions in a particular image; carrying them onto the next
    /// one would censor whatever happened to be at those coordinates.
    pub fn load(&mut self, path: PathBuf, frame: Frame) {
        self.path = Some(path);
        self.source = Some(frame);
        self.history.reset(EditDoc::default());
        self.selection.clear();
        self.drag = None;
        self.error = None;
        self.texture = None;
        self.rendered = None;
        self.next_id = 1;
    }

    pub fn image_size(&self) -> Option<Vec2> {
        self.source
            .as_ref()
            .map(|f| Vec2::new(f.width_f(), f.height_f()))
    }

    pub fn doc(&self) -> &EditDoc {
        self.history.doc()
    }

    /// Every selected box, in pick order.
    pub fn selection(&self) -> &[u64] {
        &self.selection
    }

    pub fn is_selected(&self, id: u64) -> bool {
        self.selection.contains(&id)
    }

    /// The box the controls read from: the one picked most recently.
    pub fn selected_box(&self) -> Option<&ManualBox> {
        self.selection.last().and_then(|id| self.doc().find(*id))
    }

    /// Where every selected box is now — the starting point of a group move.
    pub fn selected_rects(&self) -> Vec<(u64, BBox)> {
        let doc = self.doc();
        self.selection
            .iter()
            .filter_map(|id| doc.find(*id).map(|b| (*id, b.rect)))
            .collect()
    }

    /// Replace the selection with one box, or clear it.
    pub fn select_only(&mut self, id: Option<u64>) {
        self.selection.clear();
        if let Some(id) = id {
            self.selection.push(id);
        }
    }

    /// Add a box to the selection, or take it out — what Ctrl-click does.
    ///
    /// Re-picking moves it to the end, so the controls follow the box just
    /// clicked rather than the one picked first.
    pub fn toggle_selected(&mut self, id: u64) {
        if let Some(i) = self.selection.iter().position(|x| *x == id) {
            self.selection.remove(i);
        } else {
            self.selection.push(id);
        }
    }

    /// Add a box in the current new-box style and select it alone.
    pub fn add_box(&mut self, rect: BBox) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let style = self.new_style.clone();
        let rounding = self.new_rounding;
        self.history
            .step()
            .boxes
            .push(ManualBox::new(id, rect, style, rounding));
        self.select_only(Some(id));
        id
    }

    pub fn delete(&mut self, id: u64) {
        self.history.step().boxes.retain(|b| b.id != id);
        self.selection.retain(|x| *x != id);
    }

    /// Delete every selected box as one undo step.
    pub fn delete_selected(&mut self) {
        if self.selection.is_empty() {
            return;
        }
        let gone = std::mem::take(&mut self.selection);
        self.history.step().boxes.retain(|b| !gone.contains(&b.id));
    }

    /// Apply `f` to every selected box as one undo step, coalescing with the
    /// previous call for the same selection.
    ///
    /// This is what makes a style change act on all of them: the controls show
    /// the primary's values, and whatever the user does to those lands on each
    /// box they picked.
    pub fn edit_selection(&mut self, what: &str, now: Instant, f: impl Fn(&mut ManualBox)) {
        if self.selection.is_empty() {
            return;
        }
        // Keyed by the whole selection, so changing which boxes are picked
        // starts a new undo step rather than extending the last one.
        let key = format!("{what}:{:?}", self.selection);
        let picked = self.selection.clone();
        let doc = self.history.stroke(&key, now);
        for b in doc.boxes.iter_mut().filter(|b| picked.contains(&b.id)) {
            f(b);
        }
    }

    /// Swap a box with its neighbour. `up` moves it towards the top of the
    /// stack, i.e. later in the paint order.
    pub fn reorder(&mut self, id: u64, up: bool) {
        let doc = self.history.step();
        let Some(i) = doc.position(id) else { return };
        let j = if up { i + 1 } else { i.wrapping_sub(1) };
        if j < doc.boxes.len() {
            doc.boxes.swap(i, j);
        }
    }

    pub fn undo(&mut self) {
        if self.history.undo() {
            self.prune_selection();
        }
    }

    pub fn redo(&mut self) {
        if self.history.redo() {
            self.prune_selection();
        }
    }

    /// Drop selected ids the document no longer holds. An undo can remove a
    /// selected box, and controls pointed at a box that is gone would have
    /// nothing to edit.
    fn prune_selection(&mut self) {
        let doc = self.history.doc();
        self.selection.retain(|id| doc.find(*id).is_some());
    }

    /// The censored frame as it stands, or `None` with no image loaded.
    pub fn composite(&self) -> Option<Result<Frame, CensorError>> {
        self.source.as_ref().map(|src| render(src, self.doc()))
    }

    /// The id the next box would get. Test-only: ids are an implementation
    /// detail everywhere else.
    #[cfg(test)]
    pub fn peek_next_id(&self) -> u64 {
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    fn t0() -> Instant {
        Instant::now()
    }

    fn doc_with(rects: &[BBox]) -> EditDoc {
        EditDoc {
            boxes: rects
                .iter()
                .enumerate()
                .map(|(i, r)| ManualBox::new(i as u64 + 1, *r, CensorStyle::default(), 0.0))
                .collect(),
        }
    }

    #[test]
    fn a_discrete_edit_is_one_undo_step_each() {
        let mut h = History::default();
        assert!(!h.can_undo());
        h.step().boxes.push(ManualBox::new(
            1,
            BBox::new(0.0, 0.0, 1.0, 1.0),
            CensorStyle::default(),
            0.0,
        ));
        h.step().boxes.push(ManualBox::new(
            2,
            BBox::new(1.0, 1.0, 2.0, 2.0),
            CensorStyle::default(),
            0.0,
        ));
        assert_eq!(h.doc().boxes.len(), 2);

        // Two boxes added back to back are two presses of Ctrl-Z, even though
        // nothing separated them in time.
        assert!(h.undo());
        assert_eq!(h.doc().boxes.len(), 1);
        assert!(h.undo());
        assert_eq!(h.doc().boxes.len(), 0);
        assert!(!h.undo());
    }

    #[test]
    fn one_gesture_is_one_undo_step() {
        let mut h = History::default();
        h.step().boxes.push(ManualBox::new(
            1,
            BBox::new(0.0, 0.0, 10.0, 10.0),
            CensorStyle::default(),
            0.0,
        ));
        let start = t0();
        // Sixty frames of a drag, as the canvas would deliver them.
        for i in 0..60 {
            let at = start + Duration::from_millis(i * 16);
            h.stroke("box:1", at).boxes[0].rect = BBox::new(i as f32, 0.0, 10.0 + i as f32, 10.0);
        }
        assert_eq!(h.doc().boxes[0].rect.x1, 59.0);
        // One Ctrl-Z returns the box to where the gesture started, not to the
        // previous frame of it.
        h.undo();
        assert_eq!(h.doc().boxes[0].rect.x1, 0.0);
        assert_eq!(h.doc().boxes.len(), 1);
    }

    #[test]
    fn a_pause_breaks_a_gesture_into_two_steps() {
        let mut h = History::default();
        h.step().boxes.push(ManualBox::new(
            1,
            BBox::new(0.0, 0.0, 10.0, 10.0),
            CensorStyle::default(),
            0.0,
        ));
        let start = t0();
        h.stroke("box:1", start).boxes[0].rect = BBox::new(5.0, 0.0, 15.0, 10.0);
        // Let go, think, drag again: two separate things the user did, so two
        // separate things to undo.
        h.stroke("box:1", start + COALESCE + Duration::from_millis(1))
            .boxes[0]
            .rect = BBox::new(9.0, 0.0, 19.0, 10.0);
        h.undo();
        assert_eq!(h.doc().boxes[0].rect.x1, 5.0);
        h.undo();
        assert_eq!(h.doc().boxes[0].rect.x1, 0.0);
    }

    #[test]
    fn different_keys_never_coalesce() {
        let mut h = History::default();
        let d = h.step();
        d.boxes.push(ManualBox::new(
            1,
            BBox::new(0.0, 0.0, 10.0, 10.0),
            CensorStyle::default(),
            0.0,
        ));
        d.boxes.push(ManualBox::new(
            2,
            BBox::new(0.0, 0.0, 10.0, 10.0),
            CensorStyle::default(),
            0.0,
        ));
        let now = t0();
        h.stroke("box:1", now).boxes[0].rect = BBox::new(1.0, 0.0, 11.0, 10.0);
        h.stroke("box:2", now).boxes[1].rect = BBox::new(2.0, 0.0, 12.0, 10.0);
        h.undo();
        assert_eq!(h.doc().boxes[1].rect.x1, 0.0);
        assert_eq!(h.doc().boxes[0].rect.x1, 1.0);
    }

    /// A history holding one box, for the tests that only need somewhere to
    /// put a changing value.
    fn history_with_a_box() -> History {
        let mut h = History::default();
        h.step().boxes.push(ManualBox::new(
            1,
            BBox::new(0.0, 0.0, 10.0, 10.0),
            CensorStyle::default(),
            0.0,
        ));
        h
    }

    fn rounding(h: &History) -> f32 {
        h.doc().boxes[0].rounding
    }

    #[test]
    fn redo_replays_what_undo_took_back() {
        let mut h = history_with_a_box();
        h.step().boxes[0].rounding = 0.25;
        h.undo();
        assert_eq!(rounding(&h), 0.0);
        assert!(h.can_redo());
        h.redo();
        assert_eq!(rounding(&h), 0.25);
        assert!(!h.can_redo());
    }

    #[test]
    fn editing_after_an_undo_discards_the_redo_branch() {
        let mut h = history_with_a_box();
        h.step().boxes[0].rounding = 0.1;
        h.undo();
        h.step().boxes[0].rounding = 0.5;
        // The branch that was undone is unreachable now; offering a redo that
        // jumped to 0.1 would discard the 0.5 the user just set.
        assert!(!h.can_redo());
        assert_eq!(rounding(&h), 0.5);
    }

    #[test]
    fn an_edit_right_after_an_undo_is_not_folded_into_it() {
        let mut h = history_with_a_box();
        h.step().boxes[0].rounding = 0.1;
        let now = t0();
        h.stroke("round", now).boxes[0].rounding = 0.2;
        h.undo();
        assert_eq!(rounding(&h), 0.1);
        // Same key, well inside the coalescing window, but the undo closed the
        // gesture: this must be undoable on its own.
        h.stroke("round", now + Duration::from_millis(10)).boxes[0].rounding = 0.3;
        h.undo();
        assert_eq!(rounding(&h), 0.1);
    }

    #[test]
    fn the_history_depth_is_bounded() {
        let mut h = history_with_a_box();
        for i in 0..(DEPTH + 20) {
            h.step().boxes[0].rounding = i as f32;
        }
        let mut steps = 0;
        while h.undo() {
            steps += 1;
        }
        assert_eq!(steps, DEPTH);
    }

    #[test]
    fn the_view_round_trips_image_coordinates() {
        let avail = Rect::from_min_size(pos2(10.0, 20.0), Vec2::new(400.0, 300.0));
        let view = View::fit(800, 600, avail);
        assert!((view.scale - 0.5).abs() < 1e-6);
        // Fitted exactly, so the image fills the panel.
        assert!((view.rect.width() - 400.0).abs() < 1e-3);

        let b = BBox::new(100.0, 200.0, 300.0, 400.0);
        let screen = view.to_screen(&b);
        let back = view.to_image(screen.min);
        assert!((back.x - 100.0).abs() < 1e-3, "{back:?}");
        assert!((back.y - 200.0).abs() < 1e-3, "{back:?}");
    }

    #[test]
    fn a_letterboxed_image_is_centred() {
        // A wide image in a square panel: fitted on width, centred vertically.
        let avail = Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(400.0, 400.0));
        let view = View::fit(400, 200, avail);
        assert!((view.scale - 1.0).abs() < 1e-6);
        assert!((view.rect.min.y - 100.0).abs() < 1e-3);
        assert!((view.rect.min.x - 0.0).abs() < 1e-3);
    }

    #[test]
    fn the_corner_target_stays_the_same_size_on_screen() {
        let avail = Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(400.0, 300.0));
        // A 4000px-wide image fitted into 400px: the corner target has to cover
        // ten times as many image pixels as it would at 1:1, or the corners of
        // a big image would be unhittable.
        let view = View::fit(4000, 3000, avail);
        assert!((view.grab_in_image() - GRAB_PX * 10.0).abs() < 1e-3);
    }

    #[test]
    fn handles_of_the_selected_box_win_over_a_box_on_top() {
        // A small box sitting entirely inside a big one. Without handle
        // priority the big box would swallow every press and the small one
        // could never be resized.
        let doc = doc_with(&[
            BBox::new(0.0, 0.0, 100.0, 100.0),
            BBox::new(40.0, 40.0, 60.0, 60.0),
        ]);
        let at_corner = pos2(60.0, 60.0);
        assert_eq!(
            hit(&doc, at_corner, &[2], 5.0),
            Some((2, Grab::Corner(Corner::BottomRight)))
        );
        // Not selected: the same press moves the topmost box containing it.
        assert_eq!(hit(&doc, at_corner, &[], 5.0), Some((2, Grab::Body)));
    }

    #[test]
    fn a_press_just_inside_a_corner_resizes_rather_than_moves() {
        // The point of the wider target: a few pixels inside the corner, well
        // past the 9px handle as drawn, is still a resize. Pressing near a
        // corner is what someone does when they want to resize.
        let doc = doc_with(&[BBox::new(0.0, 0.0, 200.0, 200.0)]);
        // Past the 9px the handle is drawn at — which is the point, and which
        // fails if `GRAB_PX` is ever brought back down to the handle's size.
        let just_inside = pos2(14.0, 14.0);
        assert_eq!(
            hit(&doc, just_inside, &[1], GRAB_PX),
            Some((1, Grab::Corner(Corner::TopLeft)))
        );
        // Far enough in, it is a move again.
        assert_eq!(
            hit(&doc, pos2(60.0, 60.0), &[1], GRAB_PX),
            Some((1, Grab::Body))
        );
    }

    #[test]
    fn the_corner_target_never_swallows_a_small_box() {
        // Every point of a 30px box is within 18px of some corner. Without the
        // cap it could be resized but never moved, which is worse than a
        // slightly tighter corner target.
        let doc = doc_with(&[BBox::new(0.0, 0.0, 30.0, 30.0)]);
        assert_eq!(
            hit(&doc, pos2(15.0, 15.0), &[1], GRAB_PX),
            Some((1, Grab::Body)),
            "the middle of a small box must still move it"
        );
        // Its corners still resize.
        assert_eq!(
            hit(&doc, pos2(1.0, 1.0), &[1], GRAB_PX),
            Some((1, Grab::Corner(Corner::TopLeft)))
        );
    }

    #[test]
    fn only_selected_boxes_offer_corner_handles() {
        let doc = doc_with(&[BBox::new(0.0, 0.0, 200.0, 200.0)]);
        // Nothing selected: the same press is a move, because no handles are
        // drawn on an unselected box and a press must do what it looks like it
        // will do.
        assert_eq!(
            hit(&doc, pos2(5.0, 5.0), &[], GRAB_PX),
            Some((1, Grab::Body))
        );
    }

    #[test]
    fn a_press_on_empty_canvas_grabs_nothing() {
        let doc = doc_with(&[BBox::new(0.0, 0.0, 10.0, 10.0)]);
        assert_eq!(hit(&doc, pos2(50.0, 50.0), &[1], 4.0), None);
    }

    #[test]
    fn the_topmost_box_takes_the_press() {
        let doc = doc_with(&[
            BBox::new(0.0, 0.0, 100.0, 100.0),
            BBox::new(0.0, 0.0, 100.0, 100.0),
        ]);
        // Identical boxes: the one painted last is the one the user can see.
        assert_eq!(hit(&doc, pos2(50.0, 50.0), &[], 4.0), Some((2, Grab::Body)));
    }

    #[test]
    fn a_click_does_not_create_a_box() {
        let img = Vec2::new(100.0, 100.0);
        assert!(from_drag(pos2(10.0, 10.0), pos2(10.0, 10.0), img).is_none());
        assert!(from_drag(pos2(10.0, 10.0), pos2(12.0, 30.0), img).is_none());
        assert!(from_drag(pos2(10.0, 10.0), pos2(30.0, 30.0), img).is_some());
    }

    #[test]
    fn a_box_drawn_backwards_comes_out_the_right_way_round() {
        let img = Vec2::new(100.0, 100.0);
        let b = from_drag(pos2(80.0, 90.0), pos2(20.0, 30.0), img).unwrap();
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (20.0, 30.0, 80.0, 90.0));
    }

    #[test]
    fn a_drag_off_the_edge_is_clipped_to_the_image() {
        let img = Vec2::new(100.0, 100.0);
        let b = from_drag(pos2(50.0, 50.0), pos2(400.0, -80.0), img).unwrap();
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (50.0, 0.0, 100.0, 50.0));
    }

    #[test]
    fn moving_a_box_off_the_edge_slides_it_rather_than_shrinking_it() {
        let img = Vec2::new(100.0, 100.0);
        let origin = BBox::new(10.0, 10.0, 30.0, 40.0);
        let moved_far = moved(&origin, Vec2::new(500.0, 500.0), img);
        assert_eq!(moved_far.width(), origin.width());
        assert_eq!(moved_far.height(), origin.height());
        assert_eq!((moved_far.x2, moved_far.y2), (100.0, 100.0));

        let moved_back = moved(&origin, Vec2::new(-500.0, -500.0), img);
        assert_eq!((moved_back.x1, moved_back.y1), (0.0, 0.0));
        assert_eq!(moved_back.width(), origin.width());
    }

    #[test]
    fn a_box_larger_than_the_image_is_not_squeezed_by_a_move() {
        // Can happen after a resize at the frame edge; the clamp must not
        // invert the range it is given.
        let img = Vec2::new(50.0, 50.0);
        let origin = BBox::new(0.0, 0.0, 80.0, 80.0);
        let out = moved(&origin, Vec2::new(10.0, 10.0), img);
        assert_eq!((out.x1, out.y1), (0.0, 0.0));
        assert_eq!(out.width(), 80.0);
    }

    #[test]
    fn dragging_a_corner_past_its_opposite_flips_the_box() {
        let img = Vec2::new(100.0, 100.0);
        let origin = BBox::new(20.0, 20.0, 60.0, 60.0);
        // Pull the top-left down past the bottom-right.
        let out = resized(&origin, Corner::TopLeft, pos2(90.0, 90.0), img);
        assert_eq!((out.x1, out.y1), (60.0, 60.0));
        assert_eq!((out.x2, out.y2), (90.0, 90.0));
    }

    #[test]
    fn a_resize_cannot_collapse_a_box_to_nothing() {
        let img = Vec2::new(100.0, 100.0);
        let origin = BBox::new(20.0, 20.0, 60.0, 60.0);
        let out = resized(&origin, Corner::TopLeft, pos2(60.0, 60.0), img);
        assert!(out.width() >= MIN_BOX, "{out:?}");
        assert!(out.height() >= MIN_BOX, "{out:?}");
        // Still inside the image.
        assert!(out.x2 <= img.x && out.y2 <= img.y);
    }

    #[test]
    fn a_resize_at_the_far_edge_keeps_the_minimum_inside_the_image() {
        let img = Vec2::new(100.0, 100.0);
        let origin = BBox::new(90.0, 90.0, 100.0, 100.0);
        let out = resized(&origin, Corner::TopLeft, pos2(100.0, 100.0), img);
        assert!(out.width() >= MIN_BOX && out.height() >= MIN_BOX, "{out:?}");
        assert!(out.x2 <= 100.0 && out.y2 <= 100.0, "{out:?}");
        assert!(out.x1 >= 0.0 && out.y1 >= 0.0, "{out:?}");
    }

    /// A checkerboard: pixelating or blurring a flat colour is a no-op, so a
    /// flat test image would make every style look like it did nothing.
    fn checkerboard(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = if (x / 4 + y / 4) % 2 == 0 { 240 } else { 20 };
                data[i] = v;
                data[i + 1] = v;
                data[i + 2] = v;
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    #[test]
    fn render_covers_exactly_the_boxes_that_are_visible() {
        let src = checkerboard(64, 64);
        let mut doc = doc_with(&[BBox::new(8.0, 8.0, 24.0, 24.0)]);
        doc.boxes[0].style = CensorStyle::SolidFill {
            color: [255, 0, 0, 255],
        };
        let out = render(&src, &doc).unwrap();

        let px = |f: &Frame, x: u32, y: u32| {
            let i = ((y * f.width + x) * 3) as usize;
            [f.data[i], f.data[i + 1], f.data[i + 2]]
        };
        assert_eq!(px(&out, 12, 12), [255, 0, 0]);
        // Outside the box, untouched — no padding is applied to a hand-drawn
        // region.
        assert_eq!(px(&out, 6, 6), px(&src, 6, 6));
        assert_eq!(px(&out, 30, 30), px(&src, 30, 30));

        // Hidden: the box stays in the document but paints nothing.
        doc.boxes[0].visible = false;
        let hidden = render(&src, &doc).unwrap();
        assert_eq!(px(&hidden, 12, 12), px(&src, 12, 12));
    }

    #[test]
    fn each_box_is_rendered_in_its_own_style() {
        let src = checkerboard(64, 32);
        let doc = EditDoc {
            boxes: vec![
                ManualBox::new(
                    1,
                    BBox::new(0.0, 0.0, 16.0, 16.0),
                    CensorStyle::SolidFill {
                        color: [0, 0, 255, 255],
                    },
                    0.0,
                ),
                ManualBox::new(
                    2,
                    BBox::new(32.0, 0.0, 48.0, 16.0),
                    CensorStyle::SolidFill {
                        color: [0, 255, 0, 255],
                    },
                    0.0,
                ),
            ],
        };
        let out = render(&src, &doc).unwrap();
        let px = |x: u32, y: u32| {
            let i = ((y * out.width + x) * 3) as usize;
            [out.data[i], out.data[i + 1], out.data[i + 2]]
        };
        assert_eq!(px(8, 8), [0, 0, 255]);
        assert_eq!(px(40, 8), [0, 255, 0]);
    }

    #[test]
    fn rendering_never_compounds_earlier_edits() {
        // The composite always starts from the decoded frame. Rendering twice
        // from the same source must give the same pixels, or a blur would get
        // heavier every time the canvas repainted.
        let src = checkerboard(48, 48);
        let mut doc = doc_with(&[BBox::new(4.0, 4.0, 40.0, 40.0)]);
        doc.boxes[0].style = CensorStyle::Blur { sigma: 4.0 };
        let once = render(&src, &doc).unwrap();
        let twice = render(&src, &doc).unwrap();
        assert_eq!(once.data, twice.data);
    }

    #[test]
    fn a_new_image_forgets_the_previous_documents_history() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(32, 32));
        state.add_box(BBox::new(1.0, 1.0, 9.0, 9.0));
        assert!(state.history.can_undo());

        state.load(PathBuf::from("b.png"), checkerboard(16, 16));
        assert!(state.doc().boxes.is_empty());
        // Undoing into the previous image's boxes would censor whatever happens
        // to be at those coordinates in the new one.
        assert!(!state.history.can_undo());
        assert!(state.selection().is_empty());
        assert_eq!(state.peek_next_id(), 1);
    }

    #[test]
    fn undo_drops_a_selection_that_no_longer_exists() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(32, 32));
        let id = state.add_box(BBox::new(1.0, 1.0, 9.0, 9.0));
        assert_eq!(state.selection(), [id]);
        state.undo();
        assert!(
            state.selection().is_empty(),
            "the selected box was undone away"
        );
        state.redo();
        // Ids are stable across a step, so the box can be selected again.
        assert_eq!(state.doc().boxes[0].id, id);
    }

    #[test]
    fn ctrl_click_adds_and_removes_without_disturbing_the_rest() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(64, 64));
        let a = state.add_box(BBox::new(0.0, 0.0, 10.0, 10.0));
        let b = state.add_box(BBox::new(20.0, 20.0, 30.0, 30.0));
        let c = state.add_box(BBox::new(40.0, 40.0, 50.0, 50.0));
        // Drawing selects what was drawn, alone.
        assert_eq!(state.selection(), [c]);

        state.toggle_selected(a);
        assert_eq!(state.selection(), [c, a]);
        // The controls follow the box picked most recently.
        assert_eq!(state.selected_box().map(|x| x.id), Some(a));

        state.toggle_selected(b);
        assert_eq!(state.selection(), [c, a, b]);
        state.toggle_selected(a);
        assert_eq!(state.selection(), [c, b], "ctrl-click again takes it out");
        state.select_only(Some(a));
        assert_eq!(state.selection(), [a]);
    }

    #[test]
    fn a_style_change_lands_on_every_selected_box_in_one_step() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(64, 64));
        let a = state.add_box(BBox::new(0.0, 0.0, 10.0, 10.0));
        let b = state.add_box(BBox::new(20.0, 20.0, 30.0, 30.0));
        let c = state.add_box(BBox::new(40.0, 40.0, 50.0, 50.0));
        state.select_only(Some(a));
        state.toggle_selected(b);

        let blur = CensorStyle::Blur { sigma: 7.0 };
        let now = t0();
        state.edit_selection("style", now, |x| x.style = blur.clone());
        assert_eq!(state.doc().find(a).unwrap().style, blur);
        assert_eq!(state.doc().find(b).unwrap().style, blur);
        assert_eq!(
            state.doc().find(c).unwrap().style,
            CensorStyle::default(),
            "a box nobody selected must be left alone"
        );

        // A slider sweep over the same selection is one step...
        state.edit_selection("style", now + Duration::from_millis(16), |x| {
            x.style = CensorStyle::Blur { sigma: 9.0 }
        });
        state.undo();
        assert_eq!(state.doc().find(a).unwrap().style, CensorStyle::default());
        assert_eq!(state.doc().find(b).unwrap().style, CensorStyle::default());
    }

    #[test]
    fn changing_the_selection_starts_a_new_undo_step() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(64, 64));
        let a = state.add_box(BBox::new(0.0, 0.0, 10.0, 10.0));
        let b = state.add_box(BBox::new(20.0, 20.0, 30.0, 30.0));
        let now = t0();
        state.select_only(Some(a));
        state.edit_selection("rounding", now, |x| x.rounding = 0.3);
        // Same control, different boxes, inside the coalescing window: folding
        // these together would make one Ctrl-Z undo an edit to boxes the user
        // had already moved on from.
        state.select_only(Some(b));
        state.edit_selection("rounding", now + Duration::from_millis(10), |x| {
            x.rounding = 0.4
        });
        state.undo();
        assert_eq!(state.doc().find(b).unwrap().rounding, 0.0);
        assert_eq!(state.doc().find(a).unwrap().rounding, 0.3);
    }

    #[test]
    fn rounding_belongs_to_the_box_it_was_set_on() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(64, 64));
        let first = state.add_box(BBox::new(0.0, 0.0, 10.0, 10.0));
        // Set the picker before drawing: the next box takes it, the one already
        // drawn does not. A rounding that reshaped every box ever drawn was the
        // one control here that did not act on the selection.
        state.new_rounding = 0.5;
        let second = state.add_box(BBox::new(20.0, 20.0, 30.0, 30.0));
        assert_eq!(state.doc().find(first).unwrap().rounding, 0.0);
        assert_eq!(state.doc().find(second).unwrap().rounding, 0.5);

        state.select_only(Some(first));
        state.edit_selection("rounding", t0(), |x| x.rounding = 0.25);
        assert_eq!(state.doc().find(first).unwrap().rounding, 0.25);
        assert_eq!(state.doc().find(second).unwrap().rounding, 0.5);
    }

    #[test]
    fn each_box_is_rendered_with_its_own_rounding() {
        let src = checkerboard(64, 32);
        let doc = EditDoc {
            boxes: vec![
                // Square: the corner pixel is covered.
                ManualBox::new(
                    1,
                    BBox::new(0.0, 0.0, 24.0, 24.0),
                    CensorStyle::SolidFill {
                        color: [255, 0, 0, 255],
                    },
                    0.0,
                ),
                // Rounded all the way: the corner pixel is left as it was.
                ManualBox::new(
                    2,
                    BBox::new(32.0, 0.0, 56.0, 24.0),
                    CensorStyle::SolidFill {
                        color: [255, 0, 0, 255],
                    },
                    0.5,
                ),
            ],
        };
        let out = render(&src, &doc).unwrap();
        let px = |f: &Frame, x: u32, y: u32| {
            let i = ((y * f.width + x) * 3) as usize;
            [f.data[i], f.data[i + 1], f.data[i + 2]]
        };
        assert_eq!(px(&out, 0, 0), [255, 0, 0]);
        assert_eq!(px(&out, 32, 0), px(&src, 32, 0));
        // Both are still filled in the middle.
        assert_eq!(px(&out, 12, 12), [255, 0, 0]);
        assert_eq!(px(&out, 44, 12), [255, 0, 0]);
    }

    #[test]
    fn a_group_move_keeps_the_selection_rigid() {
        let img = Vec2::new(100.0, 100.0);
        let origins = vec![
            (1, BBox::new(0.0, 0.0, 10.0, 10.0)),
            (2, BBox::new(20.0, 20.0, 30.0, 30.0)),
        ];
        let out = moved_group(&origins, Vec2::new(5.0, 5.0), img);
        assert_eq!(out[0].1, BBox::new(5.0, 5.0, 15.0, 15.0));
        assert_eq!(out[1].1, BBox::new(25.0, 25.0, 35.0, 35.0));

        // Shoved into the corner: the group stops as one shape, keeping its
        // internal spacing, rather than piling up as members hit the edge.
        let far = moved_group(&origins, Vec2::new(500.0, 500.0), img);
        assert_eq!(far[1].1, BBox::new(90.0, 90.0, 100.0, 100.0));
        assert_eq!(far[0].1, BBox::new(70.0, 70.0, 80.0, 80.0));
    }

    #[test]
    fn delete_takes_the_whole_selection_in_one_step() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(64, 64));
        let a = state.add_box(BBox::new(0.0, 0.0, 10.0, 10.0));
        let b = state.add_box(BBox::new(20.0, 20.0, 30.0, 30.0));
        let c = state.add_box(BBox::new(40.0, 40.0, 50.0, 50.0));
        state.select_only(Some(a));
        state.toggle_selected(c);
        state.delete_selected();
        assert_eq!(
            state.doc().boxes.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![b]
        );
        assert!(state.selection().is_empty());
        state.undo();
        assert_eq!(
            state.doc().boxes.len(),
            3,
            "one Ctrl-Z brings them all back"
        );
    }

    #[test]
    fn deleting_the_selected_box_clears_the_selection() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(32, 32));
        let a = state.add_box(BBox::new(1.0, 1.0, 9.0, 9.0));
        let b = state.add_box(BBox::new(10.0, 10.0, 20.0, 20.0));
        state.select_only(Some(a));
        state.delete(a);
        assert!(state.selection().is_empty());
        assert_eq!(state.doc().boxes.len(), 1);
        assert_eq!(state.doc().boxes[0].id, b);
        // And it is undoable.
        state.undo();
        assert_eq!(state.doc().boxes.len(), 2);
    }

    #[test]
    fn reordering_moves_a_box_through_the_stack() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(32, 32));
        let a = state.add_box(BBox::new(0.0, 0.0, 8.0, 8.0));
        let b = state.add_box(BBox::new(0.0, 0.0, 8.0, 8.0));
        state.reorder(a, true);
        assert_eq!(
            state.doc().boxes.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![b, a]
        );
        state.reorder(a, false);
        assert_eq!(
            state.doc().boxes.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![a, b]
        );
        // Already at the bottom: a no-op rather than a wrap-around.
        state.reorder(a, false);
        assert_eq!(
            state.doc().boxes.iter().map(|x| x.id).collect::<Vec<_>>(),
            vec![a, b]
        );
    }

    #[test]
    fn the_fingerprint_moves_only_when_the_pixels_would() {
        let mut state = EditorState::default();
        state.load(PathBuf::from("a.png"), checkerboard(32, 32));
        let id = state.add_box(BBox::new(1.0, 1.0, 9.0, 9.0));
        let before = state.doc().fingerprint();

        // Selection is not part of the document: a re-censor here would be
        // pure waste on every click.
        state.select_only(None);
        assert_eq!(state.doc().fingerprint(), before);

        state.history.step().find_mut(id).unwrap().visible = false;
        assert_ne!(state.doc().fingerprint(), before);
        state.undo();
        assert_eq!(state.doc().fingerprint(), before);
    }
}
