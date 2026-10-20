//! The Edit page: censor one image by hand.
//!
//! The detector is the whole point of the product, but it is not always right
//! and it is not always what is wanted — a frame it missed, a licence plate or
//! a face no model here is trained to find, one region of a hundred that needs
//! a different treatment. This page is the manual fallback: load an image, drag
//! boxes over it, give each one a style, save the result.
//!
//! It shares the censor path with the batch (`ob_censor::apply`, via
//! [`crate::edit::render`]) and the style picker with the Tuning page, so a box
//! drawn here is covered by exactly the code a run would use, and the controls
//! behave the same way in both places.

use crate::app::{ObApp, ToastKind};
use crate::edit::{self, Corner, Drag, Grab, View};
use crate::pages;
use crate::theme;
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Rounding, Sense, Stroke, Vec2};
use ob_core::geometry::{BBox, Frame};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Width of the tools column. Wide enough for the style editor's widest row
/// (the overlay file picker) without taking the canvas below half the window.
const TOOLS_WIDTH: f32 = 320.0;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    pages::header(
        ui,
        "Edit",
        "Censor one image by hand: drag boxes onto it, style each one, save the result.",
    );

    shortcuts(app, ui);

    egui::SidePanel::right("editor-tools")
        .resizable(true)
        .default_width(TOOLS_WIDTH)
        .min_width(260.0)
        .max_width((ui.available_width() - 280.0).max(280.0))
        .frame(theme::card_frame().inner_margin(egui::Margin::same(12.0)))
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("editor-tools-scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| tools(app, ui));
        });

    // A CentralPanel for the rest, not the shrunken `ui`: see the scrollbar
    // note in `pages::tuning`.
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show_inside(ui, |ui| canvas(app, ui));
}

/// Ctrl-Z / Ctrl-Y, and Delete for the selected box.
///
/// Consumed here rather than watched globally so they only act while this page
/// is on screen, and so egui does not also hand them to a widget.
fn shortcuts(app: &mut ObApp, ui: &mut egui::Ui) {
    use egui::{Key, Modifiers};
    let redo_mods = Modifiers {
        command: true,
        shift: true,
        ..Default::default()
    };
    // Most specific first: egui's shortcut matching ignores an *extra* shift,
    // so a plain Ctrl-Z check would otherwise swallow Ctrl-Shift-Z and undo
    // when the user asked to redo.
    let (undo, redo) = ui.input_mut(|i| {
        let r = i.consume_key(redo_mods, Key::Z) | i.consume_key(Modifiers::COMMAND, Key::Y);
        let u = i.consume_key(Modifiers::COMMAND, Key::Z);
        (u, r)
    });
    if undo {
        app.editor.undo();
    }
    if redo {
        app.editor.redo();
    }

    if !app.editor.selection().is_empty() {
        let pressed = ui.input_mut(|i| {
            i.consume_key(Modifiers::NONE, Key::Delete)
                | i.consume_key(Modifiers::NONE, Key::Backspace)
        });
        if pressed {
            app.editor.delete_selected();
        }
    }
}

// --- the tools column ---------------------------------------------------

fn tools(app: &mut ObApp, ui: &mut egui::Ui) {
    image_card(app, ui);
    ui.add_space(10.0);
    style_card(app, ui);
    ui.add_space(10.0);
    boxes_card(app, ui);
}

fn image_card(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Image");
        ui.horizontal(|ui| {
            if ui.button("Open image…").clicked() {
                open_dialog(app);
            }
            let loaded = app.editor.source.is_some();
            if ui
                .add_enabled(loaded, egui::Button::new("Save as…"))
                .on_hover_text("Write the censored image to a new file.")
                .on_disabled_hover_text("Open an image first.")
                .clicked()
            {
                save_dialog(app);
            }
        });

        match (&app.editor.path, &app.editor.source) {
            (Some(path), Some(frame)) => {
                ui.add_space(4.0);
                ui.label(RichText::new(file_name(path)).size(12.0).color(p.text_dim))
                    .on_hover_text(path.display().to_string());
                ui.label(
                    RichText::new(format!(
                        "{}{}{} px",
                        frame.width,
                        theme::glyph::TIMES,
                        frame.height
                    ))
                    .size(11.5)
                    .color(p.text_faint),
                );
            }
            _ => {
                ui.add_space(4.0);
                theme::hint(
                    ui,
                    "Nothing loaded. Open an image, or drop one on the window.",
                );
            }
        }

        if let Some(err) = &app.editor.error {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("{} {err}", theme::glyph::FAIL))
                    .size(12.0)
                    .color(p.danger),
            );
        }
    });
}

/// The style editor, pointed either at the selected box or at the style new
/// boxes will get.
///
/// One editor rather than two: a second copy of the same four styles would be
/// twice the screen and twice the chance of them drifting apart.
fn style_card(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        // Via `selected_box` rather than a raw id: an undo can remove a
        // selected box, and the editor must then fall back to the new-box
        // values instead of pointing at nothing.
        let primary = app.editor.selected_box().map(|b| b.id);
        let picked = app.editor.selection().len();
        match (primary, picked) {
            (Some(id), 1) => theme::section(ui, &format!("Style · box {id}")),
            (Some(id), n) => theme::section(ui, &format!("Style · {n} boxes, showing box {id}")),
            (None, _) => theme::section(ui, "Style for new boxes"),
        }
        match picked {
            0 => {
                theme::hint(
                    ui,
                    "No box selected. This is what the next box you draw will use. \
                     Click a box to restyle it, Ctrl-click to add more.",
                );
                ui.add_space(4.0);
            }
            1 => {}
            n => {
                theme::hint(
                    ui,
                    &format!("Every change here lands on all {n} selected boxes."),
                );
                ui.add_space(4.0);
            }
        }

        // Edit a copy and commit only a real change: `style_controls` draws
        // several widgets and cannot report "changed" for all of them, and an
        // unconditional write would open an undo step on every frame the page
        // is visible.
        let mut style = match app.editor.selected_box() {
            Some(b) => b.style.clone(),
            None => app.editor.new_style.clone(),
        };
        let before = style.clone();
        pages::tuning::style_controls(ui, &mut style, "editor");
        if style != before {
            if primary.is_some() {
                // Coalesced per selection: dragging the blur slider is one undo
                // step, not one per frame of the drag.
                let chosen = style.clone();
                app.editor
                    .edit_selection("style", Instant::now(), |b| b.style = chosen.clone());
            } else {
                app.editor.new_style = style;
            }
        }

        ui.add_space(6.0);
        let mut rounding = match app.editor.selected_box() {
            Some(b) => b.rounding,
            None => app.editor.new_rounding,
        };
        let resp = ui
            .add(egui::Slider::new(&mut rounding, 0.0..=0.5).text("Corner rounding"))
            .on_hover_text(
                "Round this box's corners by that fraction of its shorter side \
                 (0 = square). Hand-drawn boxes get no padding: the box is \
                 exactly what you dragged.",
            );
        if resp.changed() {
            if primary.is_some() {
                app.editor
                    .edit_selection("rounding", Instant::now(), |b| b.rounding = rounding);
            } else {
                app.editor.new_rounding = rounding;
            }
        }
    });
}

/// The box list: the stack, top layer first.
fn boxes_card(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let count = app.editor.doc().boxes.len();
        theme::section(ui, &format!("Boxes ({count})"));

        ui.horizontal(|ui| {
            let can_undo = app.editor.history.can_undo();
            let can_redo = app.editor.history.can_redo();
            if ui
                .add_enabled(can_undo, egui::Button::new("Undo"))
                .on_hover_text("Ctrl-Z")
                .clicked()
            {
                app.editor.undo();
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("Redo"))
                .on_hover_text("Ctrl-Y, or Ctrl-Shift-Z")
                .clicked()
            {
                app.editor.redo();
            }
            if ui
                .add_enabled(count > 0, egui::Button::new("Clear"))
                .on_hover_text("Remove every box. Undoable.")
                .clicked()
            {
                app.editor.history.step().boxes.clear();
                app.editor.select_only(None);
            }
        });
        ui.add_space(6.0);

        if count == 0 {
            theme::hint(ui, "Drag on the image to draw a box.");
            return;
        }

        // Top of the stack first, like a layers panel: the box that paints over
        // the others is the one listed above them.
        let rows: Vec<(u64, String, bool)> = app
            .editor
            .doc()
            .boxes
            .iter()
            .rev()
            .map(|b| (b.id, b.label(), b.visible))
            .collect();
        egui::ScrollArea::vertical()
            .id_salt("editor-box-list")
            .max_height(260.0)
            .show(ui, |ui| {
                for (id, label, visible) in rows {
                    ui.horizontal(|ui| {
                        let mut shown = visible;
                        if ui
                            .checkbox(&mut shown, "")
                            .on_hover_text("Draw this box. Hiding it keeps it in the list.")
                            .changed()
                        {
                            if let Some(b) = app.editor.history.step().find_mut(id) {
                                b.visible = shown;
                            }
                        }
                        let text = if visible {
                            RichText::new(&label).size(12.0)
                        } else {
                            RichText::new(&label).size(12.0).color(p.text_faint)
                        };
                        let resp = ui
                            .selectable_label(app.editor.is_selected(id), text)
                            .on_hover_text("Click to select, Ctrl-click to add to the selection");
                        if resp.clicked() {
                            if ui.input(|i| i.modifiers.command) {
                                app.editor.toggle_selected(id);
                            } else if app.editor.selection() == [id] {
                                // Clicking the only selected row clears it,
                                // which is how the style editor gets back to
                                // "new boxes".
                                app.editor.select_only(None);
                            } else {
                                app.editor.select_only(Some(id));
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button(theme::glyph::REMOVE)
                                .on_hover_text("Delete this box")
                                .clicked()
                            {
                                app.editor.delete(id);
                            }
                        });
                    });
                }
            });
        ui.add_space(4.0);
        // Reordering acts on the selection rather than giving every row its own
        // pair of arrows: the glyphs egui's bundled fonts cover do not include
        // an up or a down triangle (`theme::glyph` has the list), and a row of
        // tiny text buttons per box would crowd out the names.
        if let Some(id) = app.editor.selected_box().map(|b| b.id) {
            let at = app.editor.doc().position(id).unwrap_or(0);
            let top = at + 1 == count;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!top, egui::Button::new("Raise").small())
                    .on_hover_text("Paint this box over the others")
                    .clicked()
                {
                    app.editor.reorder(id, true);
                }
                if ui
                    .add_enabled(at > 0, egui::Button::new("Lower").small())
                    .on_hover_text("Paint this box under the others")
                    .clicked()
                {
                    app.editor.reorder(id, false);
                }
            });
            ui.add_space(4.0);
        }
        theme::hint(
            ui,
            "Boxes paint bottom-up, so where two overlap the one higher in this \
             list wins.",
        );
    });
}

// --- the canvas ---------------------------------------------------------

fn canvas(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let Some(img) = app.editor.image_size() else {
        empty_state(app, ui);
        return;
    };

    refresh_composite(app, ui.ctx());

    let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let view = View::fit(img.x as u32, img.y as u32, rect);

    // Input before paint, so a drag draws at the pointer rather than one frame
    // behind it.
    interact(app, ui, &resp, view, img);
    paint(app, ui, rect, view);

    // The gesture that just finished changed the document after this frame's
    // composite was skipped, so ask for the frame that will catch it up. An
    // idle window would otherwise show the new outline over the old pixels
    // until something else happened to cause a repaint.
    if app.editor.rendered != Some(app.editor.doc().fingerprint()) {
        ui.ctx().request_repaint();
    }

    if app.editor.doc().boxes.is_empty() {
        ui.painter_at(rect).text(
            rect.center_bottom() - Vec2::new(0.0, 14.0),
            Align2::CENTER_BOTTOM,
            "Drag to draw a censor box",
            FontId::proportional(12.5),
            p.text_faint,
        );
    }
}

fn empty_state(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    ui.vertical_centered(|ui| {
        ui.add_space(48.0);
        ui.label(
            RichText::new("No image open")
                .size(14.0)
                .color(p.text_dim)
                .strong(),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Open a picture to draw censor boxes on it by hand, then save \
                 the result. Nothing here touches the batch queue.",
            )
            .size(12.0)
            .color(p.text_faint),
        );
        ui.add_space(12.0);
        if theme::primary_button(ui, "Open image…", true).clicked() {
            open_dialog(app);
        }
    });
}

/// Re-censor the frame when the document has moved.
///
/// Skipped while the pointer is down: a gesture changes the document every
/// frame, and re-blurring a 4K frame sixty times a second would make the drag
/// itself unusable. The live outlines carry the geometry in the meantime, and
/// the pixels catch up the moment the button is released.
fn refresh_composite(app: &mut ObApp, ctx: &egui::Context) {
    let key = app.editor.doc().fingerprint();
    if app.editor.rendered == Some(key) {
        return;
    }
    let busy = app.editor.drag.is_some() || ctx.is_using_pointer();
    if busy && app.editor.texture.is_some() {
        return;
    }
    match app.editor.composite() {
        Some(Ok(frame)) => {
            let image = crate::app::color_image(&frame);
            match &mut app.editor.texture {
                Some(tex) => tex.set(image, egui::TextureOptions::LINEAR),
                None => {
                    app.editor.texture =
                        Some(ctx.load_texture("ob-editor", image, egui::TextureOptions::LINEAR))
                }
            }
            app.editor.rendered = Some(key);
        }
        Some(Err(e)) => {
            // Mark the attempt spent as well as reporting it: a document that
            // will not render — an overlay image that has gone missing, say —
            // must not be retried on every frame forever.
            app.editor.error = Some(e.to_string());
            app.editor.rendered = Some(key);
        }
        None => {}
    }
}

fn interact(app: &mut ObApp, ui: &egui::Ui, resp: &egui::Response, view: View, img: Vec2) {
    let now = Instant::now();
    let grab = view.grab_in_image();
    let add = ui.input(|i| i.modifiers.command);

    if let Some(screen) = resp.interact_pointer_pos() {
        let at = view.to_image(screen);
        if resp.drag_started() {
            match edit::hit(app.editor.doc(), at, app.editor.selection(), grab) {
                Some((id, grab)) => {
                    // Dragging a box that is not part of the selection takes
                    // the selection over; dragging one that is keeps it, so a
                    // group can be moved as a group.
                    if !app.editor.is_selected(id) {
                        app.editor.select_only(Some(id));
                    }
                    let origins = match grab {
                        Grab::Body => app.editor.selected_rects(),
                        // A resize acts on the box actually grabbed: there is
                        // no one obvious meaning for resizing four at once.
                        Grab::Corner(_) => app
                            .editor
                            .doc()
                            .find(id)
                            .map(|b| vec![(id, b.rect)])
                            .unwrap_or_default(),
                    };
                    app.editor.drag = Some(Drag::Existing {
                        id,
                        grab,
                        anchor: at,
                        origins,
                    });
                }
                None => {
                    app.editor.select_only(None);
                    app.editor.drag = Some(Drag::New {
                        anchor: at,
                        current: at,
                    });
                }
            }
        } else if resp.dragged() {
            match app.editor.drag.clone() {
                Some(Drag::New { anchor, .. }) => {
                    app.editor.drag = Some(Drag::New {
                        anchor,
                        current: at,
                    });
                }
                Some(Drag::Existing {
                    id,
                    grab,
                    anchor,
                    origins,
                }) => {
                    let moved = match grab {
                        Grab::Body => edit::moved_group(&origins, at - anchor, img),
                        Grab::Corner(c) => origins
                            .iter()
                            .map(|(bid, origin)| (*bid, edit::resized(origin, c, at, img)))
                            .collect(),
                    };
                    // One undo step per gesture, keyed by what it has hold of.
                    let doc = app.editor.history.stroke(&format!("box:{id}"), now);
                    for (bid, rect) in moved {
                        if let Some(b) = doc.find_mut(bid) {
                            b.rect = rect;
                        }
                    }
                }
                None => {}
            }
        }
        if resp.clicked() {
            // A press that never moved. Ctrl-click adds to or removes from the
            // selection; a plain click replaces it, or clears it on empty
            // canvas.
            match edit::hit(app.editor.doc(), at, app.editor.selection(), grab) {
                Some((id, _)) if add => app.editor.toggle_selected(id),
                Some((id, _)) => app.editor.select_only(Some(id)),
                // Ctrl-click on empty canvas keeps what is picked: it reads as
                // a miss, not as "deselect everything".
                None if add => {}
                None => app.editor.select_only(None),
            }
        }
    }

    if resp.drag_stopped() {
        if let Some(Drag::New { anchor, current }) = app.editor.drag {
            if let Some(rect) = edit::from_drag(anchor, current, img) {
                app.editor.add_box(rect);
            }
        }
        app.editor.drag = None;
    }

    // Say what a press would do before it is made.
    if resp.hovered() && app.editor.drag.is_none() {
        if let Some(screen) = ui.input(|i| i.pointer.hover_pos()) {
            let at = view.to_image(screen);
            let icon = match edit::hit(app.editor.doc(), at, app.editor.selection(), grab) {
                Some((_, Grab::Corner(Corner::TopLeft | Corner::BottomRight))) => {
                    egui::CursorIcon::ResizeNwSe
                }
                Some((_, Grab::Corner(Corner::TopRight | Corner::BottomLeft))) => {
                    egui::CursorIcon::ResizeNeSw
                }
                Some((_, Grab::Body)) => egui::CursorIcon::Grab,
                None => egui::CursorIcon::Crosshair,
            };
            ui.ctx().set_cursor_icon(icon);
        }
    }
}

fn paint(app: &ObApp, ui: &egui::Ui, rect: Rect, view: View) {
    let p = theme::palette();
    let painter = ui.painter_at(rect);
    // A ground behind the image, so a letterboxed picture reads as a canvas
    // rather than as a floating fragment.
    painter.rect_filled(rect, Rounding::same(theme::CARD_RADIUS), p.panel);
    if let Some(tex) = &app.editor.texture {
        painter.image(
            tex.id(),
            view.rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    for b in &app.editor.doc().boxes {
        let r = view.to_screen(&b.rect).intersect(view.rect);
        let selected = app.editor.is_selected(b.id);
        let color = match (selected, b.visible) {
            (true, _) => p.accent,
            (false, true) => p.accent_hover.gamma_multiply(0.85),
            (false, false) => p.text_faint,
        };
        painter.rect_stroke(
            r,
            Rounding::ZERO,
            Stroke::new(if selected { 2.0_f32 } else { 1.5_f32 }, color),
        );
        // The number ties the outline to its row in the list.
        painter.text(
            r.min + Vec2::new(3.0, 2.0),
            Align2::LEFT_TOP,
            b.id.to_string(),
            FontId::proportional(11.0),
            color,
        );
        if selected {
            for corner in Corner::ALL {
                let c = corner.of(&b.rect);
                let at = view.rect.min + Vec2::new(c.x, c.y) * view.scale;
                let h = Rect::from_center_size(at, Vec2::splat(edit::HANDLE_PX));
                painter.rect_filled(h, Rounding::same(1.0), p.accent);
                painter.rect_stroke(h, Rounding::same(1.0), Stroke::new(1.0_f32, p.on_accent));
            }
        }
    }

    // The box being drawn right now, which is not in the document yet.
    if let Some(Drag::New { anchor, current }) = app.editor.drag {
        let band = BBox::new(
            anchor.x.min(current.x),
            anchor.y.min(current.y),
            anchor.x.max(current.x),
            anchor.y.max(current.y),
        );
        painter.rect_stroke(
            view.to_screen(&band).intersect(view.rect),
            Rounding::ZERO,
            Stroke::new(1.5_f32, p.accent),
        );
    }
}

// --- loading and saving -------------------------------------------------

fn open_dialog(app: &mut ObApp) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("Images", ob_media::IMAGE_EXTS)
        .pick_file()
    else {
        return;
    };
    load(app, path);
}

/// Decode `path` into the editor, replacing whatever was open.
///
/// Public to the crate because dropping a file on the window while this page is
/// up loads it here instead of adding it to the batch.
pub fn load(app: &mut ObApp, path: PathBuf) {
    match ob_media::load_image(&path) {
        Ok(frame) => {
            let name = file_name(&path);
            app.editor.load(path, frame);
            app.toast(format!("Opened {name}"), ToastKind::Info);
        }
        Err(e) => {
            app.editor.error = Some(e.to_string());
            app.toast(format!("Could not open that image: {e}"), ToastKind::Error);
        }
    }
}

fn save_dialog(app: &mut ObApp) {
    let Some(Ok(frame)) = app.editor.composite() else {
        // A composite that will not render is reported where it happened, by
        // `refresh_composite`; nothing useful to add here.
        return;
    };
    let suggested = app
        .editor
        .path
        .as_deref()
        .and_then(default_name)
        .unwrap_or_else(|| "edited.png".to_string());
    let Some(path) = rfd::FileDialog::new()
        .set_file_name(suggested)
        .add_filter("PNG", &["png"])
        .add_filter("JPEG", &["jpg", "jpeg"])
        .add_filter("WebP", &["webp"])
        .save_file()
    else {
        return;
    };
    save(app, frame, &path);
}

/// Write `frame` to `path`, reporting either way.
///
/// Takes the frame by value: the composite was built for this save and nobody
/// else holds it, so the encoder can have the buffer instead of a full-
/// resolution copy of it.
fn save(app: &mut ObApp, frame: Frame, path: &Path) {
    match ob_media::save_image_owned(frame, path) {
        Ok(()) => {
            app.editor.error = None;
            app.toast(format!("Saved {}", path.display()), ToastKind::Success);
        }
        Err(e) => {
            app.editor.error = Some(e.to_string());
            app.toast(format!("Could not save: {e}"), ToastKind::Error);
        }
    }
}

/// `beach.jpg` → `beach-censored.png`.
///
/// Always PNG, whatever came in: the source may be a JPEG, and re-encoding a
/// censored frame as lossy JPEG can leave ghosts of a pixelated region's edges.
/// Never the input's own name — a manual edit must not silently overwrite the
/// original.
fn default_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy();
    Some(format!("{stem}-censored.png"))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// True when this page should take a dropped file instead of the batch queue.
///
/// Images only. A video dropped here has no manual-edit path — there is one
/// canvas and one frame — so it belongs in the queue, which is where a drop on
/// any other page sends it.
pub fn wants_drop(path: &Path) -> bool {
    matches!(ob_media::classify(path), ob_media::MediaKind::Image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::test_app;
    use crate::prefs::Tab;
    use egui::{Event, Key, Modifiers, PointerButton, RawInput};
    use ob_core::censor::CensorStyle;

    const WIN: Vec2 = Vec2::new(1180.0, 820.0);

    /// A checkerboard, so a pixelate or a blur has something to destroy.
    fn test_frame(w: u32, h: u32) -> Frame {
        let mut data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v = if (x / 8 + y / 8) % 2 == 0 { 230 } else { 25 };
                data[i] = v;
                data[i + 1] = v;
                data[i + 2] = v;
            }
        }
        Frame::new(w, h, data).unwrap()
    }

    /// An app sitting on the Edit page with an image already open.
    fn editing(name: &str) -> (ObApp, crate::app::tests::TestDir, egui::Context) {
        let (mut app, dir) = test_app(name);
        app.show_setup = false;
        app.prefs.tab = Tab::Edit;
        app.editor
            .load(PathBuf::from("sample.png"), test_frame(400, 300));
        let ctx = egui::Context::default();
        crate::theme::install(&ctx);
        (app, dir, ctx)
    }

    fn run(app: &mut ObApp, ctx: &egui::Context, events: Vec<Event>) {
        run_held(app, ctx, events, Modifiers::NONE);
    }

    /// One frame with `held` down. The modifiers a widget reads come from
    /// `RawInput::modifiers`, not from the events — an integration reports the
    /// keyboard's state alongside them — so a test that only decorated its
    /// events would see no Ctrl at all.
    fn run_held(app: &mut ObApp, ctx: &egui::Context, events: Vec<Event>, held: Modifiers) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, WIN)),
            modifiers: held,
            events,
            ..Default::default()
        };
        let _ = ctx.run(input, |c| app.ui(c));
    }

    fn press(pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed,
            modifiers: Modifiers::NONE,
        }
    }

    fn key(key: Key, modifiers: Modifiers) -> Vec<Event> {
        vec![
            Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            },
            Event::Key {
                key,
                physical_key: None,
                pressed: false,
                repeat: false,
                modifiers,
            },
        ]
    }

    /// Drag the pointer from `from` to `to` across the frames egui needs to see
    /// it as a drag rather than a click.
    fn drag(app: &mut ObApp, ctx: &egui::Context, from: Pos2, to: Pos2) {
        run(app, ctx, vec![Event::PointerMoved(from)]);
        run(app, ctx, vec![press(from, true)]);
        // Several intermediate moves: egui only calls it a drag once the
        // pointer has travelled far enough to rule out a click.
        for t in [0.25_f32, 0.5, 0.75, 1.0] {
            let at = from + (to - from) * t;
            run(app, ctx, vec![Event::PointerMoved(at)]);
        }
        run(app, ctx, vec![press(to, false)]);
        run(app, ctx, vec![]);
    }

    /// A point inside the canvas: the window less the nav rail, the tools
    /// column and the chrome above and below.
    fn on_canvas(dx: f32, dy: f32) -> Pos2 {
        Pos2::new(260.0 + dx, 160.0 + dy)
    }

    #[test]
    fn the_page_lays_out_with_an_image_and_boxes_on_it() {
        let (mut app, _dir, ctx) = editing("editor-layout");
        app.editor.add_box(BBox::new(10.0, 10.0, 120.0, 90.0));
        app.editor.add_box(BBox::new(200.0, 150.0, 380.0, 280.0));
        app.editor.new_style = CensorStyle::Blur { sigma: 6.0 };
        run(&mut app, &ctx, vec![]);
        run(&mut app, &ctx, vec![]);
        // The composite is built and uploaded, and matches the document.
        assert!(app.editor.texture.is_some());
        assert_eq!(app.editor.rendered, Some(app.editor.doc().fingerprint()));
    }

    #[test]
    fn dragging_on_the_canvas_draws_a_box() {
        let (mut app, _dir, ctx) = editing("editor-draw");
        drag(
            &mut app,
            &ctx,
            on_canvas(60.0, 60.0),
            on_canvas(220.0, 180.0),
        );

        assert_eq!(app.editor.doc().boxes.len(), 1, "the drag drew no box");
        let b = &app.editor.doc().boxes[0];
        let (iw, ih) = (400.0, 300.0);
        assert!(
            b.rect.width() > 1.0 && b.rect.height() > 1.0,
            "{:?}",
            b.rect
        );
        // Inside the image, whatever the on-screen scale worked out to be.
        assert!(b.rect.x1 >= 0.0 && b.rect.x2 <= iw, "{:?}", b.rect);
        assert!(b.rect.y1 >= 0.0 && b.rect.y2 <= ih, "{:?}", b.rect);
        // Drawing selects what was drawn, so the style editor points at it.
        assert_eq!(app.editor.selection(), [b.id]);
        // And the pixels caught up with the gesture: the composite is skipped
        // while the button is down, so this only holds if the page asked for
        // the frame that finishes the job.
        assert_eq!(
            app.editor.rendered,
            Some(app.editor.doc().fingerprint()),
            "the censored image is still showing the document from before the drag"
        );
    }

    #[test]
    fn the_new_box_takes_the_style_the_picker_is_showing() {
        let (mut app, _dir, ctx) = editing("editor-style");
        app.editor.new_style = CensorStyle::SolidFill {
            color: [1, 2, 3, 255],
        };
        drag(
            &mut app,
            &ctx,
            on_canvas(40.0, 40.0),
            on_canvas(200.0, 160.0),
        );
        assert_eq!(
            app.editor.doc().boxes[0].style,
            CensorStyle::SolidFill {
                color: [1, 2, 3, 255]
            }
        );
    }

    #[test]
    fn ctrl_z_and_ctrl_y_walk_the_drawing_back_and_forward() {
        let (mut app, _dir, ctx) = editing("editor-undo");
        drag(
            &mut app,
            &ctx,
            on_canvas(40.0, 40.0),
            on_canvas(160.0, 140.0),
        );
        drag(
            &mut app,
            &ctx,
            on_canvas(200.0, 40.0),
            on_canvas(320.0, 140.0),
        );
        assert_eq!(app.editor.doc().boxes.len(), 2);

        run(&mut app, &ctx, key(Key::Z, Modifiers::COMMAND));
        assert_eq!(app.editor.doc().boxes.len(), 1, "Ctrl-Z undid nothing");
        run(&mut app, &ctx, key(Key::Z, Modifiers::COMMAND));
        assert_eq!(app.editor.doc().boxes.len(), 0);

        run(&mut app, &ctx, key(Key::Y, Modifiers::COMMAND));
        assert_eq!(app.editor.doc().boxes.len(), 1, "Ctrl-Y redid nothing");
        // Ctrl-Shift-Z is the other spelling of redo, and must not be read as
        // an undo: egui's shortcut matching ignores an extra shift.
        run(
            &mut app,
            &ctx,
            key(
                Key::Z,
                Modifiers {
                    command: true,
                    shift: true,
                    ..Default::default()
                },
            ),
        );
        assert_eq!(app.editor.doc().boxes.len(), 2);
    }

    /// Press and release at one spot, with modifiers, without moving: a click.
    fn click(app: &mut ObApp, ctx: &egui::Context, at: Pos2, modifiers: Modifiers) {
        run_held(app, ctx, vec![Event::PointerMoved(at)], modifiers);
        for pressed in [true, false] {
            run_held(
                app,
                ctx,
                vec![Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed,
                    modifiers,
                }],
                modifiers,
            );
        }
    }

    #[test]
    fn ctrl_click_adds_a_box_to_the_selection() {
        let (mut app, _dir, ctx) = editing("editor-ctrl-click");
        // Two boxes side by side. The midpoint of each drag is inside the box
        // that drag created, whatever the panel layout worked out to be.
        drag(
            &mut app,
            &ctx,
            on_canvas(40.0, 60.0),
            on_canvas(160.0, 160.0),
        );
        drag(
            &mut app,
            &ctx,
            on_canvas(200.0, 60.0),
            on_canvas(320.0, 160.0),
        );
        let ids: Vec<u64> = app.editor.doc().boxes.iter().map(|b| b.id).collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(
            app.editor.selection(),
            [ids[1]],
            "drawing selects what was drawn"
        );

        // A plain click on the other box replaces the selection.
        click(&mut app, &ctx, on_canvas(100.0, 110.0), Modifiers::NONE);
        assert_eq!(app.editor.selection(), [ids[0]]);

        // Ctrl-click adds rather than replaces, so both can be restyled at once.
        click(&mut app, &ctx, on_canvas(260.0, 110.0), Modifiers::COMMAND);
        assert_eq!(app.editor.selection(), [ids[0], ids[1]]);

        // And Ctrl-click again takes it back out.
        click(&mut app, &ctx, on_canvas(260.0, 110.0), Modifiers::COMMAND);
        assert_eq!(app.editor.selection(), [ids[0]]);
    }

    #[test]
    fn a_group_move_carries_every_selected_box() {
        let (mut app, _dir, ctx) = editing("editor-group-move");
        drag(
            &mut app,
            &ctx,
            on_canvas(40.0, 60.0),
            on_canvas(160.0, 160.0),
        );
        drag(
            &mut app,
            &ctx,
            on_canvas(200.0, 60.0),
            on_canvas(320.0, 160.0),
        );
        click(&mut app, &ctx, on_canvas(100.0, 110.0), Modifiers::NONE);
        click(&mut app, &ctx, on_canvas(260.0, 110.0), Modifiers::COMMAND);
        assert_eq!(app.editor.selection().len(), 2);

        let before: Vec<BBox> = app.editor.doc().boxes.iter().map(|b| b.rect).collect();
        // Drag one of them: the other has to come along, or the selection would
        // look like it had been forgotten.
        drag(
            &mut app,
            &ctx,
            on_canvas(100.0, 110.0),
            on_canvas(120.0, 125.0),
        );
        let after: Vec<BBox> = app.editor.doc().boxes.iter().map(|b| b.rect).collect();
        assert_eq!(app.editor.doc().boxes.len(), 2, "the drag drew a third box");
        for (b, a) in before.iter().zip(&after) {
            assert_ne!(a, b, "a selected box stayed behind");
            assert_eq!(a.width(), b.width(), "a move must not resize");
        }
        // Moved by the same amount: the group keeps its shape.
        assert!((after[0].x1 - before[0].x1 - (after[1].x1 - before[1].x1)).abs() < 1e-3);

        // And the whole group move is one undo step.
        run(&mut app, &ctx, key(Key::Z, Modifiers::COMMAND));
        let undone: Vec<BBox> = app.editor.doc().boxes.iter().map(|b| b.rect).collect();
        assert_eq!(undone, before);
    }

    #[test]
    fn delete_removes_the_selected_box() {
        let (mut app, _dir, ctx) = editing("editor-delete");
        drag(
            &mut app,
            &ctx,
            on_canvas(40.0, 40.0),
            on_canvas(160.0, 140.0),
        );
        assert_eq!(app.editor.doc().boxes.len(), 1);
        run(&mut app, &ctx, key(Key::Delete, Modifiers::NONE));
        assert_eq!(app.editor.doc().boxes.len(), 0);
        assert!(app.editor.selection().is_empty());
    }

    #[test]
    fn a_whole_drag_is_one_undo_step() {
        let (mut app, _dir, ctx) = editing("editor-drag-step");
        // Nearly the whole image, so a press anywhere the image is drawn lands
        // on the box without the test having to predict the panel layout. It
        // still has room to move, so the clamp at the frame edge stays out of
        // the way.
        let id = app.editor.add_box(BBox::new(20.0, 15.0, 380.0, 285.0));
        let before = app.editor.doc().find(id).unwrap().rect;
        // Grab the box and shove it: the canvas fires a change per frame, and
        // all of them belong to one gesture.
        drag(
            &mut app,
            &ctx,
            on_canvas(200.0, 150.0),
            on_canvas(230.0, 170.0),
        );
        assert_eq!(
            app.editor.doc().boxes.len(),
            1,
            "the press missed the box and drew a new one"
        );
        let after = app.editor.doc().find(id).unwrap().rect;
        assert_ne!(after, before, "the box did not move");
        assert_eq!(after.width(), before.width(), "a move must not resize");

        run(&mut app, &ctx, key(Key::Z, Modifiers::COMMAND));
        assert_eq!(
            app.editor.doc().find(id).unwrap().rect,
            before,
            "one Ctrl-Z should undo the whole drag"
        );
    }

    #[test]
    fn a_dropped_image_lands_in_the_editor_rather_than_the_batch() {
        let (mut app, dir, ctx) = editing("editor-drop");
        let path = dir.join("dropped.png");
        let frame = test_frame(64, 48);
        ob_media::save_image(&frame, &path).unwrap();

        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, WIN)),
            dropped_files: vec![egui::DroppedFile {
                path: Some(path.clone()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run(input, |c| app.ui(c));

        assert_eq!(app.editor.path.as_deref(), Some(path.as_path()));
        assert_eq!(app.editor.image_size(), Some(Vec2::new(64.0, 48.0)));
        // And it did not also queue it up for a batch run.
        assert!(app.inputs.is_empty());
        assert_eq!(app.prefs.tab, Tab::Edit);
    }

    #[test]
    fn a_dropped_video_still_goes_to_the_batch() {
        let (mut app, dir, ctx) = editing("editor-drop-video");
        let clip = dir.join("clip.mp4");
        std::fs::write(&clip, b"not really a video").unwrap();
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, WIN)),
            dropped_files: vec![egui::DroppedFile {
                path: Some(clip.clone()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let _ = ctx.run(input, |c| app.ui(c));
        assert_eq!(app.inputs, vec![clip]);
        assert_eq!(app.prefs.tab, Tab::Queue);
    }

    #[test]
    fn the_suggested_name_never_matches_the_source() {
        let p = Path::new("/tmp/holiday/beach-01.jpg");
        assert_eq!(default_name(p).unwrap(), "beach-01-censored.png");
        // A PNG input must not suggest its own filename either.
        let p = Path::new("beach-01.png");
        assert_ne!(default_name(p).unwrap(), "beach-01.png");
    }

    #[test]
    fn only_images_are_taken_from_a_drop() {
        assert!(wants_drop(Path::new("a.png")));
        assert!(wants_drop(Path::new("a.JPEG")));
        assert!(!wants_drop(Path::new("clip.mp4")));
        assert!(!wants_drop(Path::new("notes.txt")));
    }
}
