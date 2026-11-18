//! The Review tab: walk what the detector found, fix it, then write.
//!
//! This is the second half of a two-phase batch. The detect pass has run and
//! written nothing; what is on screen is the geometry it produced, on the same
//! canvas the Edit page uses, and what leaves here is what gets burned into the
//! files.
//!
//! Every box drawn here goes through `CensorConfig::resolve`, the same call
//! the commit makes, so the canvas is not an impression of the output, it is
//! the output with the pixels still reversible.

use crate::app::{ObApp, ToastKind};
use crate::pages;
use crate::review::ReviewPhase;
use crate::theme;
use egui::RichText;
use ob_job::review::ReviewItem;
use ob_media::MediaKind;
use std::path::Path;

/// Width of the item list beside the canvas.
const LIST_WIDTH: f32 = 240.0;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    let Some(phase) = app.review.as_ref().map(|r| r.phase) else {
        empty_state(ui);
        return;
    };
    match phase {
        ReviewPhase::Detecting => {
            running_state(ui, "Looking at every file…", "Nothing is written yet.")
        }
        ReviewPhase::Committing => running_state(
            ui,
            "Writing the reviewed files…",
            "Using the boxes you approved. The detector is not running.",
        ),
        ReviewPhase::Done => done_state(app, ui),
        ReviewPhase::Ready => ready(app, ui),
    }
}

fn empty_state(ui: &mut egui::Ui) {
    let p = theme::palette();
    ui.vertical_centered(|ui| {
        ui.add_space(52.0);
        ui.label(
            RichText::new("Nothing to review yet")
                .size(14.0)
                .color(p.text_dim)
                .strong(),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Turn on \"Review before writing\" on the Queue tab and run the \
                 batch. The detector runs over everything first and stops here, \
                 so you can correct what it found before any file is written.",
            )
            .size(12.0)
            .color(p.text_faint),
        );
    });
}

fn running_state(ui: &mut egui::Ui, title: &str, detail: &str) {
    let p = theme::palette();
    ui.vertical_centered(|ui| {
        ui.add_space(52.0);
        ui.add(egui::Spinner::new().size(20.0));
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(14.0).color(p.text).strong());
        ui.add_space(4.0);
        ui.label(RichText::new(detail).size(12.0).color(p.text_faint));
    });
}

fn done_state(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let (items, edited) = app
        .review
        .as_ref()
        .map(|r| (r.review.items.len(), r.edited_count()))
        .unwrap_or((0, 0));
    ui.vertical_centered(|ui| {
        ui.add_space(52.0);
        ui.label(
            RichText::new(format!("{} {items} file(s) written", theme::glyph::OK))
                .size(15.0)
                .color(p.success)
                .strong(),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(if edited == 0 {
                "Committed exactly as detected. Nothing was changed in review.".to_string()
            } else {
                format!("{edited} of them with corrections you made here.")
            })
            .size(12.0)
            .color(p.text_dim),
        );
        ui.add_space(14.0);
        if ui.button("Start a new review").clicked() {
            app.review = None;
        }
    });
}

fn ready(app: &mut ObApp, ui: &mut egui::Ui) {
    summary_bar(app, ui);
    ui.add_space(8.0);

    // The list is a side panel rather than a column so the canvas gets every
    // pixel left over: judging a missed region is the job, and the job is done
    // at the size of the picture.
    egui::SidePanel::left("review-items")
        .resizable(true)
        .default_width(LIST_WIDTH)
        .min_width(170.0)
        .max_width((ui.available_width() - 360.0).max(200.0))
        .frame(theme::card_frame().inner_margin(egui::Margin::same(8.0)))
        .show_inside(ui, |ui| item_list(app, ui));

    egui::CentralPanel::default()
        .frame(egui::Frame::none().inner_margin(egui::Margin {
            left: 10.0,
            ..Default::default()
        }))
        .show_inside(ui, |ui| {
            toolbar(app, ui);
            ui.add_space(6.0);
            if let Some(err) = app.review.as_ref().and_then(|r| r.error.clone()) {
                theme::card_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(format!("{} {err}", theme::glyph::FAIL))
                            .size(12.5)
                            .color(theme::palette().danger),
                    );
                });
                return;
            }
            frame_strip(app, ui);
            if app.editor.source.is_some() {
                pages::editor::canvas(app, ui);
            }
        });
}

/// What the whole batch looks like, above the walk through it.
fn summary_bar(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    // Read the numbers out before drawing: the card's closure takes `app`
    // mutably to write the checkbox back, and a live `&app.review` across it
    // would not compile. The same borrow trap the Models page hit.
    let (total, regions, clean, failed, edited, mut show_clean) = {
        let Some(session) = app.review.as_ref() else {
            return;
        };
        (
            session.review.items.len(),
            session.review.region_count(),
            session.review.clean_count(),
            session.review.failed_count(),
            session.edited_count(),
            session.show_clean,
        )
    };

    let mut changed: Option<bool> = None;
    theme::card_frame()
        .inner_margin(egui::Margin::symmetric(12.0, 9.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!("{total} file(s)"))
                        .size(13.0)
                        .strong(),
                );
                theme::pill(ui, &format!("{regions} region(s)"), p.accent);
                // The count that matters most in a review: these are the files
                // where a miss would go out uncorrected.
                if clean > 0 {
                    theme::pill(ui, &format!("{clean} with nothing found"), p.warning);
                }
                if failed > 0 {
                    theme::pill(ui, &format!("{failed} unreadable"), p.danger);
                }
                if edited > 0 {
                    theme::pill(ui, &format!("{edited} corrected"), p.success);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .checkbox(&mut show_clean, "Show files with nothing found")
                        .on_hover_text(
                            "A file the detector found nothing in is where a miss \
                             goes unnoticed, so these are listed by default.",
                        )
                        .changed()
                    {
                        changed = Some(show_clean);
                    }
                });
            });
        });
    if let Some(value) = changed {
        if let Some(s) = app.review.as_mut() {
            s.show_clean = value;
        }
    }
}

fn item_list(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    theme::section(ui, "Files");
    let Some(session) = app.review.as_ref() else {
        return;
    };
    let visible = session.visible();
    let current = session.item;
    let rows: Vec<(usize, String, usize, bool, bool, MediaKind)> = visible
        .iter()
        .filter_map(|&i| {
            let item = session.review.items.get(i)?;
            Some((
                i,
                file_name(&item.path),
                item.region_count(),
                session.edited.contains(&i),
                item.error.is_some(),
                item.kind,
            ))
        })
        .collect();

    if rows.is_empty() {
        theme::hint(ui, "Nothing matches the filter.");
        return;
    }

    let mut jump: Option<usize> = None;
    egui::ScrollArea::vertical()
        .id_salt("review-items-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (index, name, regions, edited, failed, kind) in rows {
                let selected = index == current;
                let glyph = match (failed, kind) {
                    (true, _) => theme::glyph::FAIL,
                    (_, MediaKind::Video) => theme::glyph::VIDEO,
                    _ => theme::glyph::IMAGE,
                };
                let colour = if failed {
                    p.danger
                } else if regions == 0 {
                    p.warning
                } else {
                    p.text_dim
                };
                let resp = ui.add(
                    egui::Button::new(
                        RichText::new(format!("{glyph}  {name}"))
                            .size(12.5)
                            .color(if selected { p.on_accent } else { p.text }),
                    )
                    .fill(if selected {
                        p.accent
                    } else {
                        egui::Color32::TRANSPARENT
                    })
                    .stroke(egui::Stroke::NONE)
                    .rounding(egui::Rounding::same(theme::RADIUS))
                    .min_size(egui::vec2(ui.available_width(), 24.0)),
                );
                if resp.clicked() {
                    jump = Some(index);
                }
                ui.horizontal(|ui| {
                    ui.add_space(22.0);
                    ui.label(
                        RichText::new(if failed {
                            "could not be read".to_string()
                        } else if regions == 0 {
                            "nothing found".to_string()
                        } else {
                            format!("{regions} region(s)")
                        })
                        .size(10.5)
                        .color(if selected {
                            p.on_accent
                        } else {
                            colour
                        }),
                    );
                    if edited {
                        ui.label(RichText::new("· corrected").size(10.5).color(if selected {
                            p.on_accent
                        } else {
                            p.success
                        }));
                    }
                });
                ui.add_space(2.0);
            }
        });

    if let Some(index) = jump {
        let editor = std::mem::take(&mut app.editor);
        if let Some(s) = app.review.as_mut() {
            s.go_to(&editor, index);
        }
        app.editor = editor;
    }
}

/// Item navigation and the canvas's own actions.
fn toolbar(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let (label, name, kind, regions) = {
        let Some(session) = app.review.as_ref() else {
            return;
        };
        (
            session.position_label(),
            session
                .current()
                .map(|i| file_name(&i.path))
                .unwrap_or_default(),
            session.current().map(|i| i.kind),
            session.current().map(|i| i.region_count()).unwrap_or(0),
        )
    };

    ui.horizontal(|ui| {
        let mut step: Option<bool> = None;
        if ui.button("‹ Previous").clicked() {
            step = Some(false);
        }
        if ui.button("Next ›").clicked() {
            step = Some(true);
        }
        if let Some(forward) = step {
            let editor = std::mem::take(&mut app.editor);
            if let Some(s) = app.review.as_mut() {
                s.step_item(&editor, forward);
            }
            app.editor = editor;
        }
        ui.label(RichText::new(label).size(11.5).color(p.text_faint));
        ui.label(RichText::new(name).size(12.5).strong());
        if kind == Some(MediaKind::Video) {
            theme::pill(ui, "video · experimental", p.warning);
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(app.editor.history.can_redo(), egui::Button::new("Redo"))
                .clicked()
            {
                app.editor.history.redo();
            }
            if ui
                .add_enabled(app.editor.history.can_undo(), egui::Button::new("Undo"))
                .clicked()
            {
                app.editor.history.undo();
            }
            let has_selection = !app.editor.selection().is_empty();
            if ui
                .add_enabled(has_selection, egui::Button::new("Delete box"))
                .on_hover_text("Remove the selected box so this region is not censored.")
                .clicked()
            {
                for id in app.editor.selection().to_vec() {
                    app.editor.delete(id);
                }
            }
            // What the model called the selected region, and how sure it was.
            // Without this a review is a walk through anonymous rectangles, and
            // "is that a false positive" is unanswerable.
            match app.editor.selected_box() {
                Some(b) => {
                    ui.label(
                        RichText::new(format!("{}  ·  {:.0}%", b.review_label(), b.score * 100.0))
                            .size(11.5)
                            .color(p.text_dim),
                    );
                }
                None => {
                    ui.label(
                        RichText::new(format!("{regions} here"))
                            .size(11.5)
                            .color(p.text_faint),
                    );
                }
            }
        });
    });
}

/// Video only: which analysed frame is on screen, and how to move between them.
fn frame_strip(app: &mut ObApp, ui: &mut egui::Ui) {
    let Some((slots, slot, every, index, populated)) = app.review.as_ref().and_then(|s| {
        let item = s.current()?;
        (item.kind == MediaKind::Video).then(|| {
            (
                item.slot_count(),
                s.slot,
                item.detect_every,
                item.slot_index(s.slot),
                item.frames_with_regions().count(),
            )
        })
    }) else {
        return;
    };
    let p = theme::palette();

    theme::card_frame()
        .inner_margin(egui::Margin::symmetric(12.0, 8.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let mut target: Option<usize> = None;
                if ui
                    .add_enabled(slot > 0, egui::Button::new("‹ Region"))
                    .on_hover_text("Previous analysed frame that has something on it.")
                    .clicked()
                {
                    target = app
                        .review
                        .as_ref()
                        .and_then(|s| s.prev_populated_slot(slot));
                }
                if ui
                    .add_enabled(slot + 1 < slots, egui::Button::new("Region ›"))
                    .on_hover_text("Next analysed frame that has something on it.")
                    .clicked()
                {
                    target = app
                        .review
                        .as_ref()
                        .and_then(|s| s.next_populated_slot(slot));
                }

                let mut pos = slot;
                let slider = egui::Slider::new(&mut pos, 0..=slots.saturating_sub(1))
                    .show_value(false)
                    .text(format!("frame {index}"));
                if ui
                    .add(slider)
                    .on_hover_text(format!(
                        "The detector ran on one frame in {every}; those are the \
                         frames you can step between. Between them the tracker \
                         carries the boxes forward, so a correction here follows \
                         through to the frames either side of it."
                    ))
                    .changed()
                {
                    target = Some(pos);
                }

                if let Some(to) = target {
                    let editor = std::mem::take(&mut app.editor);
                    if let Some(s) = app.review.as_mut() {
                        s.go_to_slot(&editor, to);
                    }
                    app.editor = editor;
                }

                ui.label(
                    RichText::new(format!("{populated} frame(s) with regions"))
                        .size(11.0)
                        .color(p.text_faint),
                );
            });
        });
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Bring the canvas into step with the cursor, decoding the frame if it moved.
///
/// Called from the frame loop rather than from the page body so it also runs
/// while the user is on another tab: a review left on the Queue tab and come
/// back to should not need a click to redraw itself.
pub fn sync_canvas(app: &mut ObApp) {
    let Some(session) = app.review.as_ref() else {
        return;
    };
    if session.phase != ReviewPhase::Ready || session.canvas_is_current() {
        return;
    }
    let Some(item) = session.current().cloned() else {
        return;
    };
    let (slot, index) = (session.slot, item.slot_index(session.slot));

    if let Some(err) = &item.error {
        let message = err.clone();
        if let Some(s) = app.review.as_mut() {
            s.error = Some(message);
            s.loaded = Some((s.item, slot));
        }
        app.editor.source = None;
        return;
    }

    // Decoded on the UI thread. An image is a decode; a video frame is an
    // ffmpeg seek, which is the slower of the two and the reason stepping
    // through a clip pauses briefly. It only happens when the cursor actually
    // moves, so it costs a deliberate click rather than a frame.
    match ob_job::review::load_review_frame(&item, index) {
        Ok(frame) => {
            let dets = item.detections_at(slot).to_vec();
            app.editor.load_review(item.path.clone(), frame, &dets);
            if let Some(s) = app.review.as_mut() {
                s.error = None;
                s.loaded = Some((s.item, slot));
            }
        }
        Err(e) => {
            app.editor.source = None;
            if let Some(s) = app.review.as_mut() {
                s.error = Some(e.to_string());
                s.loaded = Some((s.item, slot));
            }
        }
    }
}

/// Flush the canvas and start writing the batch.
pub fn commit(app: &mut ObApp) {
    let editor = std::mem::take(&mut app.editor);
    if let Some(s) = app.review.as_mut() {
        s.flush(&editor);
    }
    app.editor = editor;

    let Some(session) = app.review.as_ref() else {
        return;
    };
    let Some(output_dir) = app.prefs.output_dir.clone() else {
        app.toast("Choose an output folder first.", ToastKind::Error);
        return;
    };
    let review = session.review.clone();
    let detect_every = app.prefs.detect_every;

    app.sync_profile();
    let _ = app.prefs.save();
    app.run_state = crate::run::RunState::starting(None, app.calibration());
    app.run = Some(crate::run::RunHandle::spawn_commit(
        &app.prefs.profile,
        review,
        output_dir,
        detect_every,
    ));
    if let Some(s) = app.review.as_mut() {
        s.phase = ReviewPhase::Committing;
    }
}

/// How many files a commit would write, for the button that starts it.
pub fn committable(item: &ReviewItem) -> bool {
    item.error.is_none() && item.kind != MediaKind::Unknown
}
