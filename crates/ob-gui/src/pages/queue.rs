//! The Batch page: what to process, where it goes, the preview, and the log.

use crate::app::{open_in_file_manager, ObApp, ToastKind};
use crate::pages;
use crate::theme;
use egui::RichText;
use ob_core::registry::human_bytes;
use ob_media::{classify_resolved, MediaKind};
use std::path::{Path, PathBuf};

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    pages::header(
        ui,
        "Batch",
        "Drop files or folders here, choose where the results go, then run.",
    );

    inputs_section(app, ui);
    ui.add_space(8.0);
    estimate_line(app, ui);
    ui.add_space(12.0);
    output_section(app, ui);
    ui.add_space(12.0);

    // The lower half is split: preview on the left, activity on the right.
    // Both are secondary to the queue above, and both benefit from height.
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            preview_section(app, ui);
            ui.add_space(12.0);
            log_section(app, ui);
        });
}

/// What the batch as queued is going to cost.
///
/// Shown before the run rather than after the first file, because this is the
/// number someone uses to decide whether to start it now or leave it overnight.
/// Until a run has been measured on this machine the figure rests on a built-in
/// guess at inference speed, which varies by two orders of magnitude across
/// hardware — so say so rather than presenting a guess as a measurement.
fn estimate_line(app: &ObApp, ui: &mut egui::Ui) {
    if app.inputs.is_empty() || app.is_running() {
        return;
    }
    let p = theme::palette();

    if app.probed.is_none() {
        theme::hint(ui, "Measuring the batch…");
        return;
    }
    let Some(w) = app.workload() else {
        return;
    };

    let mut parts: Vec<String> = Vec::new();
    if w.images > 0 {
        parts.push(format!("{} image(s)", w.images));
    }
    if w.videos > 0 {
        parts.push(format!("{} video(s)", w.videos));
    }
    let what = parts.join(", ");

    let cal = app.calibration();
    let eta = crate::downloads::human_eta(cal.secs_for(w.total_work));
    let text = if cal.is_measured() {
        format!("{what} — about {eta} at this machine's last measured speed")
    } else {
        format!("{what} — roughly {eta}, until a run has been timed here")
    };
    ui.label(RichText::new(text).size(12.0).color(p.text_dim));

    if w.unknown > 0 {
        // Excluded from the total rather than guessed at, so say it is missing
        // instead of letting the estimate quietly understate the batch.
        theme::hint(
            ui,
            &format!(
                "Plus {} video(s) whose length could not be read — not included above.",
                w.unknown
            ),
        );
    }
}

fn inputs_section(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    ui.horizontal(|ui| {
        if ui.button("Add files…").clicked() {
            if let Some(files) = rfd::FileDialog::new().pick_files() {
                let n = files.len();
                app.add_inputs(files);
                app.toast(format!("Added {n} file(s)."), ToastKind::Success);
            }
        }
        if ui.button("Add folder…").clicked() {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                app.add_inputs([dir]);
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let can_clear = !app.inputs.is_empty() && !app.is_running();
            if ui
                .add_enabled(can_clear, egui::Button::new("Clear all"))
                .clicked()
            {
                app.inputs.clear();
            }
            let (files, folders) = counts(&app.inputs);
            ui.label(
                RichText::new(format!("{files} file(s), {folders} folder(s)"))
                    .size(12.0)
                    .color(p.text_dim),
            );
        });
    });
    ui.add_space(8.0);

    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.set_min_height(150.0);

        if app.inputs.is_empty() {
            // A real empty state, not a blank box: it says what to do and what
            // is accepted, which is the whole job of this screen for a new user.
            ui.vertical_centered(|ui| {
                ui.add_space(34.0);
                ui.label(
                    RichText::new("Drop images, videos or folders here")
                        .size(15.0)
                        .color(p.text_dim),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!(
                        "Images: {} · Video: {}",
                        ob_media::IMAGE_EXTS.join(", "),
                        ob_media::VIDEO_EXTS.join(", ")
                    ))
                    .size(11.5)
                    .color(p.text_faint),
                );
                ui.add_space(30.0);
            });
            return;
        }

        let mut remove: Option<usize> = None;
        egui::ScrollArea::vertical()
            .max_height(190.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (i, path) in app.inputs.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let (glyph, color) = kind_glyph(path);
                        ui.label(RichText::new(glyph).color(color));
                        ui.label(RichText::new(display_name(path)).size(13.0).color(p.text))
                            .on_hover_text(path.display().to_string());

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add_enabled(
                                    !app.is_running(),
                                    egui::Button::new(
                                        RichText::new(theme::glyph::REMOVE).size(12.0),
                                    )
                                    .frame(false),
                                )
                                .on_hover_text("Remove from the batch")
                                .clicked()
                            {
                                remove = Some(i);
                            }
                            if let Some(size) = file_size(path) {
                                ui.label(
                                    RichText::new(human_bytes(size))
                                        .size(11.5)
                                        .color(p.text_faint),
                                );
                            }
                        });
                    });
                }
            });
        if let Some(i) = remove {
            app.inputs.remove(i);
        }
    });
}

fn output_section(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    ui.horizontal(|ui| {
        if ui.button("Output folder…").clicked() {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                app.prefs.output_dir = Some(dir);
                let _ = app.prefs.save();
            }
        }
        match &app.prefs.output_dir {
            Some(dir) => {
                let dir = dir.clone();
                ui.label(
                    RichText::new(dir.display().to_string())
                        .size(12.5)
                        .color(p.text_dim),
                );
                if ui.button("Open").clicked() {
                    let _ = std::fs::create_dir_all(&dir);
                    open_in_file_manager(&dir);
                }
            }
            None => {
                ui.label(
                    RichText::new("no output folder chosen")
                        .size(12.5)
                        .color(p.warning),
                );
            }
        }
    });
}

fn preview_section(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        pages::preview::panel(app, ui);
    });
}

fn log_section(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    if app.run_state.log.is_empty() && !app.is_running() {
        return;
    }
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Activity").strong());
            let errors = app.run_state.error_count();
            if errors > 0 {
                theme::pill(ui, &format!("{errors} error(s)"), p.danger);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // On a big batch the failures are the only lines anyone needs,
                // and they are otherwise buried among thousands of successes.
                ui.checkbox(&mut app.errors_only, "Errors only");
            });
        });

        if let Some(current) = &app.run_state.current {
            ui.label(
                RichText::new(format!(
                    "{} {}",
                    theme::glyph::RUNNING,
                    display_name(current)
                ))
                .size(12.0)
                .color(p.accent_hover),
            );
        }
        ui.add_space(4.0);

        egui::ScrollArea::vertical()
            .max_height(200.0)
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut shown = 0;
                for entry in app.run_state.log.iter().rev() {
                    if app.errors_only && !entry.is_error() {
                        continue;
                    }
                    shown += 1;
                    ui.horizontal_wrapped(|ui| match &entry.error {
                        Some(err) => {
                            ui.label(RichText::new(theme::glyph::FAIL).color(p.danger));
                            ui.label(
                                RichText::new(display_name(&entry.path))
                                    .size(12.0)
                                    .color(p.text),
                            );
                            ui.label(RichText::new(err).size(11.5).color(p.danger));
                        }
                        None => {
                            ui.label(RichText::new(theme::glyph::OK).color(p.success));
                            ui.label(
                                RichText::new(display_name(&entry.path))
                                    .size(12.0)
                                    .color(p.text_dim),
                            );
                            ui.label(
                                RichText::new(format!("{} region(s)", entry.regions))
                                    .size(11.5)
                                    .color(p.text_faint),
                            );
                        }
                    });
                }
                if shown == 0 {
                    ui.label(RichText::new("No errors.").size(12.0).color(p.text_faint));
                }
            });

        if let Some(f) = &app.run_state.finished {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if let Some(dir) = app.prefs.output_dir.clone() {
                    if theme::primary_button(ui, "Open output folder", true).clicked() {
                        open_in_file_manager(&dir);
                    }
                }
                ui.label(
                    RichText::new(if f.cancelled {
                        format!("stopped — {} written", f.ok)
                    } else {
                        format!("{} written, {} failed", f.ok, f.failed)
                    })
                    .size(12.0)
                    .color(p.text_dim),
                );
            });
        }
    });
}

// --- small helpers ---------------------------------------------------------

/// Trailing path component, which is what identifies a file to a human. The
/// full path is available on hover.
fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn file_size(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    meta.is_file().then_some(meta.len())
}

/// A glyph distinguishing folders, images, video and unsupported files.
fn kind_glyph(path: &Path) -> (&'static str, egui::Color32) {
    use theme::glyph;
    let p = theme::palette();
    if path.is_dir() {
        return (glyph::FOLDER, p.accent_hover);
    }
    match classify_resolved(path) {
        MediaKind::Image => (glyph::IMAGE, p.text_dim),
        MediaKind::Video => (glyph::VIDEO, p.text_dim),
        // Flagged rather than hidden: a file the batch will skip should be
        // visible in the list, not silently dropped at run time.
        MediaKind::Unknown => (glyph::UNSUPPORTED, p.warning),
    }
}

fn counts(inputs: &[PathBuf]) -> (usize, usize) {
    let folders = inputs.iter().filter(|p| p.is_dir()).count();
    (inputs.len() - folders, folders)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_name_is_the_file_name() {
        assert_eq!(display_name(Path::new("/a/b/c.png")), "c.png");
        // A path with no final component still renders as something.
        assert!(!display_name(Path::new("/")).is_empty());
    }

    #[test]
    fn counts_split_files_from_folders() {
        let dir = std::env::temp_dir();
        let file = dir.join("ob-queue-test.png");
        std::fs::write(&file, b"x").unwrap();
        let (files, folders) = counts(&[dir.clone(), file.clone()]);
        assert_eq!((files, folders), (1, 1));
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn unsupported_files_are_flagged_not_hidden() {
        // The batch skips these; the queue must still show them so the user
        // is not left wondering why a file "did nothing". Compared against the
        // glyph table rather than literals, so changing a symbol in one place
        // does not mean editing it in two.
        use theme::glyph;
        assert_eq!(kind_glyph(Path::new("notes.txt")).0, glyph::UNSUPPORTED);
        assert_eq!(kind_glyph(Path::new("a.png")).0, glyph::IMAGE);
        assert_eq!(kind_glyph(Path::new("a.mp4")).0, glyph::VIDEO);
        // The four kinds must stay visually distinguishable.
        assert_ne!(glyph::IMAGE, glyph::VIDEO);
        assert_ne!(glyph::IMAGE, glyph::FOLDER);
        assert_ne!(glyph::UNSUPPORTED, glyph::IMAGE);
    }
}
