//! The preview panel, shared by the Batch and Tuning pages.
//!
//! One widget in two places rather than two that drift: what a preview shows,
//! and what it says when it has nothing to show, should not depend on which
//! page you happen to be reading it from.
//!
//! On the Tuning page this sits beside the controls and re-renders itself as
//! they change. That adjacency is the point — judging a confidence threshold or
//! a padding fraction means seeing the result, and a preview that costs a page
//! switch and a button press is a preview that stops being consulted.

use crate::app::ObApp;
use crate::theme;
use egui::RichText;
use std::path::Path;

/// Below this, an A/B comparison stacks instead of sitting side by side.
const SIDE_BY_SIDE_MIN: f32 = 460.0;

/// A preview is for judging what was covered, not for pixel-peeping; past this
/// it just pushes the controls under it off screen.
const MAX_IMAGE_HEIGHT: f32 = 420.0;

/// Draw the whole panel: controls, status line, and the image(s).
pub fn panel(app: &mut ObApp, ui: &mut egui::Ui) {
    header(app, ui);
    ui.add_space(6.0);
    source_row(app, ui);
    ui.add_space(6.0);
    body(app, ui);
}

fn header(app: &mut ObApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Preview").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if app.preview.is_some() {
                // Judging padding or a missed region is guesswork without the
                // untouched frame to compare against.
                ui.checkbox(&mut app.prefs.preview_compare, "Original")
                    .on_hover_text("Show the uncensored source next to the result.");
            }
            ui.checkbox(&mut app.prefs.preview_auto, "Live")
                .on_hover_text(
                    "Re-render as you change the settings, shortly after you stop \
                     adjusting. Changing a censor style or the category filter \
                     re-paints the frame that was already analysed; changing the \
                     model or a detection setting runs the detector again.",
                );
        });
    });
}

/// Which file is being previewed, and how to change it.
fn source_row(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let path = app.preview_path();
    ui.horizontal(|ui| {
        if ui.button("Sample…").clicked() {
            if let Some(picked) = rfd::FileDialog::new().pick_file() {
                app.preview_source_path = Some(picked);
                // A different file needs a different detection pass; letting the
                // stale one stand would be showing the wrong picture.
                app.preview_cache = None;
            }
        }
        if app.preview_source_path.is_some() && ui.button("Use first input").clicked() {
            app.preview_source_path = None;
            app.preview_cache = None;
        }
        match &path {
            Some(path) => {
                ui.label(
                    RichText::new(display_name(path))
                        .size(11.5)
                        .color(p.text_dim),
                )
                .on_hover_text(path.display().to_string());
            }
            None => {
                ui.label(
                    RichText::new("no file chosen")
                        .size(11.5)
                        .color(p.text_faint),
                );
            }
        }
    })
    .response
    .on_hover_text(
        "The frame to tune against. Defaults to the first file in the batch; \
         pick a sample to tune against something representative without adding \
         it to the queue. Videos preview their first frame.",
    );
}

fn body(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();

    if let Some(err) = &app.preview_error {
        ui.label(
            RichText::new(format!("{} {err}", theme::glyph::FAIL))
                .size(12.5)
                .color(p.danger),
        );
        return;
    }

    let Some(preview) = &app.preview else {
        empty_state(app, ui);
        return;
    };

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{} region(s) censored", preview.regions))
                .size(12.0)
                .color(if preview.regions == 0 {
                    p.warning
                } else {
                    p.text_dim
                }),
        );
        // The image below is the *previous* render until this lands. Saying so
        // is what makes leaving it on screen honest rather than misleading.
        if let Some(job) = &app.preview_job {
            ui.add(egui::Spinner::new().size(12.0));
            ui.label(
                RichText::new(if job.is_detect {
                    "detecting…"
                } else {
                    "rendering…"
                })
                .size(11.5)
                .color(p.text_faint),
            );
        }
    });
    if preview.regions == 0 && app.preview_job.is_none() {
        ui.label(
            RichText::new(
                "Nothing was censored. Lower the confidence threshold, check the \
                 category filter, or try a model trained on this kind of content.",
            )
            .size(11.5)
            .color(p.text_faint),
        );
    }
    ui.add_space(6.0);

    let compare = app.prefs.preview_compare;
    let avail = ui.available_width();
    // Side by side halves the width each frame gets, which in a narrow side
    // panel leaves two images too small to judge anything by. Stack them there
    // instead — the comparison is just as direct and each frame is twice the
    // size.
    let stacked = compare && avail < SIDE_BY_SIDE_MIN;
    let each = if compare && !stacked {
        (avail - 8.0) / 2.0
    } else {
        avail
    };
    // Stacked, the two frames share the panel's height, so cap each at half of
    // what is left: an A/B comparison you have to scroll between is not one.
    let tall = if stacked {
        ((ui.available_height() - 44.0) / 2.0).clamp(120.0, MAX_IMAGE_HEIGHT)
    } else {
        MAX_IMAGE_HEIGHT
    };
    let original = |ui: &mut egui::Ui| {
        ui.label(RichText::new("original").size(11.0).color(p.text_faint));
        image(ui, &preview.original, each, tall);
    };
    let censored = |ui: &mut egui::Ui| {
        if compare {
            ui.label(RichText::new("censored").size(11.0).color(p.text_faint));
        }
        image(ui, &preview.censored, each, tall);
    };

    if stacked {
        original(ui);
        ui.add_space(8.0);
        censored(ui);
    } else {
        ui.horizontal_top(|ui| {
            if compare {
                ui.vertical(original);
            }
            ui.vertical(censored);
        });
    }
}

/// What the panel says before it has anything to show. Each case names the one
/// thing that is missing, rather than a single generic prompt.
fn empty_state(app: &ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let message = if app.preview_job.is_some() {
        "Analysing the first frame…"
    } else if app.preview_path().is_none() {
        "Add a file to the batch, or pick a sample above, to see what will be censored."
    } else if !app.model_ready() {
        "The selected model is not installed — get it on the Models page."
    } else if app.prefs.preview_auto {
        "Rendering…"
    } else {
        "Live preview is off. Press \"Refresh preview\" below."
    };
    ui.vertical_centered(|ui| {
        ui.add_space(18.0);
        if app.preview_job.is_some() {
            ui.add(egui::Spinner::new().size(18.0));
            ui.add_space(6.0);
        }
        ui.label(RichText::new(message).size(12.5).color(p.text_faint));
        ui.add_space(18.0);
    });
}

fn image(ui: &mut egui::Ui, tex: &egui::TextureHandle, max_width: f32, max_height: f32) {
    let sized = egui::load::SizedTexture::from_handle(tex);
    ui.add(
        egui::Image::from_texture(sized)
            .max_width(max_width)
            .max_height(max_height)
            .rounding(egui::Rounding::same(4.0)),
    );
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
