//! First-run setup.
//!
//! Figura Obscura cannot do anything without detector weights, and the weights are
//! too large to ship inside the installer for every model. So the first launch
//! (and the installer's optional post-install step) offers exactly one action:
//! download the recommended models.
//!
//! This screen replaces the whole window rather than appearing as a dialog. A
//! new user with no model has nothing useful to do behind it, and a modal over
//! a fully-populated UI would suggest otherwise.

use crate::app::{ObApp, ToastKind};
use crate::theme;
use egui::RichText;
use ob_core::registry::human_bytes;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();

    // Centre the card rather than stretching it: this is a one-decision screen.
    ui.vertical_centered(|ui| {
        ui.add_space(48.0);
        ui.set_max_width(560.0);

        ui.label(RichText::new("Welcome to Figura Obscura").size(24.0).strong());
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "One more step: Figura Obscura needs a detection model before it can \
                 censor anything. Models are downloaded once and then used entirely \
                 offline — nothing you process ever leaves your machine.",
            )
            .color(p.text_dim),
        );
        ui.add_space(20.0);

        let defaults: Vec<_> = app
            .registry
            .iter()
            .filter(|m| m.default_download)
            .cloned()
            .collect();

        let pending: Vec<_> = defaults
            .iter()
            .filter(|m| !app.downloads.is_installed(m.id))
            .collect();
        let total: u64 = pending.iter().map(|m| m.approx_bytes).sum();
        let busy = app.downloads.any_active();

        theme::card_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            for m in &defaults {
                let installed = app.downloads.is_installed(m.id);
                // Name and size share one line; the wrapping description gets a
                // line of its own. Nesting a wrapping label and a right-aligned
                // one in the same row makes the wrapping side claim the full
                // width and the two draw on top of each other.
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(if installed {
                            theme::glyph::OK
                        } else {
                            theme::glyph::PENDING
                        })
                        .color(if installed {
                            p.success
                        } else {
                            p.text_faint
                        }),
                    );
                    ui.label(RichText::new(m.display_name).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if installed {
                                "installed".to_string()
                            } else {
                                format!("~{}", human_bytes(m.approx_bytes))
                            })
                            .size(12.0)
                            .color(p.text_faint),
                        );
                    });
                });
                ui.indent(m.id, |ui| {
                    // The screen is centred, but a wrapping paragraph must not
                    // be: centred body text is markedly harder to read.
                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                        ui.label(RichText::new(m.summary).size(12.0).color(p.text_dim));
                    });
                });

                // Per-model progress while the batch of downloads runs.
                if let Some(active) = app.downloads.active(m.id) {
                    let bar = match active.fraction() {
                        Some(f) => egui::ProgressBar::new(f).text(format!(
                            "{} / {}",
                            human_bytes(active.downloaded),
                            active.total.map(human_bytes).unwrap_or_else(|| "?".into())
                        )),
                        None => egui::ProgressBar::new(0.0)
                            .animate(true)
                            .text(human_bytes(active.downloaded)),
                    };
                    ui.add(bar.desired_height(12.0).fill(p.accent));
                }
                if let Some(err) = app.downloads.error(m.id) {
                    ui.label(
                        RichText::new(format!("{} {err}", theme::glyph::FAIL))
                            .size(12.0)
                            .color(p.danger),
                    );
                }
                ui.add_space(6.0);
            }
        });

        ui.add_space(18.0);

        if pending.is_empty() && !busy {
            ui.label(
                RichText::new("All set — the recommended models are installed.").color(p.success),
            );
            ui.add_space(10.0);
            if theme::primary_button(ui, "Start using Figura Obscura", true).clicked() {
                finish(app);
            }
        } else {
            let label = if busy {
                "Downloading…".to_string()
            } else {
                format!(
                    "Download {} models  ({})",
                    pending.len(),
                    human_bytes(total)
                )
            };
            if theme::primary_button(ui, &label, !busy).clicked() {
                for m in &defaults {
                    if !app.downloads.is_installed(m.id) {
                        let m = (*m).clone();
                        app.downloads.start(&m, false);
                    }
                }
            }
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                if busy {
                    if ui.button("Cancel downloads").clicked() {
                        app.downloads.cancel_all();
                    }
                } else if ui.button("Skip — I'll do this later").clicked() {
                    // An offline install is a legitimate state, and the Models
                    // page can do this any time. Don't trap anyone here.
                    finish(app);
                    app.toast(
                        "You can download models any time from the Models page.",
                        ToastKind::Info,
                    );
                }
            });
        }

        ui.add_space(24.0);
        ui.label(
            RichText::new(format!(
                "Models are stored in {}",
                ob_models::cache_dir()
                    .map(|d| d.display().to_string())
                    .unwrap_or_else(|_| "your cache directory".into())
            ))
            .size(11.5)
            .color(p.text_faint),
        );
    });
}

/// Leave the setup screen and remember that we did.
fn finish(app: &mut ObApp) {
    app.show_setup = false;
    app.prefs.setup_done = true;
    let _ = app.prefs.save();
    // Anything downloaded during setup should now be selectable.
    let registry = std::mem::take(&mut app.registry);
    app.downloads.refresh_all(&registry);
    app.registry = registry;
}
