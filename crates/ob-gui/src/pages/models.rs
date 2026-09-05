//! The Models page: install, update, remove and choose detector weights.

use crate::app::{ObApp, ToastKind};
use crate::pages;
use crate::theme;
use egui::RichText;
use ob_core::registry::{human_bytes, ModelEntry};
use ob_models::ModelStatus;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    pages::header(
        ui,
        "Models",
        "Detector weights. Downloaded once, then used entirely offline.",
    );

    let p = theme::palette();
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!(
                "Cache: {}",
                ob_models::cache_dir()
                    .map(|d| d.display().to_string())
                    .unwrap_or_else(|_| "unavailable".into())
            ))
            .size(12.0)
            .color(p.text_faint),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Open folder").clicked() {
                if let Ok(dir) = ob_models::cache_dir() {
                    let _ = std::fs::create_dir_all(&dir);
                    crate::app::open_in_file_manager(&dir);
                }
            }
            if ui.button("Re-scan").clicked() {
                let registry = std::mem::take(&mut app.registry);
                app.downloads.refresh_all(&registry);
                app.registry = registry;
                app.toast("Model cache re-scanned.", ToastKind::Info);
            }
        });
    });
    ui.add_space(10.0);

    // The registry is moved out for the loop so each card can take `&mut app`
    // without borrowing `app.registry` at the same time.
    let registry = std::mem::take(&mut app.registry);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for entry in &registry {
                card(app, ui, entry);
                ui.add_space(10.0);
            }
        });
    app.registry = registry;
}

fn card(app: &mut ObApp, ui: &mut egui::Ui, entry: &ModelEntry) {
    let p = theme::palette();
    let selected = app.prefs.profile.model_id == entry.id;
    let status = app.downloads.status(entry.id);
    let downloading = app.downloads.is_downloading(entry.id);

    // The selected model gets an accent border: on a page of five near-identical
    // cards, "which one am I actually using" must be answerable at a glance.
    let frame = theme::card_frame().stroke(egui::Stroke::new(
        if selected { 1.5_f32 } else { 1.0_f32 },
        if selected { p.accent } else { p.stroke },
    ));

    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(entry.display_name).strong());
                    if selected {
                        theme::pill(ui, "in use", p.accent);
                    }
                });
                ui.label(RichText::new(entry.id).size(11.5).color(p.text_faint));
            });

            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| match &status {
                    ModelStatus::Installed { bytes, verified } => {
                        ui.label(
                            RichText::new(if *verified {
                                format!("{} {} · verified", theme::glyph::OK, human_bytes(*bytes))
                            } else {
                                format!("{} {}", theme::glyph::OK, human_bytes(*bytes))
                            })
                            .size(12.0)
                            .color(p.success),
                        );
                    }
                    ModelStatus::Missing if downloading => {}
                    ModelStatus::Missing => {
                        ui.label(
                            RichText::new(format!("~{}", human_bytes(entry.approx_bytes)))
                                .size(12.0)
                                .color(p.text_faint),
                        );
                    }
                },
            );
        });

        ui.add_space(6.0);
        pages::model_badges(ui, entry);
        ui.add_space(6.0);
        ui.label(RichText::new(entry.summary).size(12.5).color(p.text_dim));
        ui.add_space(10.0);

        // --- in-flight download ------------------------------------------
        if let Some(active) = app.downloads.active(entry.id) {
            let rate = active
                .bytes_per_sec()
                .map(|r| format!("{}/s", human_bytes(r as u64)))
                .unwrap_or_default();
            let eta = active
                .eta_secs()
                .map(|s| format!(" · {} left", crate::downloads::human_eta(s)))
                .unwrap_or_default();
            let text = format!(
                "{} / {} {rate}{eta}",
                human_bytes(active.downloaded),
                active
                    .total
                    .map(human_bytes)
                    .unwrap_or_else(|| "?".to_string()),
            );
            let bar = match active.fraction() {
                Some(f) => egui::ProgressBar::new(f).text(text),
                None => egui::ProgressBar::new(0.0).animate(true).text(text),
            };
            ui.add(bar.desired_height(15.0).fill(p.accent));
            ui.add_space(6.0);
        }

        if let Some(err) = app.downloads.error(entry.id) {
            // Model URLs point at third-party hosts that occasionally serve a
            // sign-in or rate-limit page instead of the file. The error text
            // from ob-models explains that; show it in full rather than a
            // generic "download failed".
            ui.label(
                RichText::new(format!("{} {err}", theme::glyph::FAIL))
                    .size(12.0)
                    .color(p.danger),
            );
            ui.add_space(6.0);
        }

        // --- actions -------------------------------------------------------
        ui.horizontal(|ui| {
            if downloading {
                if theme::danger_button(ui, "Cancel", true).clicked() {
                    app.downloads.cancel(entry.id);
                }
            } else {
                match &status {
                    ModelStatus::Missing => {
                        if theme::primary_button(
                            ui,
                            &format!("Download ({})", human_bytes(entry.approx_bytes)),
                            true,
                        )
                        .clicked()
                        {
                            app.downloads.start(entry, false);
                        }
                    }
                    ModelStatus::Installed { .. } => {
                        if theme::primary_button(ui, "Use this model", !selected).clicked() {
                            app.select_model(entry.id);
                            app.toast(
                                format!("Now using {}.", entry.display_name),
                                ToastKind::Success,
                            );
                        }
                        if ui
                            .button("Re-download")
                            .on_hover_text(
                                "Fetch the file again, replacing the cached copy. Use this \
                                 if detection results look wrong or the file may be damaged.",
                            )
                            .clicked()
                        {
                            app.downloads.start(entry, true);
                        }
                        // Deleting the model you are using would leave the app
                        // in a state where Run silently refuses.
                        let can_remove = !selected;
                        let remove = ui.add_enabled(can_remove, egui::Button::new("Remove"));
                        if remove.clicked() {
                            match app.downloads.remove(entry) {
                                Ok(()) => {
                                    app.toast(format!("Removed {}.", entry.id), ToastKind::Info)
                                }
                                Err(e) => app.toast(e, ToastKind::Error),
                            }
                        }
                        remove.on_disabled_hover_text(
                            "This is the model in use — switch to another one first.",
                        );
                    }
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.hyperlink_to(RichText::new("Source & licence").size(12.0), entry.homepage);
            });
        });
    });
}
