//! The About page: version, credits, licences and support information.
//!
//! Not decoration. The detector weights come from third parties under their own
//! licences, and a product that is sold has to say so plainly and name them.

use crate::app::{ObApp, ToastKind};
use crate::pages;
use crate::theme;
use egui::RichText;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    pages::header(ui, "About", "Version, credits and where things live.");

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            theme::card_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Figura Obscura").size(18.0).strong());
                ui.label(
                    RichText::new(format!("version {}", env!("CARGO_PKG_VERSION")))
                        .color(p.text_dim),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(
                        "Offline batch censoring for images and video. Detection runs \
                         entirely on your machine; the only time Figura Obscura uses the \
                         network is to download a detector model.",
                    )
                    .color(p.text_dim),
                );
            });

            ui.add_space(10.0);

            theme::card_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::section(ui, "Detector models");
                theme::hint(
                    ui,
                    "The weights are third-party work, used under the licences below.",
                );
                ui.add_space(6.0);
                let registry = std::mem::take(&mut app.registry);
                for m in &registry {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(m.display_name).size(12.5));
                        pages::license_pill(ui, m.license);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.hyperlink_to(RichText::new("project page").size(12.0), m.homepage);
                        });
                    });
                }
                app.registry = registry;
            });

            ui.add_space(10.0);

            theme::card_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::section(ui, "Built with");
                for (name, what, url) in [
                    (
                        "ONNX Runtime",
                        "model inference (MIT)",
                        "https://onnxruntime.ai",
                    ),
                    (
                        "FFmpeg",
                        // Naming FFmpeg and its licence is itself a licence
                        // obligation when a build is bundled. The exact terms
                        // depend on which build shipped, so point at the file
                        // that travels with it.
                        "video decode and encode (LGPL-2.1 or GPL — see THIRD-PARTY.md)",
                        "https://ffmpeg.org",
                    ),
                    (
                        "egui / eframe",
                        "this interface (MIT / Apache-2.0)",
                        "https://github.com/emilk/egui",
                    ),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(name).size(12.5).strong());
                        ui.label(RichText::new(what).size(12.0).color(p.text_dim));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.hyperlink_to(RichText::new("site").size(12.0), url);
                        });
                    });
                }
            });

            ui.add_space(10.0);

            theme::card_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::section(ui, "Locations");
                // Support questions are almost always "where did it put X".
                // Copyable rows answer that without a support round-trip.
                path_row(
                    app,
                    ui,
                    "Models",
                    ob_models::cache_dir()
                        .map(|d| d.display().to_string())
                        .unwrap_or_else(|_| "unavailable".into()),
                );
                path_row(
                    app,
                    ui,
                    "Settings",
                    crate::prefs::prefs_path()
                        .map(|d| d.display().to_string())
                        .unwrap_or_else(|| "unavailable".into()),
                );
                let ffmpeg = ob_media::tools::ffmpeg();
                path_row(app, ui, "ffmpeg", ffmpeg.display().to_string());
                // Whether the shipped copy or the user's own is in use answers
                // both "why does video behave oddly" and "what am I actually
                // redistributing", so it is stated rather than implied.
                let bundled = ffmpeg.is_absolute();
                ui.label(
                    egui::RichText::new(if bundled {
                        "using the copy that shipped with Figura Obscura"
                    } else {
                        "using your system ffmpeg, found on PATH"
                    })
                    .size(11.5)
                    .color(p.text_faint),
                );
            });

            ui.add_space(10.0);

            theme::card_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                theme::section(ui, "Reset");
                theme::hint(
                    ui,
                    "Restores every setting to its default. Downloaded models are kept.",
                );
                ui.add_space(6.0);
                if theme::danger_button(ui, "Reset all settings", !app.is_running()).clicked() {
                    let keep_output = app.prefs.output_dir.clone();
                    app.prefs = crate::prefs::Prefs {
                        // Don't re-run the welcome wizard for an existing user
                        // who only wanted their sliders back at defaults.
                        setup_done: true,
                        output_dir: keep_output,
                        ..Default::default()
                    };
                    let entry = app.current_entry().clone();
                    app.settings = ob_core::settings::defaults(&entry.settings);
                    app.preview = None;
                    let _ = app.prefs.save();
                    app.toast("Settings reset to defaults.", ToastKind::Success);
                }
            });
        });
}

fn path_row(app: &mut ObApp, ui: &mut egui::Ui, label: &str, value: String) {
    let p = theme::palette();
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(12.5).color(p.text_dim));
        ui.label(RichText::new(&value).size(12.0).monospace());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Copy").clicked() {
                ui.output_mut(|o| o.copied_text = value.clone());
                app.toast("Copied.", ToastKind::Info);
            }
        });
    });
}
