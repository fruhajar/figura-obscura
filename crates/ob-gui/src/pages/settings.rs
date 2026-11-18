//! The Settings page: how the app looks, and where it keeps what it remembers.
//!
//! Appearance only, deliberately. What gets censored and how lives on Tuning,
//! because that travels with a profile and belongs to the job; a colour scheme
//! belongs to the machine, and putting the two on one page invites someone to
//! export their theme to a colleague along with their thresholds.

use crate::app::{open_in_file_manager, ObApp, ToastKind};
use crate::pages;
use crate::prefs::ThemeMode;
use crate::theme::{self, Accent};
use egui::RichText;

/// Side of an accent swatch button.
const SWATCH: f32 = 26.0;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    pages::header(
        ui,
        "Settings",
        "How the app looks. These are per-machine and are not part of a profile.",
    );

    egui::ScrollArea::vertical()
        .id_salt("settings-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            appearance(app, ui);
            ui.add_space(12.0);
            storage(app, ui);
        });
}

fn appearance(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());

        theme::section(ui, "Colour scheme");
        ui.horizontal(|ui| {
            for mode in ThemeMode::ALL {
                let selected = app.prefs.theme == mode;
                let label = RichText::new(mode.label()).color(if selected {
                    p.on_accent
                } else {
                    p.text_dim
                });
                let button = egui::Button::new(label)
                    .fill(if selected { p.accent } else { p.card_hover })
                    .stroke(egui::Stroke::new(1.0_f32, p.stroke))
                    .rounding(egui::Rounding::same(theme::RADIUS))
                    .min_size(egui::vec2(84.0, 28.0));
                if ui.add(button).clicked() {
                    app.prefs.theme = mode;
                }
            }
        });
        ui.add_space(4.0);
        // Naming what System actually resolved to, because on a desktop that
        // reports nothing the setting looks like it did not take.
        let hint = match app.prefs.theme {
            ThemeMode::System => match app.system_dark {
                Some(true) => "Following the desktop, which is currently dark.",
                Some(false) => "Following the desktop, which is currently light.",
                None => {
                    "This desktop does not report a light/dark preference, so the \
                     app stays dark. Pick Light or Dark to override."
                }
            },
            ThemeMode::Light => "Light, whatever the desktop is set to.",
            ThemeMode::Dark => "Dark, whatever the desktop is set to.",
        };
        theme::hint(ui, hint);

        theme::section(ui, "Accent colour");
        theme::hint(
            ui,
            "Used for the primary action, selection and progress: the few \
             things that should stand out.",
        );
        ui.add_space(6.0);
        let dark = p.is_dark;
        ui.horizontal(|ui| {
            for accent in Accent::ALL {
                if swatch(ui, accent, dark, app.prefs.accent == accent).clicked() {
                    app.prefs.accent = accent;
                }
            }
        });
        ui.add_space(4.0);
        theme::hint(ui, app.prefs.accent.label());

        theme::section(ui, "Preview");
        sample_row(ui);
    });
}

/// One accent swatch: a filled square, ringed when it is the current choice.
fn swatch(ui: &mut egui::Ui, accent: Accent, dark: bool, selected: bool) -> egui::Response {
    let p = theme::palette();
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(SWATCH, SWATCH),
        egui::Sense::click().union(egui::Sense::hover()),
    );
    let colour = accent.swatch(dark);
    ui.painter().rect(
        rect,
        egui::Rounding::same(theme::RADIUS),
        colour,
        // The ring is the only selection cue, so it has to read against both a
        // white card and a near-black one: the text colour does, the accent's
        // own colour would not against itself.
        if selected {
            egui::Stroke::new(2.0_f32, p.text)
        } else if resp.hovered() {
            egui::Stroke::new(1.0_f32, p.stroke_strong)
        } else {
            egui::Stroke::new(1.0_f32, p.stroke)
        },
    );
    resp.on_hover_text(accent.label())
}

/// A row of the widgets the palette actually has to carry, so a scheme can be
/// judged here rather than by switching pages and looking for something that
/// went wrong.
fn sample_row(ui: &mut egui::Ui) {
    let p = theme::palette();
    ui.horizontal(|ui| {
        let _ = theme::primary_button(ui, "Run batch", true);
        let _ = theme::danger_button(ui, "Stop", true);
        let _ = ui.button("Add files…");
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        theme::pill(ui, "ready", p.success);
        theme::pill(ui, "unverified", p.warning);
        theme::pill(ui, "failed", p.danger);
    });
    ui.add_space(6.0);
    ui.add(
        egui::ProgressBar::new(0.6)
            .desired_height(14.0)
            .fill(p.accent)
            .text("12/20"),
    );
    ui.add_space(6.0);
    let mut sample = 0.42_f32;
    ui.add(egui::Slider::new(&mut sample, 0.0..=1.0).text("Confidence"));
    let mut on = true;
    ui.checkbox(&mut on, "A checkbox");
}

/// Where the settings file is, and how to get rid of it.
fn storage(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Stored settings");
        match crate::prefs::prefs_path() {
            Some(path) => {
                theme::hint(
                    ui,
                    "Your scheme, the batch's output folder and the current profile \
                     are kept in one file, written when the app closes and when a \
                     run starts.",
                );
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(path.display().to_string())
                                .monospace()
                                .size(11.5)
                                .color(theme::palette().text_dim),
                        )
                        .wrap(),
                    );
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Show folder").clicked() {
                        if let Some(dir) = path.parent() {
                            open_in_file_manager(dir);
                        }
                    }
                    if ui.button("Copy path").clicked() {
                        let text = path.display().to_string();
                        ui.output_mut(|o| o.copied_text = text);
                        app.toast("Path copied to the clipboard.", ToastKind::Info);
                    }
                    if ui.button("Reset appearance").clicked() {
                        app.prefs.theme = ThemeMode::default();
                        app.prefs.accent = Accent::default();
                        app.toast("Appearance reset.", ToastKind::Success);
                    }
                });
            }
            None => {
                // No config directory at all: the app still runs, it just
                // forgets everything between launches, and saying so is better
                // than leaving someone to discover it.
                theme::hint(
                    ui,
                    "No configuration directory could be found on this system, so \
                     nothing is saved between launches.",
                );
            }
        }
    });
}
