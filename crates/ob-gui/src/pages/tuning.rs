//! The Tuning page: which model, what it looks for, and how it covers it.
//!
//! Every settings widget renders its tooltip straight from `ob-core` metadata,
//! so the CLI's `obscura models show` help and the GUI's tooltips cannot drift (R7).

use crate::app::{ObApp, ToastKind};
use crate::pages;
use crate::theme;
use egui::RichText;
use ob_core::censor::{CensorStyle, OverlayFit};
use ob_core::geometry::{BBox, Detection};
use ob_core::profile::OnDetectFailure;
use ob_core::settings::{SettingKind, SettingValue};
use ob_core::taxonomy::{Category, Part, NUDENET_CATEGORIES};

/// Every body part, in a stable display order.
const ALL_PARTS: [Part; 9] = [
    Part::Genitalia,
    Part::Breasts,
    Part::Buttocks,
    Part::Anus,
    Part::Feet,
    Part::Belly,
    Part::Armpits,
    Part::Face,
    Part::Eyes,
];

fn part_label(p: Part) -> &'static str {
    match p {
        Part::Breasts => "Breasts",
        Part::Buttocks => "Buttocks",
        Part::Genitalia => "Genitalia",
        Part::Anus => "Anus",
        Part::Feet => "Feet",
        Part::Belly => "Belly",
        Part::Armpits => "Armpits",
        Part::Face => "Face",
        Part::Eyes => "Eyes",
    }
}

/// Below this much room for the controls, two columns are narrower than the
/// widgets in them and everything wraps; one column and a scrollbar is the
/// better trade.
const TWO_COLUMN_MIN: f32 = 620.0;

pub fn show(app: &mut ObApp, ui: &mut egui::Ui) {
    pages::header(
        ui,
        "Tuning",
        "What gets detected, and how it is covered. Hover any control for what it does.",
    );

    // The preview lives here, next to the controls, and re-renders itself as
    // they change. Tuning a detector is a tight loop of adjust-and-look; when
    // looking costs a page switch, a button press and a wait, the loop stops
    // being run and the settings get guessed at instead.
    egui::SidePanel::right("tuning-preview")
        .resizable(true)
        .default_width(400.0)
        .min_width(280.0)
        // Leave the controls enough room to stay usable however far the
        // preview is dragged out.
        .max_width((ui.available_width() - 320.0).max(320.0))
        .frame(theme::card_frame().inner_margin(egui::Margin::same(12.0)))
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("tuning-preview-scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| pages::preview::panel(app, ui));
        });

    // A CentralPanel for what is left, not a bare `ui`: a scroll area given the
    // shrunken `ui` directly still hangs its scrollbar over the side panel.
    egui::CentralPanel::default()
        .frame(egui::Frame::none())
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("tuning-controls")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Two columns where there is room — policy on the left,
                    // appearance on the right. The stacked single column this
                    // replaced ran to three screens, which is also what a
                    // narrow window gets.
                    if ui.available_width() >= TWO_COLUMN_MIN {
                        ui.columns(2, |cols| {
                            model_and_detection(app, &mut cols[0]);
                            appearance(app, &mut cols[1]);
                        });
                    } else {
                        model_and_detection(app, ui);
                        ui.add_space(10.0);
                        appearance(app, ui);
                    }
                });
        });
}

fn model_and_detection(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Model");
        model_picker(app, ui);
        ui.add_space(4.0);
        pages::model_badges(ui, app.current_entry());
        ui.add_space(6.0);
        theme::hint(ui, app.current_entry().summary);

        theme::section(ui, "Detection settings");
        settings_editors(app, ui);
    });

    ui.add_space(10.0);

    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Cross-examination");
        cross_examination(app, ui);
    });

    ui.add_space(10.0);

    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Censor which parts");
        filter_tree(app, ui);
    });

    ui.add_space(10.0);

    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Safety");
        safety(app, ui);
    });
}

fn appearance(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::card_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        theme::section(ui, "Default censor style");
        style_controls(ui, &mut app.prefs.profile.censor.default_style, "default");

        theme::section(ui, "Region shape");
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut app.prefs.profile.censor.shape.padding)
                    .speed(0.01)
                    .range(0.0..=f32::MAX)
                    .clamp_existing_to_range(true),
            );
            ui.label("Padding");
        })
        .response
        .on_hover_text(
            "Grow each box by this fraction of its size on every side before \
             censoring. Not capped at 1.0: a detector that reports a small part \
             of a region — the anime model's nipple box, say — needs several \
             times the box's own size to cover the surrounding area. Boxes are \
             still clamped to the frame.",
        );
        ui.add(
            egui::Slider::new(&mut app.prefs.profile.censor.shape.rounding, 0.0..=0.5)
                .text("Corner rounding"),
        )
        .on_hover_text("Round box corners by this fraction of the shorter side (0 = square).");

        theme::section(ui, "Per-part overrides");
        theme::hint(ui, "Give one body part a different style from the rest.");
        ui.add_space(4.0);
        for part in ALL_PARTS {
            let mut has = app.prefs.profile.censor.per_part.contains_key(&part);
            if ui.checkbox(&mut has, part_label(part)).changed() {
                if has {
                    app.prefs
                        .profile
                        .censor
                        .per_part
                        .insert(part, CensorStyle::default());
                } else {
                    app.prefs.profile.censor.per_part.remove(&part);
                }
            }
            if let Some(style) = app.prefs.profile.censor.per_part.get_mut(&part) {
                ui.indent(part_label(part), |ui| {
                    style_controls(ui, style, part_label(part));
                });
            }
        }
    });
}

fn model_picker(app: &mut ObApp, ui: &mut egui::Ui) {
    let current = app.prefs.profile.model_id.clone();
    let mut selected = current.clone();
    let display = app.current_entry().display_name;

    // Choices are listed with their install state: picking a model that is not
    // downloaded is a legitimate action (the Models page will fetch it), but it
    // should not be a surprise at Run time.
    let entries: Vec<(String, String, bool)> = app
        .registry
        .iter()
        .map(|m| {
            (
                m.id.to_string(),
                m.display_name.to_string(),
                app.downloads.is_installed(m.id),
            )
        })
        .collect();

    egui::ComboBox::from_id_salt("model-picker")
        .selected_text(display)
        .width(ui.available_width() - 10.0)
        .show_ui(ui, |ui| {
            for (id, name, installed) in &entries {
                let label = if *installed {
                    name.clone()
                } else {
                    format!("{name}  (not installed)")
                };
                ui.selectable_value(&mut selected, id.clone(), label);
            }
        });

    if selected != current {
        app.select_model(&selected);
        if !app.model_ready() {
            app.toast(
                "That model is not downloaded yet — get it on the Models page.",
                ToastKind::Info,
            );
        }
    }
}

/// Running more than one model over each frame, and how much agreement is
/// required before a region is censored.
///
/// The CLI equivalent is `--also-model` plus `--min-votes`. The default stays
/// one model: a second one roughly doubles the time per frame, which on a long
/// batch is the difference between overnight and two nights.
fn cross_examination(app: &mut ObApp, ui: &mut egui::Ui) {
    theme::hint(
        ui,
        "Run more than one model over every frame. More models find more, and \
         asking them to agree finds less but with fewer false positives. Each \
         extra model costs another pass over each frame.",
    );
    ui.add_space(6.0);

    let primary = app.prefs.profile.model_id.clone();

    // Collect before touching `app` mutably: the checkbox loop needs `&mut app`
    // and must not be holding a borrow of `app.registry` while it runs.
    let mut candidates: Vec<(String, String, bool)> = app
        .registry
        .iter()
        .filter(|m| m.id != primary)
        .map(|m| {
            (
                m.id.to_string(),
                m.display_name.to_string(),
                app.downloads.is_installed(m.id),
            )
        })
        .collect();
    // A model can be selected and then uninstalled. Keep it on screen (marked)
    // rather than dropping it silently, or the run would fail against a
    // companion the user can no longer see, let alone clear.
    candidates.retain(|(id, _, installed)| *installed || app.prefs.extra_models.contains(id));

    if candidates.is_empty() {
        theme::hint(
            ui,
            "No other models are installed — add one on the Models page to \
             cross-examine with.",
        );
        return;
    }

    for (id, name, installed) in &candidates {
        let mut on = app.prefs.extra_models.contains(id);
        let label = if *installed {
            name.clone()
        } else {
            format!("{name}  (not installed)")
        };
        if ui.checkbox(&mut on, label).changed() {
            if on {
                app.prefs.extra_models.push(id.clone());
            } else {
                app.prefs.extra_models.retain(|x| x != id);
            }
        }
        if !*installed && on {
            theme::hint(
                ui,
                "This model is selected but not downloaded — the run will fail \
                 until you get it on the Models page or clear it here.",
            );
        }
    }

    let spec = app.ensemble_spec();
    if !spec.is_ensemble() {
        return;
    }

    ui.add_space(8.0);
    let members = spec.members();
    // Bound the control to what the current member count can satisfy, so the
    // saved value never outruns the models it refers to.
    let mut votes = spec.effective_votes();
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut votes, 1..=members).clamping(egui::SliderClamping::Always))
            .on_hover_text(
                "How many models must independently find a region before it is \
                 censored. 1 censors anything any model saw.",
            );
        ui.label("Models that must agree");
    });
    if votes != app.prefs.min_votes {
        app.prefs.min_votes = votes;
    }

    theme::hint(
        ui,
        &match votes {
            1 => format!(
                "Any of the {members} models is enough. The widest coverage, and \
                 the safest default for censoring."
            ),
            v if v == members => format!(
                "All {members} must agree. Expect misses: a region only one model \
                 saw is left uncensored."
            ),
            v => format!("{v} of {members} must agree."),
        },
    );
    theme::hint(
        ui,
        "Votes are counted per category among the models that can report it, so \
         pairing a 3-class anime model with an 18-class one never deletes the \
         categories only one of them covers.",
    );
}

/// The R7 payoff: each widget's tooltip comes from the setting metadata.
fn settings_editors(app: &mut ObApp, ui: &mut egui::Ui) {
    // Clone the metadata so the editors can borrow `app.settings` mutably.
    let settings = app.current_entry().settings.clone();
    for setting in &settings {
        let value = app
            .settings
            .entry(setting.key.to_string())
            .or_insert_with(|| setting.default.clone());

        let resp = match (&setting.kind, value) {
            (SettingKind::Float { min, max, step }, SettingValue::Float(v)) => ui.add(
                egui::Slider::new(v, *min..=*max)
                    .step_by(*step)
                    .text(setting.label),
            ),
            (SettingKind::Int { min, max, .. }, SettingValue::Int(v)) => {
                // Spin box, not a slider: the metadata bound is a sanity limit
                // (e.g. 256 tiles), not a range worth scrubbing.
                ui.horizontal(|ui| {
                    let r = ui.add(
                        egui::DragValue::new(v)
                            .range(*min..=*max)
                            .clamp_existing_to_range(true),
                    );
                    ui.label(setting.label);
                    r
                })
                .inner
            }
            (SettingKind::Bool, SettingValue::Bool(v)) => ui.checkbox(v, setting.label),
            (SettingKind::Enum { choices }, SettingValue::Text(v)) => {
                ui.horizontal(|ui| {
                    let r = egui::ComboBox::from_id_salt(setting.key)
                        .selected_text(v.clone())
                        .show_ui(ui, |ui| {
                            for c in choices {
                                ui.selectable_value(v, (*c).to_string(), *c);
                            }
                        })
                        .response;
                    ui.label(setting.label);
                    r
                })
                .inner
            }
            _ => ui.label(format!("{}: (editor pending)", setting.label)),
        };
        resp.on_hover_text(setting.tooltip);
    }

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button("Reset to defaults").clicked() {
            app.settings = ob_core::settings::defaults(&settings);
            app.toast("Detection settings reset.", ToastKind::Info);
        }
        theme::hint(
            ui,
            "Per-model. Each model publishes its own best threshold.",
        );
    });

    app.sync_profile();
}

/// Render the canonical taxonomy as a checkbox tree bound to the filter.
///
/// Each of the 18 NudeNet categories is checked iff the current policy would
/// select it; toggling rebuilds `filter.rules` as an explicit set of exact
/// per-category rules, which preserves selection semantics.
fn filter_tree(app: &mut ObApp, ui: &mut egui::Ui) {
    let p = theme::palette();
    let mut enabled: Vec<(Category, bool)> = NUDENET_CATEGORIES
        .iter()
        .map(|c| (*c, app.prefs.profile.filter.selects(&probe_det(*c))))
        .collect();
    let mut changed = false;

    // Bulk actions: 18 checkboxes is a lot of clicking to get to "everything"
    // or "nothing", and both are common starting points.
    ui.horizontal(|ui| {
        if ui.button("Select all").clicked() {
            for e in enabled.iter_mut() {
                e.1 = true;
            }
            changed = true;
        }
        if ui.button("Select none").clicked() {
            for e in enabled.iter_mut() {
                e.1 = false;
            }
            changed = true;
        }
        let on = enabled.iter().filter(|(_, v)| *v).count();
        ui.label(
            RichText::new(format!("{on}/{} categories", enabled.len()))
                .size(12.0)
                .color(p.text_faint),
        );
    });
    ui.add_space(4.0);

    for part in ALL_PARTS {
        let idxs: Vec<usize> = enabled
            .iter()
            .enumerate()
            .filter(|(_, (c, _))| c.part == part)
            .map(|(i, _)| i)
            .collect();
        // Eyes has no NudeNet class, so it contributes nothing to display.
        if idxs.is_empty() {
            continue;
        }
        let on_here = idxs.iter().filter(|i| enabled[**i].1).count();
        egui::CollapsingHeader::new(format!("{}  ({on_here}/{})", part_label(part), idxs.len()))
            .id_salt(part_label(part))
            .default_open(matches!(
                part,
                Part::Genitalia | Part::Breasts | Part::Buttocks | Part::Anus
            ))
            .show(ui, |ui| {
                for i in idxs {
                    let (cat, on) = &mut enabled[i];
                    let label = cat.to_string();
                    if ui.checkbox(on, label).changed() {
                        changed = true;
                    }
                }
            });
    }

    if changed {
        app.prefs.profile.filter.rules = enabled
            .iter()
            .filter(|(_, on)| *on)
            .map(|(c, _)| ob_core::filter::FilterRule::exact(*c))
            .collect();
    }

    ui.add_space(6.0);
    ui.add(egui::Slider::new(&mut app.prefs.profile.filter.min_score, 0.0..=1.0).text("Min score"))
        .on_hover_text(
            "Detections below this confidence are never censored (unless a rule overrides it).",
        );
}

fn safety(app: &mut ObApp, ui: &mut egui::Ui) {
    egui::ComboBox::from_label("On detection failure")
        .selected_text(failure_label(app.prefs.profile.on_detect_failure))
        .show_ui(ui, |ui| {
            for opt in [
                OnDetectFailure::Blank,
                OnDetectFailure::Skip,
                OnDetectFailure::PassThrough,
            ] {
                ui.selectable_value(
                    &mut app.prefs.profile.on_detect_failure,
                    opt,
                    failure_label(opt),
                );
            }
        })
        .response
        .on_hover_text(
            "What to do when a frame can't be analyzed: Blank obliterates it \
             (safest), Skip drops the file, Pass-through emits it uncensored.",
        );

    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(&mut app.prefs.detect_every)
                .range(1..=u32::MAX)
                .clamp_existing_to_range(true),
        );
        ui.label("Video: detect every N frames");
    })
    .response
    .on_hover_text(
        "Run the detector on every Nth video frame; the tracker coasts between \
         for speed. Uncapped, because tiled detection makes a pass several \
         times more expensive and a long clip may be worth a much sparser \
         detect interval.",
    );

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if ui.button("Export profile…").clicked() {
            export_profile(app);
        }
        if ui.button("Import profile…").clicked() {
            import_profile(app);
        }
    });
    theme::hint(
        ui,
        "Profiles are the same format `obscura process --profile` reads.",
    );
}

fn export_profile(app: &mut ObApp) {
    app.sync_profile();
    let Some(path) = rfd::FileDialog::new()
        .set_file_name("figura-obscura-profile.json")
        .add_filter("JSON", &["json"])
        .save_file()
    else {
        return;
    };
    match app
        .prefs
        .profile
        .to_json()
        .map_err(|e| e.to_string())
        .and_then(|j| std::fs::write(&path, j).map_err(|e| e.to_string()))
    {
        Ok(()) => app.toast(format!("Saved to {}", path.display()), ToastKind::Success),
        Err(e) => app.toast(format!("Could not save: {e}"), ToastKind::Error),
    }
}

fn import_profile(app: &mut ObApp) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("JSON", &["json"])
        .pick_file()
    else {
        return;
    };
    let loaded = std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
        .and_then(|t| ob_core::profile::Profile::from_json(&t).map_err(|e| e.to_string()));
    match loaded {
        Ok(profile) => {
            let model_id = profile.model_id.clone();
            // A profile can name a model this build does not have; keep the
            // rest of it rather than rejecting the whole file.
            let known = app.registry.iter().any(|m| m.id == model_id);
            app.prefs.profile = profile;
            if !known {
                app.prefs.profile.model_id = app.registry[0].id.to_string();
                app.toast(
                    format!("Imported, but model `{model_id}` is unknown — using the default."),
                    ToastKind::Info,
                );
            } else {
                app.toast("Profile imported.", ToastKind::Success);
            }
            // Re-layer settings against the (possibly new) model's metadata.
            let entry = app.current_entry().clone();
            let mut settings = ob_core::settings::defaults(&entry.settings);
            for (k, v) in &app.prefs.profile.model_settings {
                if entry.settings.iter().any(|s| s.key == k.as_str()) {
                    settings.insert(k.clone(), v.clone());
                }
            }
            app.settings = settings;
            app.preview = None;
        }
        Err(e) => app.toast(format!("Could not read profile: {e}"), ToastKind::Error),
    }
}

/// A unit-score detection used only to test the filter's current selection.
fn probe_det(category: Category) -> Detection {
    Detection {
        bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
        category,
        score: 1.0,
    }
}

fn failure_label(f: OnDetectFailure) -> &'static str {
    match f {
        OnDetectFailure::Blank => "Blank frame (safest)",
        OnDetectFailure::Skip => "Skip file",
        OnDetectFailure::PassThrough => "Pass through (uncensored)",
    }
}

fn style_name(s: &CensorStyle) -> &'static str {
    match s {
        CensorStyle::SolidFill { .. } => "Solid fill",
        CensorStyle::Pixelate { .. } => "Pixelate",
        CensorStyle::Blur { .. } => "Blur",
        CensorStyle::ImageOverlay { .. } => "Image overlay",
    }
}

/// Draw the style-type combo and the parameter widgets for the current variant.
/// `id_salt` disambiguates widgets when several editors share a panel.
pub fn style_controls(ui: &mut egui::Ui, style: &mut CensorStyle, id_salt: &str) {
    let current = style_name(style);
    let mut chosen = current;
    egui::ComboBox::from_id_salt(format!("style-{id_salt}"))
        .selected_text(current)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut chosen, "Solid fill", "Solid fill");
            ui.selectable_value(&mut chosen, "Pixelate", "Pixelate");
            ui.selectable_value(&mut chosen, "Blur", "Blur");
            ui.selectable_value(&mut chosen, "Image overlay", "Image overlay");
        });
    if chosen != current {
        *style = match chosen {
            "Solid fill" => CensorStyle::SolidFill {
                color: [0, 0, 0, 255],
            },
            "Pixelate" => CensorStyle::Pixelate { block: 16 },
            "Blur" => CensorStyle::Blur { sigma: 8.0 },
            _ => CensorStyle::ImageOverlay {
                path: String::new(),
                fit: OverlayFit::Cover,
                opacity: 1.0,
            },
        };
    }

    match style {
        CensorStyle::SolidFill { color } => {
            ui.horizontal(|ui| {
                let mut c =
                    egui::Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3]);
                if ui.color_edit_button_srgba(&mut c).changed() {
                    *color = [c.r(), c.g(), c.b(), c.a()];
                }
                ui.label("Fill colour");
            });
        }
        CensorStyle::Pixelate { block } => {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(block)
                        .range(2..=u32::MAX)
                        .clamp_existing_to_range(true),
                );
                ui.label("Block size (px)");
            })
            .response
            .on_hover_text(
                "Mosaic cell size. A block that looked coarse on a 720p frame is \
                 barely visible on a 4K one, so this is not capped at a preset \
                 maximum — scale it with the source.",
            );
        }
        CensorStyle::Blur { sigma } => {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(sigma)
                        .speed(0.25)
                        .range(0.5..=f32::MAX)
                        .clamp_existing_to_range(true),
                );
                ui.label("Blur strength (σ)");
            })
            .response
            .on_hover_text(
                "Gaussian sigma. Large regions on high-resolution sources need \
                 far more than the old 50 cap to be genuinely unreadable.",
            );
        }
        CensorStyle::ImageOverlay { path, fit, opacity } => {
            ui.horizontal(|ui| {
                if ui.button("Overlay image…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_file() {
                        *path = p.display().to_string();
                    }
                }
                ui.label(if path.is_empty() {
                    "(none selected)".to_string()
                } else {
                    path.clone()
                });
            });
            egui::ComboBox::from_id_salt(format!("fit-{id_salt}"))
                .selected_text(format!("{fit:?}"))
                .show_ui(ui, |ui| {
                    for f in [
                        OverlayFit::Cover,
                        OverlayFit::Contain,
                        OverlayFit::Stretch,
                        OverlayFit::Tile,
                    ] {
                        ui.selectable_value(fit, f, format!("{f:?}"));
                    }
                });
            ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
        }
    }
}
