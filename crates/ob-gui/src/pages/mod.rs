//! One module per page in the window, plus the first-run setup screen.

pub mod about;
pub mod models;
pub mod preview;
pub mod queue;
pub mod setup;
pub mod tuning;

use crate::theme;
use ob_core::registry::{Domain, License, ModelEntry};

/// Page title plus a one-line explanation of what the page is for.
pub fn header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.label(egui::RichText::new(title).heading().strong());
    ui.label(
        egui::RichText::new(subtitle)
            .size(12.5)
            .color(theme::palette().text_dim),
    );
    ui.add_space(12.0);
}

/// Domain badge. Which content a model was trained on is the single most
/// consequential choice in the app — a photographic detector on illustration
/// finds very little — so it is shown as a coloured chip, not buried in prose.
pub fn domain_pill(ui: &mut egui::Ui, domain: Domain) {
    let p = theme::palette();
    match domain {
        Domain::RealLife => theme::pill(ui, "photo", p.accent_hover),
        Domain::Anime => theme::pill(ui, "drawn", p.success),
    }
}

/// License badge. Permissive licences are informational; a non-commercial one
/// is a warning, because this app is sold and its users may be selling too.
pub fn license_pill(ui: &mut egui::Ui, license: License) {
    let p = theme::palette();
    let (text, color) = match license {
        License::Apache2 => ("Apache-2.0", p.text_dim),
        License::Mit => ("MIT", p.text_dim),
        License::OpenRail => ("OpenRAIL", p.warning),
        License::NonCommercial => ("non-commercial", p.danger),
        License::Unknown => ("licence unknown", p.warning),
    };
    theme::pill(ui, text, color);
}

/// The badge row shown under a model's name.
pub fn model_badges(ui: &mut egui::Ui, entry: &ModelEntry) {
    ui.horizontal(|ui| {
        domain_pill(ui, entry.domain);
        license_pill(ui, entry.license);
        theme::pill(
            ui,
            &format!("{}px", entry.input_size),
            theme::palette().text_faint,
        );
        if entry.default_download {
            theme::pill(ui, "recommended", theme::palette().accent);
        }
    });
}
