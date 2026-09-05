//! Visual identity for the Figura Obscura desktop app.
//!
//! egui's stock look is a grey developer theme; a product someone paid for
//! should not ship it. Everything here is one dark palette applied to egui's
//! `Style`/`Visuals`, plus a small set of drawing helpers (`section`, `card`,
//! `pill`) the pages share so spacing and weight stay consistent between them.
//!
//! Dark only, deliberately. The palette is a single struct, so a light variant
//! is a matter of supplying a second one — but shipping one properly tuned
//! theme beats shipping two half-tuned ones.

use egui::{Color32, FontFamily, FontId, Rounding, Stroke, TextStyle};

/// Every colour the app uses. Nothing outside this struct hard-codes one.
pub struct Palette {
    /// Window background, behind everything.
    pub bg: Color32,
    /// Side rail and status bar.
    pub panel: Color32,
    /// Raised surfaces: model cards, queue rows, the preview frame.
    pub card: Color32,
    /// A card under the pointer.
    pub card_hover: Color32,
    /// Slider rails, checkbox boxes, progress-bar troughs — anything that must
    /// stay visible *on top of* a card.
    pub track: Color32,
    pub track_hover: Color32,
    /// Hairlines between surfaces.
    pub stroke: Color32,
    /// Borders that need to be seen (focused fields, selected rows).
    pub stroke_strong: Color32,
    /// Primary body text.
    pub text: Color32,
    /// Secondary text: units, hints, metadata.
    pub text_dim: Color32,
    /// Tertiary text: placeholders, disabled labels.
    pub text_faint: Color32,
    /// The brand colour. Primary actions and selection only — if everything is
    /// accented, nothing is.
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_press: Color32,
    /// Text drawn on top of `accent`.
    pub on_accent: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub danger: Color32,
}

pub const DARK: Palette = Palette {
    bg: Color32::from_rgb(0x10, 0x10, 0x15),
    panel: Color32::from_rgb(0x17, 0x17, 0x1E),
    card: Color32::from_rgb(0x1E, 0x1E, 0x27),
    card_hover: Color32::from_rgb(0x26, 0x26, 0x33),
    track: Color32::from_rgb(0x35, 0x35, 0x45),
    track_hover: Color32::from_rgb(0x42, 0x42, 0x55),
    stroke: Color32::from_rgb(0x2E, 0x2E, 0x3B),
    stroke_strong: Color32::from_rgb(0x3D, 0x3D, 0x4E),
    text: Color32::from_rgb(0xE8, 0xE8, 0xF0),
    text_dim: Color32::from_rgb(0x9C, 0x9C, 0xB0),
    text_faint: Color32::from_rgb(0x6E, 0x6E, 0x82),
    accent: Color32::from_rgb(0x7C, 0x6C, 0xFF),
    accent_hover: Color32::from_rgb(0x93, 0x85, 0xFF),
    accent_press: Color32::from_rgb(0x6A, 0x58, 0xF0),
    on_accent: Color32::from_rgb(0xF5, 0xF4, 0xFF),
    success: Color32::from_rgb(0x4A, 0xDE, 0x80),
    warning: Color32::from_rgb(0xFB, 0xBF, 0x24),
    danger: Color32::from_rgb(0xF8, 0x71, 0x71),
};

/// The palette in force. A single constant today; the indirection is what a
/// future light mode would swap.
pub fn palette() -> &'static Palette {
    &DARK
}

/// Corner radius for interactive widgets.
///
/// Kept small on purpose. egui applies this to *every* widget including
/// checkboxes, and at 7px a 14px checkbox comes out perfectly circular — which
/// reads as a radio button, i.e. as a mutually exclusive choice. 4px is
/// unambiguously a rounded square at that size and still soft on a 30px button.
pub const RADIUS: f32 = 4.0;

/// Corner radius for cards and panels, which are large enough to carry more.
pub const CARD_RADIUS: f32 = 8.0;

/// Every non-ASCII glyph the interface draws.
///
/// Centralised because egui's bundled fonts (Ubuntu-Light, NotoEmoji and an
/// icon font) cover an awkward subset of the symbol blocks: `▶` and `▣` are
/// present, `●`, `◈`, `◍`, `▤`, `✕` and `↓` are not, and a missing glyph
/// renders as a tofu box with no warning at build time. `glyphs_are_all_covered`
/// asserts every entry here is representable, so a tofu box fails the test
/// suite instead of shipping.
pub mod glyph {
    /// Nav rail, one per page.
    pub const NAV_BATCH: &str = "▶";
    pub const NAV_TUNING: &str = "⚙";
    pub const NAV_MODELS: &str = "▣";
    pub const NAV_ABOUT: &str = "ℹ";

    /// Queue rows, by media kind.
    pub const FOLDER: &str = "📂";
    pub const IMAGE: &str = "■";
    pub const VIDEO: &str = "▶";
    /// A file the batch will skip.
    pub const UNSUPPORTED: &str = "⚠";

    /// Status marks.
    pub const OK: &str = "✔";
    pub const FAIL: &str = "✖";
    pub const WARN: &str = "⚠";
    pub const PENDING: &str = "○";
    pub const DOT: &str = "■";
    /// Remove-from-list button.
    pub const REMOVE: &str = "✖";
    /// The file currently being processed.
    pub const RUNNING: &str = "▶";

    /// Everything above, for `glyphs_are_all_covered_by_the_bundled_fonts`.
    /// Test-only, so it does not sit in the shipped binary as dead data.
    #[cfg(test)]
    pub const ALL: &[&str] = &[
        NAV_BATCH,
        NAV_TUNING,
        NAV_MODELS,
        NAV_ABOUT,
        FOLDER,
        IMAGE,
        VIDEO,
        UNSUPPORTED,
        OK,
        FAIL,
        WARN,
        PENDING,
        DOT,
        REMOVE,
        RUNNING,
    ];
}

/// Install the theme on a freshly created context.
pub fn install(ctx: &egui::Context) {
    let p = palette();
    let mut style = (*ctx.style()).clone();

    // --- type scale ------------------------------------------------------
    // A real scale rather than egui's two sizes: headings, body, and a small
    // size for metadata, so hierarchy is carried by type instead of by colour
    // alone.
    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(19.0, FontFamily::Proportional),
        ),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (
            TextStyle::Button,
            FontId::new(14.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(12.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(12.5, FontFamily::Monospace),
        ),
    ]
    .into();

    // --- spacing ---------------------------------------------------------
    let s = &mut style.spacing;
    s.item_spacing = egui::vec2(8.0, 8.0);
    s.button_padding = egui::vec2(12.0, 6.0);
    s.menu_margin = egui::Margin::same(6.0);
    s.indent = 18.0;
    s.interact_size.y = 26.0;
    s.slider_width = 180.0;
    s.combo_width = 180.0;
    s.scroll.bar_width = 9.0;
    s.scroll.floating = false;

    // --- colours ---------------------------------------------------------
    let v = &mut style.visuals;
    v.dark_mode = true;
    v.panel_fill = p.panel;
    v.window_fill = p.bg;
    v.extreme_bg_color = p.bg;
    v.faint_bg_color = p.card;
    v.window_stroke = Stroke::new(1.0_f32, p.stroke);
    v.window_rounding = Rounding::same(10.0);
    v.override_text_color = Some(p.text);
    v.hyperlink_color = p.accent_hover;
    v.selection = egui::style::Selection {
        bg_fill: p.accent.gamma_multiply(0.35),
        stroke: Stroke::new(1.0_f32, p.accent_hover),
    };

    // Widget states. `weak_bg_fill` is the resting button colour in egui;
    // `bg_fill` is used by toggles and selected states.
    let rounding = Rounding::same(RADIUS);
    v.widgets.noninteractive.bg_fill = p.track;
    v.widgets.noninteractive.weak_bg_fill = p.card;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, p.stroke);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, p.text_dim);
    v.widgets.noninteractive.rounding = rounding;

    // `bg_fill` is the *track* colour: egui paints slider rails, checkbox boxes
    // and radio dots with it, while buttons use `weak_bg_fill`. Leaving both at
    // the card colour made every slider rail invisible against the card it sat
    // on — the handle appeared to float in empty space.
    v.widgets.inactive.bg_fill = p.track;
    v.widgets.inactive.weak_bg_fill = p.card;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, p.stroke);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, p.text);
    v.widgets.inactive.rounding = rounding;

    v.widgets.hovered.bg_fill = p.track_hover;
    v.widgets.hovered.weak_bg_fill = p.card_hover;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, p.stroke_strong);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, p.text);
    v.widgets.hovered.rounding = rounding;
    v.widgets.hovered.expansion = 0.0;

    v.widgets.active.bg_fill = p.accent_press;
    v.widgets.active.weak_bg_fill = p.accent_press;
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, p.accent);
    v.widgets.active.fg_stroke = Stroke::new(1.0_f32, p.on_accent);
    v.widgets.active.rounding = rounding;
    v.widgets.active.expansion = 0.0;

    v.widgets.open.bg_fill = p.track_hover;
    v.widgets.open.weak_bg_fill = p.card_hover;
    v.widgets.open.bg_stroke = Stroke::new(1.0_f32, p.stroke_strong);
    v.widgets.open.fg_stroke = Stroke::new(1.0_f32, p.text);
    v.widgets.open.rounding = rounding;

    // Shadows off: they read as blur against a near-black background and cost
    // fill rate for nothing.
    v.popup_shadow = egui::epaint::Shadow::NONE;
    v.window_shadow = egui::epaint::Shadow::NONE;

    ctx.set_style(style);
}

// ---------------------------------------------------------------------------
// Shared drawing helpers
// ---------------------------------------------------------------------------

/// A small uppercase section label — the app's one piece of typographic
/// texture, used to break long settings columns into scannable groups.
pub fn section(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(11.0)
            .color(palette().text_faint)
            .strong(),
    );
    ui.add_space(2.0);
}

/// Secondary explanatory text under a control.
pub fn hint(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .size(12.0)
            .color(palette().text_dim),
    );
}

/// A raised, outlined container. `Frame` rather than a group so the padding and
/// radius match the rest of the app.
pub fn card_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(palette().card)
        .stroke(Stroke::new(1.0_f32, palette().stroke))
        .rounding(Rounding::same(CARD_RADIUS))
        .inner_margin(egui::Margin::symmetric(14.0, 12.0))
}

/// A compact coloured label: license badges, domain badges, status chips.
pub fn pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        FontId::new(11.0, FontFamily::Proportional),
        color,
    );
    let padding = egui::vec2(7.0, 3.0);
    let (rect, _) = ui.allocate_exact_size(galley.size() + padding * 2.0, egui::Sense::hover());
    ui.painter().rect(
        rect,
        Rounding::same(4.0),
        color.gamma_multiply(0.16),
        Stroke::new(1.0_f32, color.gamma_multiply(0.45)),
    );
    ui.painter().galley(rect.min + padding, galley, color);
}

/// A filled, accent-coloured primary action button.
///
/// egui has no notion of a primary button, so this paints one explicitly
/// rather than leaving the most important control in the window looking
/// identical to "Clear".
pub fn primary_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> egui::Response {
    let p = palette();
    let text = egui::RichText::new(text)
        .color(if enabled { p.on_accent } else { p.text_faint })
        .strong();
    let fill = if enabled { p.accent } else { p.card };
    ui.add_enabled(
        enabled,
        egui::Button::new(text)
            .fill(fill)
            .rounding(Rounding::same(RADIUS))
            .min_size(egui::vec2(0.0, 30.0)),
    )
}

/// A destructive/stop action: outlined in `danger` rather than filled, so it
/// reads as available without shouting over the primary action beside it.
pub fn danger_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> egui::Response {
    let p = palette();
    ui.add_enabled(
        enabled,
        egui::Button::new(egui::RichText::new(text).color(p.danger))
            .fill(p.card)
            .stroke(Stroke::new(1.0_f32, p.danger.gamma_multiply(0.55)))
            .rounding(Rounding::same(RADIUS))
            .min_size(egui::vec2(0.0, 30.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_sets_the_palette_and_type_scale() {
        let ctx = egui::Context::default();
        install(&ctx);
        let style = ctx.style();
        assert!(style.visuals.dark_mode);
        assert_eq!(style.visuals.panel_fill, DARK.panel);
        assert_eq!(style.visuals.override_text_color, Some(DARK.text));
        // Five named text styles, so no widget falls back to a default size.
        assert_eq!(style.text_styles.len(), 5);
        assert!(
            style.text_styles[&TextStyle::Heading].size > style.text_styles[&TextStyle::Body].size
        );
    }

    #[test]
    fn glyphs_are_all_covered_by_the_bundled_fonts() {
        // egui ships Ubuntu-Light plus an emoji/icon font, and their coverage of
        // the symbol blocks is uneven and undocumented: `▶` and `▣` are there,
        // `●`, `◈`, `◍`, `▤`, `✕` and `↓` are not. An uncovered character draws
        // as a tofu box with no compile-time or run-time warning, so the only
        // way to keep them out of a release is to assert coverage.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let font = FontId::new(14.0, FontFamily::Proportional);

        let missing: Vec<&str> = ctx.fonts(|f| {
            glyph::ALL
                .iter()
                .copied()
                .filter(|g| !f.has_glyphs(&font, g))
                .collect()
        });
        assert!(
            missing.is_empty(),
            "these glyphs would render as tofu boxes: {missing:?}"
        );
    }

    #[test]
    fn checkboxes_cannot_come_out_circular() {
        // egui applies the widget radius to checkboxes too. At a radius of half
        // the ~14px box they become circles, which reads as a radio button —
        // a mutually exclusive choice, which none of these are.
        assert!(
            RADIUS < 7.0,
            "RADIUS {RADIUS} would round a 14px checkbox into a circle"
        );
        assert!(CARD_RADIUS > RADIUS, "cards should be softer than controls");
    }

    #[test]
    fn control_tracks_are_distinct_from_the_cards_they_sit_on() {
        // Slider rails, checkbox boxes and progress troughs are painted with
        // `bg_fill`. When that equalled the card colour every rail vanished and
        // the handle appeared to float in empty space.
        let ctx = egui::Context::default();
        install(&ctx);
        let w = &ctx.style().visuals.widgets;
        assert_ne!(w.inactive.bg_fill, DARK.card);
        assert_ne!(w.noninteractive.bg_fill, DARK.card);
        assert_ne!(w.hovered.bg_fill, DARK.card_hover);
    }

    #[test]
    fn every_widget_state_shares_one_corner_radius() {
        // Mixed radii are the most visible way a hand-built theme looks
        // unfinished, so this is pinned rather than left to inspection.
        let ctx = egui::Context::default();
        install(&ctx);
        let w = &ctx.style().visuals.widgets;
        for r in [
            w.noninteractive.rounding,
            w.inactive.rounding,
            w.hovered.rounding,
            w.active.rounding,
            w.open.rounding,
        ] {
            assert_eq!(r, Rounding::same(RADIUS));
        }
    }
}
