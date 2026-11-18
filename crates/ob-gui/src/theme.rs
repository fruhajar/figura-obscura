//! Visual identity for the Figura Obscura desktop app.
//!
//! egui's stock look is a grey developer theme; a product someone paid for
//! should not ship it. Everything here is one palette applied to egui's
//! `Style`/`Visuals`, plus a small set of drawing helpers (`section`, `card`,
//! `pill`) the pages share so spacing and weight stay consistent between them.
//!
//! The palette is composed, not hard-coded: a neutral [`Base`] (two of them,
//! dark and light) crossed with an [`Accent`]. Every combination is built once
//! and [`palette`] returns whichever one [`set_scheme`] last selected, so the
//! several hundred `palette().text` call sites need to know nothing about the
//! user's choice. `DARK` is combination zero, which is what the app looked like
//! before any of this was configurable.

use egui::{Color32, FontFamily, FontId, Rounding, Stroke, TextStyle};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Every colour the app uses. Nothing outside this struct hard-codes one.
pub struct Palette {
    /// Whether this is a dark scheme. Handed to egui as `Visuals::dark_mode`,
    /// which decides how it tints the few things it paints itself.
    pub is_dark: bool,
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

/// The neutral half of a palette: everything that is not the accent.
struct Base {
    is_dark: bool,
    bg: Color32,
    panel: Color32,
    card: Color32,
    card_hover: Color32,
    track: Color32,
    track_hover: Color32,
    stroke: Color32,
    stroke_strong: Color32,
    text: Color32,
    text_dim: Color32,
    text_faint: Color32,
    success: Color32,
    warning: Color32,
    danger: Color32,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

const DARK_BASE: Base = Base {
    is_dark: true,
    bg: rgb(0x10, 0x10, 0x15),
    panel: rgb(0x17, 0x17, 0x1E),
    card: rgb(0x1E, 0x1E, 0x27),
    card_hover: rgb(0x26, 0x26, 0x33),
    track: rgb(0x35, 0x35, 0x45),
    track_hover: rgb(0x42, 0x42, 0x55),
    stroke: rgb(0x2E, 0x2E, 0x3B),
    stroke_strong: rgb(0x3D, 0x3D, 0x4E),
    text: rgb(0xE8, 0xE8, 0xF0),
    text_dim: rgb(0x9C, 0x9C, 0xB0),
    text_faint: rgb(0x6E, 0x6E, 0x82),
    success: rgb(0x4A, 0xDE, 0x80),
    warning: rgb(0xFB, 0xBF, 0x24),
    danger: rgb(0xF8, 0x71, 0x71),
};

/// The light neutrals.
///
/// Not an inversion of the dark ones. Two places refuse to invert: raised
/// surfaces run *towards* white as they rise, so `card` tops out at white and
/// `card_hover` has to go the other way to register as a highlight at all; and
/// the status colours have to be darkened rather than reused, because the
/// saturated greens and ambers that read clearly on near-black are close to
/// invisible on near-white.
const LIGHT_BASE: Base = Base {
    is_dark: false,
    bg: rgb(0xEC, 0xEC, 0xF2),
    panel: rgb(0xF4, 0xF4, 0xF8),
    card: rgb(0xFF, 0xFF, 0xFF),
    card_hover: rgb(0xF2, 0xF2, 0xFA),
    track: rgb(0xD7, 0xD7, 0xE4),
    track_hover: rgb(0xC5, 0xC5, 0xD6),
    stroke: rgb(0xDD, 0xDD, 0xE8),
    stroke_strong: rgb(0xB8, 0xB8, 0xC8),
    text: rgb(0x1A, 0x1A, 0x24),
    text_dim: rgb(0x55, 0x55, 0x66),
    text_faint: rgb(0x77, 0x77, 0x8A),
    success: rgb(0x14, 0x7F, 0x3C),
    warning: rgb(0x92, 0x50, 0x0B),
    danger: rgb(0xC6, 0x26, 0x26),
};

/// The four colours an accent contributes.
struct AccentColors {
    accent: Color32,
    hover: Color32,
    press: Color32,
    /// Text on top of `accent`. Per accent and per mode, because a bright amber
    /// needs dark text over it and a deep violet needs light text, and getting
    /// that wrong is the one theme mistake that makes a button unreadable.
    on_accent: Color32,
}

/// An accent, with the variant of it that each mode uses.
struct AccentSpec {
    id: Accent,
    label: &'static str,
    dark: AccentColors,
    light: AccentColors,
}

/// The brand colour, as a user choice.
///
/// A closed set rather than a colour picker: each entry carries a hand-picked
/// hover, press and on-accent colour per mode, which is what keeps a chosen
/// accent as legible as the default one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Accent {
    #[default]
    Violet,
    Blue,
    Teal,
    Green,
    Amber,
    Rose,
}

/// Near-black, for text on a light accent.
const INK: Color32 = rgb(0x14, 0x14, 0x1C);
/// Near-white, for text on a dark accent.
const PAPER: Color32 = rgb(0xF7, 0xF6, 0xFF);

/// Index zero is the default, and is the app's original look.
const ACCENTS: [AccentSpec; 6] = [
    AccentSpec {
        id: Accent::Violet,
        label: "Violet",
        dark: AccentColors {
            accent: rgb(0x7C, 0x6C, 0xFF),
            // Was 0x9385FF, which put the primary button's own label at 2.78:1
            // against it — below the 3:1 floor, and only on hover, which is why
            // it survived to 0.6.0 unnoticed. This is as light as the hover can
            // go and still carry `PAPER` text.
            hover: rgb(0x88, 0x78, 0xFF),
            press: rgb(0x6A, 0x58, 0xF0),
            on_accent: PAPER,
        },
        light: AccentColors {
            accent: rgb(0x5B, 0x49, 0xDE),
            hover: rgb(0x6D, 0x5B, 0xEE),
            press: rgb(0x4B, 0x3A, 0xC4),
            on_accent: PAPER,
        },
    },
    AccentSpec {
        id: Accent::Blue,
        label: "Blue",
        dark: AccentColors {
            accent: rgb(0x58, 0xA6, 0xFF),
            hover: rgb(0x79, 0xB8, 0xFF),
            press: rgb(0x3D, 0x8E, 0xEB),
            on_accent: INK,
        },
        light: AccentColors {
            accent: rgb(0x10, 0x66, 0xC9),
            hover: rgb(0x1C, 0x77, 0xDC),
            press: rgb(0x0B, 0x55, 0xAA),
            on_accent: PAPER,
        },
    },
    AccentSpec {
        id: Accent::Teal,
        label: "Teal",
        dark: AccentColors {
            accent: rgb(0x2D, 0xD4, 0xBF),
            hover: rgb(0x5E, 0xEA, 0xD4),
            press: rgb(0x14, 0xB8, 0xA6),
            on_accent: INK,
        },
        light: AccentColors {
            accent: rgb(0x0B, 0x7A, 0x70),
            hover: rgb(0x0F, 0x8E, 0x82),
            press: rgb(0x08, 0x62, 0x5A),
            on_accent: PAPER,
        },
    },
    AccentSpec {
        id: Accent::Green,
        label: "Green",
        dark: AccentColors {
            accent: rgb(0x4A, 0xDE, 0x80),
            hover: rgb(0x86, 0xEF, 0xAC),
            press: rgb(0x22, 0xC5, 0x5E),
            on_accent: INK,
        },
        light: AccentColors {
            accent: rgb(0x12, 0x6E, 0x34),
            hover: rgb(0x17, 0x80, 0x3E),
            press: rgb(0x0D, 0x58, 0x29),
            on_accent: PAPER,
        },
    },
    AccentSpec {
        id: Accent::Amber,
        label: "Amber",
        dark: AccentColors {
            accent: rgb(0xFB, 0xBF, 0x24),
            hover: rgb(0xFC, 0xD3, 0x4D),
            press: rgb(0xF5, 0x9E, 0x0B),
            on_accent: INK,
        },
        light: AccentColors {
            accent: rgb(0x92, 0x50, 0x0B),
            hover: rgb(0xA8, 0x5D, 0x0E),
            press: rgb(0x76, 0x40, 0x08),
            on_accent: PAPER,
        },
    },
    AccentSpec {
        id: Accent::Rose,
        label: "Rose",
        dark: AccentColors {
            accent: rgb(0xFB, 0x71, 0x85),
            hover: rgb(0xFD, 0xA4, 0xAF),
            press: rgb(0xF4, 0x3F, 0x5E),
            on_accent: INK,
        },
        light: AccentColors {
            accent: rgb(0xBC, 0x24, 0x4E),
            hover: rgb(0xD1, 0x2C, 0x58),
            press: rgb(0x9D, 0x1B, 0x40),
            on_accent: PAPER,
        },
    },
];

impl Accent {
    pub const ALL: [Accent; 6] = [
        Accent::Violet,
        Accent::Blue,
        Accent::Teal,
        Accent::Green,
        Accent::Amber,
        Accent::Rose,
    ];

    fn index(self) -> usize {
        // Linear search over six entries, against a table whose order is also
        // the palette index: deriving the index from the table is what stops
        // the two from drifting apart.
        let mut i = 0;
        while i < ACCENTS.len() {
            if ACCENTS[i].id as usize == self as usize {
                return i;
            }
            i += 1;
        }
        0
    }

    pub fn label(self) -> &'static str {
        ACCENTS[self.index()].label
    }

    /// The accent colour as the given mode renders it, for the swatch row on
    /// the Settings page — which has to draw the colours of a scheme that is
    /// not the one currently in force.
    pub fn swatch(self, dark: bool) -> Color32 {
        let spec = &ACCENTS[self.index()];
        if dark {
            spec.dark.accent
        } else {
            spec.light.accent
        }
    }
}

const fn compose(base: &Base, a: &AccentColors) -> Palette {
    Palette {
        is_dark: base.is_dark,
        bg: base.bg,
        panel: base.panel,
        card: base.card,
        card_hover: base.card_hover,
        track: base.track,
        track_hover: base.track_hover,
        stroke: base.stroke,
        stroke_strong: base.stroke_strong,
        text: base.text,
        text_dim: base.text_dim,
        text_faint: base.text_faint,
        accent: a.accent,
        accent_hover: a.hover,
        accent_press: a.press,
        on_accent: a.on_accent,
        success: base.success,
        warning: base.warning,
        danger: base.danger,
    }
}

/// The default scheme: the violet dark palette the app shipped with. Also the
/// fallback [`palette`] returns if the active index is ever out of range.
pub const DARK: Palette = compose(&DARK_BASE, &ACCENTS[0].dark);

/// Where a `(accent, dark)` pair lands in [`palettes`].
fn scheme_index(accent: Accent, dark: bool) -> usize {
    accent.index() * 2 + usize::from(!dark)
}

/// Every mode × accent combination, built once.
///
/// Twelve palettes of nineteen colours each is under a kilobyte, so building
/// them all up front is cheaper than any arrangement that lets [`palette`]
/// return something other than a `&'static`.
fn palettes() -> &'static [Palette] {
    static PALETTES: OnceLock<Vec<Palette>> = OnceLock::new();
    PALETTES.get_or_init(|| {
        let mut out = Vec::with_capacity(ACCENTS.len() * 2);
        for spec in &ACCENTS {
            out.push(compose(&DARK_BASE, &spec.dark));
            out.push(compose(&LIGHT_BASE, &spec.light));
        }
        out
    })
}

/// Index into [`palettes`] of the scheme in force. Zero — violet dark — until
/// the app reads the saved preference.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Choose the scheme every later `palette()` call returns.
///
/// Does not touch egui's `Style`; call [`install`] after this to push the new
/// colours into the context.
pub fn set_scheme(accent: Accent, dark: bool) {
    ACTIVE.store(scheme_index(accent, dark), Ordering::Relaxed);
}

/// The palette in force.
pub fn palette() -> &'static Palette {
    let all = palettes();
    // Defensive rather than indexing blind: an out-of-range store would
    // otherwise panic inside every draw call in the program.
    all.get(ACTIVE.load(Ordering::Relaxed)).unwrap_or(&DARK)
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
    pub const NAV_SETTINGS: &str = "◑";
    pub const NAV_EDIT: &str = "✏";

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
    /// Between a width and a height.
    pub const TIMES: &str = "×";

    /// Everything above, for `glyphs_are_all_covered_by_the_bundled_fonts`.
    /// Test-only, so it does not sit in the shipped binary as dead data.
    #[cfg(test)]
    pub const ALL: &[&str] = &[
        NAV_BATCH,
        NAV_TUNING,
        NAV_MODELS,
        NAV_ABOUT,
        NAV_EDIT,
        NAV_SETTINGS,
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
        TIMES,
    ];
}

/// `a` blended `t` of the way towards `b`, in gamma space.
///
/// Gamma rather than linear because this feeds egui's non-linear framebuffer
/// path, which is what the app renders through.
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
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
    v.dark_mode = p.is_dark;
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

    // The pressed state carries two jobs egui does not separate: it is what a
    // held-down widget looks like, and `fg_stroke` here is *also* where egui
    // takes the colour of every `RichText::strong()` in the program from —
    // `override_text_color` does not reach it. Setting it to `on_accent`, as
    // this did, meant every bold label in the app was drawn in the colour meant
    // for text sitting on an accent fill. On the dark scheme those were both
    // near-white and nobody noticed; on a light one it put white headings on a
    // white card.
    //
    // So `fg_stroke` is plain body text, and the fill becomes the accent mixed
    // into the card rather than the accent itself — still unmistakably a press,
    // still something body text can be read on. `pressed_fill_carries_body_text`
    // pins that for every scheme.
    let pressed = mix(p.card, p.accent, if p.is_dark { 0.38 } else { 0.22 });
    v.widgets.active.bg_fill = pressed;
    v.widgets.active.weak_bg_fill = pressed;
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, p.accent);
    v.widgets.active.fg_stroke = Stroke::new(1.0_f32, p.text);
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
    let label = egui::RichText::new(text)
        .color(if enabled { p.on_accent } else { p.text_faint })
        .strong();
    if !enabled {
        return ui.add_enabled(
            false,
            egui::Button::new(label)
                .fill(p.card)
                .rounding(Rounding::same(RADIUS))
                .min_size(egui::vec2(0.0, 30.0)),
        );
    }
    // The accent's hover and press colours exist for exactly this control, and
    // `Button::fill` would not use them: it paints one colour in every state,
    // so the most important button in the window was the only one that gave no
    // feedback when you pressed it. Styling the scope instead lets egui pick
    // per state, as it does for every other button.
    ui.scope(|ui| {
        let w = &mut ui.style_mut().visuals.widgets;
        w.inactive.weak_bg_fill = p.accent;
        w.hovered.weak_bg_fill = p.accent_hover;
        w.active.weak_bg_fill = p.accent_press;
        w.hovered.bg_stroke = Stroke::NONE;
        w.active.bg_stroke = Stroke::NONE;
        ui.add(
            egui::Button::new(label)
                .rounding(Rounding::same(RADIUS))
                .min_size(egui::vec2(0.0, 30.0)),
        )
    })
    .inner
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

/// Serialisation for tests that touch process-global UI state.
///
/// Three things in this program are global and are read while the UI draws: the
/// active scheme here, and `OBSCURA_CONFIG_DIR`/`OBSCURA_MODEL_DIR`. Tests run
/// in parallel threads, so one lock covers all of them — one rather than three,
/// because three locks acquired in two different orders is a deadlock waiting
/// to be written.
#[cfg(test)]
pub(crate) mod testing {
    use super::{set_scheme, Accent};
    use std::sync::{Mutex, MutexGuard};

    static UI_LOCK: Mutex<()> = Mutex::new(());

    /// Held for the duration of a test; restores the default scheme on the way
    /// out, panic or not.
    pub(crate) struct UiGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl Drop for UiGuard {
        fn drop(&mut self) {
            set_scheme(Accent::default(), true);
            std::env::remove_var("OBSCURA_CONFIG_DIR");
            std::env::remove_var("OBSCURA_MODEL_DIR");
        }
    }

    /// Take the lock, leaving the scheme at its default.
    pub(crate) fn lock() -> UiGuard {
        // A poisoned lock means another test panicked while holding it, which
        // is already being reported; failing every later test on top of that
        // only buries the real one.
        UiGuard(UI_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Take the lock and select a scheme for the duration.
    pub(crate) fn with_scheme(accent: Accent, dark: bool) -> UiGuard {
        let guard = lock();
        set_scheme(accent, dark);
        guard
    }
}

#[cfg(test)]
mod tests {
    use super::testing::with_scheme;
    use super::*;

    /// WCAG 2.1 relative luminance.
    fn luminance(c: Color32) -> f64 {
        let channel = |v: u8| {
            let s = v as f64 / 255.0;
            if s <= 0.040_45 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(c.r()) + 0.7152 * channel(c.g()) + 0.0722 * channel(c.b())
    }

    /// WCAG 2.1 contrast ratio, 1.0 (identical) to 21.0 (black on white).
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (x, y) = (luminance(a), luminance(b));
        let (hi, lo) = if x > y { (x, y) } else { (y, x) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// Every palette the Settings page can select, with a name for the failure
    /// message.
    fn all_schemes() -> Vec<(String, &'static Palette)> {
        let mut out = Vec::new();
        for accent in Accent::ALL {
            for dark in [true, false] {
                let name = format!("{} {}", accent.label(), if dark { "dark" } else { "light" });
                out.push((name, &palettes()[scheme_index(accent, dark)]));
            }
        }
        out
    }

    #[test]
    fn contrast_helper_matches_the_known_extremes() {
        // Guards the assertions below from a silently broken formula.
        assert!((contrast(Color32::BLACK, Color32::WHITE) - 21.0).abs() < 0.05);
        assert!((contrast(Color32::WHITE, Color32::WHITE) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn body_text_is_readable_on_every_surface_of_every_scheme() {
        // 4.5:1 is WCAG AA for body text. The app's body size is 14px, which is
        // not "large text" under any reading of the spec, so this is the bar
        // that applies — and the reason a light scheme needed its own neutrals
        // rather than the dark ones inverted.
        for (name, p) in all_schemes() {
            for (surface, bg) in [("bg", p.bg), ("panel", p.panel), ("card", p.card)] {
                let ratio = contrast(p.text, bg);
                assert!(
                    ratio >= 4.5,
                    "{name}: text on {surface} is {ratio:.2}:1, needs 4.5"
                );
                let dim = contrast(p.text_dim, bg);
                assert!(
                    dim >= 4.5,
                    "{name}: text_dim on {surface} is {dim:.2}:1, needs 4.5"
                );
            }
        }
    }

    #[test]
    fn tertiary_text_and_status_colours_stay_above_the_large_text_bar() {
        // `text_faint` and the status colours are deliberately quiet, and all
        // of them are drawn at 11-12px in short runs (placeholders, section
        // labels, pills). 3:1 is the floor at which they are still legible
        // rather than decorative.
        for (name, p) in all_schemes() {
            for (role, c) in [
                ("text_faint", p.text_faint),
                ("success", p.success),
                ("warning", p.warning),
                ("danger", p.danger),
            ] {
                let ratio = contrast(c, p.card);
                assert!(
                    ratio >= 3.0,
                    "{name}: {role} on a card is {ratio:.2}:1, needs 3.0"
                );
            }
        }
    }

    #[test]
    fn button_text_is_readable_on_its_own_accent() {
        // The one theme mistake that makes a control unusable rather than ugly:
        // white text on a bright amber, or near-black on a deep violet. Each
        // accent picks its own `on_accent` per mode precisely to pass this.
        for (name, p) in all_schemes() {
            for (state, fill) in [
                ("accent", p.accent),
                ("accent_hover", p.accent_hover),
                ("accent_press", p.accent_press),
            ] {
                let ratio = contrast(p.on_accent, fill);
                assert!(
                    ratio >= 3.0,
                    "{name}: on_accent over {state} is {ratio:.2}:1, needs 3.0"
                );
            }
        }
    }

    #[test]
    fn the_accent_stands_out_against_the_surfaces_it_is_painted_on() {
        // Progress bars, selected nav entries and the Run button are accent
        // fills on a card or the panel. 3:1 is WCAG's bar for a non-text UI
        // component, and an accent that fails it is a primary action nobody
        // can find.
        for (name, p) in all_schemes() {
            for (surface, bg) in [("card", p.card), ("panel", p.panel)] {
                let ratio = contrast(p.accent, bg);
                assert!(
                    ratio >= 3.0,
                    "{name}: accent on {surface} is {ratio:.2}:1, needs 3.0"
                );
            }
        }
    }

    #[test]
    fn control_tracks_are_distinct_from_the_cards_they_sit_on_in_every_scheme() {
        // Slider rails, checkbox boxes and progress troughs are painted with
        // `track`. When it equals the card colour every rail vanishes and the
        // handle appears to float in empty space — which is what happened once
        // already, in the only scheme that existed then.
        for (name, p) in all_schemes() {
            assert_ne!(p.track, p.card, "{name}: slider rails would be invisible");
            assert_ne!(p.track_hover, p.card_hover, "{name}: hovered rails vanish");
            assert_ne!(p.stroke, p.card, "{name}: card borders would be invisible");
        }
    }

    #[test]
    fn set_scheme_changes_what_palette_returns() {
        let _g = with_scheme(Accent::Teal, false);
        let p = palette();
        assert!(!p.is_dark);
        assert_eq!(p.card, LIGHT_BASE.card);
        assert_eq!(p.accent, Accent::Teal.swatch(false));

        set_scheme(Accent::Violet, true);
        assert_eq!(palette().card, DARK_BASE.card);
    }

    #[test]
    fn installing_a_light_scheme_tells_egui_it_is_light() {
        // egui paints a handful of things itself from `dark_mode` — the text
        // cursor and the resize handles among them — so leaving it hard-coded
        // would put dark-theme chrome on a light window.
        let _g = with_scheme(Accent::Violet, false);
        let ctx = egui::Context::default();
        install(&ctx);
        let style = ctx.style();
        assert!(!style.visuals.dark_mode);
        assert_eq!(style.visuals.panel_fill, LIGHT_BASE.panel);
        assert_eq!(style.visuals.override_text_color, Some(LIGHT_BASE.text));
    }

    #[test]
    fn pressed_fill_carries_body_text() {
        // egui draws every `strong()` label in `widgets.active.fg_stroke`, so
        // that colour has to be ordinary body text — which in turn means the
        // pressed fill underneath it has to be something body text reads on.
        for (name, p) in all_schemes() {
            let ctx = egui::Context::default();
            let _g = with_scheme(accent_of(p), p.is_dark);
            install(&ctx);
            let w = &ctx.style().visuals.widgets;
            assert_eq!(
                w.active.fg_stroke.color, p.text,
                "{name}: strong text would not be body text"
            );
            let ratio = contrast(p.text, w.active.bg_fill);
            assert!(
                ratio >= 4.5,
                "{name}: text on a pressed widget is {ratio:.2}:1, needs 4.5"
            );
            // Still visibly a press rather than a resting widget.
            assert_ne!(w.active.bg_fill, p.card, "{name}: press is invisible");
        }
    }

    /// Which accent a palette was built from, by matching its accent colour.
    fn accent_of(p: &Palette) -> Accent {
        Accent::ALL
            .into_iter()
            .find(|a| a.swatch(p.is_dark) == p.accent)
            .expect("every palette comes from the accent table")
    }

    #[test]
    fn every_accent_is_in_the_table_exactly_once() {
        // `Accent::index` is what maps a saved preference onto a palette; an
        // accent missing from the table would silently resolve to violet.
        for accent in Accent::ALL {
            assert_eq!(
                ACCENTS[accent.index()].id,
                accent,
                "{accent:?} resolves to the wrong table entry"
            );
            assert!(!accent.label().is_empty());
        }
        assert_eq!(ACCENTS.len(), Accent::ALL.len());
        // Index zero is the app's original look, and `DARK` is built from it.
        assert_eq!(DARK.accent, Accent::Violet.swatch(true));
    }

    #[test]
    fn every_scheme_has_its_own_slot() {
        let mut seen = std::collections::HashSet::new();
        for accent in Accent::ALL {
            for dark in [true, false] {
                let i = scheme_index(accent, dark);
                assert!(i < palettes().len(), "{accent:?} {dark} is out of range");
                assert!(seen.insert(i), "{accent:?} {dark} collides with another");
            }
        }
        assert_eq!(seen.len(), palettes().len());
    }

    #[test]
    fn install_sets_the_palette_and_type_scale() {
        let _g = with_scheme(Accent::default(), true);
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
    fn every_widget_state_shares_one_corner_radius() {
        // Mixed radii are the most visible way a hand-built theme looks
        // unfinished, so this is pinned rather than left to inspection.
        let _g = with_scheme(Accent::default(), true);
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
