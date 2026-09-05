//! The single canonical vocabulary for Obscura.
//!
//! Every detector, regardless of the labels its ONNX model was trained with,
//! maps its native classes onto [`Category`]. Filters, profiles and CLI flags
//! speak *only* this vocabulary (requirement R3). The taxonomy is derived from
//! NudeNet v3's 18 classes, which map almost 1:1 onto sex × part × state.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Perceived sex associated with a detected region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sex {
    Male,
    Female,
    /// The model does not distinguish sex for this part (e.g. anus, feet).
    Unknown,
}

/// The body part a region covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Breasts,
    Buttocks,
    Genitalia,
    Anus,
    Feet,
    Belly,
    Armpits,
    Face,
    /// Derived in v1 from the top strip of a `Face` region — no detector class.
    Eyes,
}

/// Whether the part is bare or clothed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Exposed,
    Covered,
}

/// A fully-qualified region category: sex × part × state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Category {
    pub sex: Sex,
    pub part: Part,
    pub state: State,
}

impl Category {
    pub const fn new(sex: Sex, part: Part, state: State) -> Self {
        Self { sex, part, state }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Renders like NudeNet's own label style, e.g. "FEMALE_BREAST_EXPOSED".
        let sex = match self.sex {
            Sex::Male => "MALE_",
            Sex::Female => "FEMALE_",
            Sex::Unknown => "",
        };
        let part = match self.part {
            Part::Breasts => "BREAST",
            Part::Buttocks => "BUTTOCKS",
            Part::Genitalia => "GENITALIA",
            Part::Anus => "ANUS",
            Part::Feet => "FEET",
            Part::Belly => "BELLY",
            Part::Armpits => "ARMPITS",
            Part::Face => "FACE",
            Part::Eyes => "EYES",
        };
        let state = match self.state {
            State::Exposed => "_EXPOSED",
            State::Covered => "_COVERED",
        };
        write!(f, "{sex}{part}{state}")
    }
}

/// Convenience constructors for the canonical NudeNet-derived categories.
///
/// These constants double as the mapping targets for every model adapter, and
/// as the checklist the GUI filter tree is generated from.
pub mod cat {
    use super::{Category, Part, Sex, State};
    use Part::*;
    use Sex::*;
    use State::*;

    macro_rules! c {
        ($name:ident, $sex:expr, $part:expr, $state:expr) => {
            pub const $name: Category = Category::new($sex, $part, $state);
        };
    }

    c!(FEMALE_GENITALIA_COVERED, Female, Genitalia, Covered);
    c!(FEMALE_GENITALIA_EXPOSED, Female, Genitalia, Exposed);
    c!(FEMALE_BREAST_COVERED, Female, Breasts, Covered);
    c!(FEMALE_BREAST_EXPOSED, Female, Breasts, Exposed);
    c!(MALE_GENITALIA_EXPOSED, Male, Genitalia, Exposed);
    c!(MALE_BREAST_EXPOSED, Male, Breasts, Exposed);
    c!(BUTTOCKS_COVERED, Unknown, Buttocks, Covered);
    c!(BUTTOCKS_EXPOSED, Unknown, Buttocks, Exposed);
    c!(ANUS_COVERED, Unknown, Anus, Covered);
    c!(ANUS_EXPOSED, Unknown, Anus, Exposed);
    c!(FEET_COVERED, Unknown, Feet, Covered);
    c!(FEET_EXPOSED, Unknown, Feet, Exposed);
    c!(BELLY_COVERED, Unknown, Belly, Covered);
    c!(BELLY_EXPOSED, Unknown, Belly, Exposed);
    c!(ARMPITS_COVERED, Unknown, Armpits, Covered);
    c!(ARMPITS_EXPOSED, Unknown, Armpits, Exposed);
    c!(FACE_FEMALE, Female, Face, Exposed);
    c!(FACE_MALE, Male, Face, Exposed);
    // Derived, not emitted by NudeNet directly:
    c!(EYES_FEMALE, Female, Eyes, Exposed);
    c!(EYES_MALE, Male, Eyes, Exposed);
}

/// The 18 canonical categories that correspond exactly to NudeNet v3 classes,
/// in NudeNet's class-index order. Adapters map onto these; the derived `Eyes`
/// categories are intentionally excluded here.
pub const NUDENET_CATEGORIES: [Category; 18] = [
    cat::FEMALE_GENITALIA_COVERED,
    cat::FACE_FEMALE,
    cat::BUTTOCKS_EXPOSED,
    cat::FEMALE_BREAST_EXPOSED,
    cat::FEMALE_GENITALIA_EXPOSED,
    cat::MALE_BREAST_EXPOSED,
    cat::ANUS_EXPOSED,
    cat::FEET_EXPOSED,
    cat::BELLY_COVERED,
    cat::FEET_COVERED,
    cat::ARMPITS_COVERED,
    cat::ARMPITS_EXPOSED,
    cat::FACE_MALE,
    cat::BELLY_EXPOSED,
    cat::MALE_GENITALIA_EXPOSED,
    cat::ANUS_COVERED,
    cat::FEMALE_BREAST_COVERED,
    cat::BUTTOCKS_COVERED,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nudenet_order_and_labels_match_spec() {
        // Index 3 in NudeNet is FEMALE_BREAST_EXPOSED.
        assert_eq!(NUDENET_CATEGORIES[3].to_string(), "FEMALE_BREAST_EXPOSED");
        assert_eq!(NUDENET_CATEGORIES.len(), 18);
    }

    #[test]
    fn unknown_sex_omits_prefix() {
        assert_eq!(cat::ANUS_EXPOSED.to_string(), "ANUS_EXPOSED");
    }
}
