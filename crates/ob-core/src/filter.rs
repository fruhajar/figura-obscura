//! Selective-censoring filter rules (requirement R3).
//!
//! A [`FilterSet`] decides which detections are censored. Matching is expressed
//! purely in the canonical taxonomy, so it is model-independent.

use crate::geometry::Detection;
use crate::taxonomy::{Category, Part, Sex, State};
use serde::{Deserialize, Serialize};

/// A wildcard-capable predicate over one category axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Match<T> {
    /// Matches any value on this axis.
    Any,
    /// Matches exactly this value.
    Only(T),
}

impl<T: PartialEq> Match<T> {
    fn matches(&self, value: &T) -> bool {
        match self {
            Match::Any => true,
            Match::Only(v) => v == value,
        }
    }
}

/// One rule: a category pattern plus an optional per-rule score floor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterRule {
    pub sex: Match<Sex>,
    pub part: Match<Part>,
    pub state: Match<State>,
    /// Override the set-wide minimum score for detections matching this rule.
    #[serde(default)]
    pub min_score: Option<f32>,
}

impl FilterRule {
    /// A rule that selects a single fully-qualified category.
    pub fn exact(cat: Category) -> Self {
        Self {
            sex: Match::Only(cat.sex),
            part: Match::Only(cat.part),
            state: Match::Only(cat.state),
            min_score: None,
        }
    }

    /// Select an entire part regardless of sex/state (e.g. all genitalia).
    pub fn part(part: Part) -> Self {
        Self {
            sex: Match::Any,
            part: Match::Only(part),
            state: Match::Any,
            min_score: None,
        }
    }

    fn matches_category(&self, cat: &Category) -> bool {
        self.sex.matches(&cat.sex) && self.part.matches(&cat.part) && self.state.matches(&cat.state)
    }
}

/// The active censoring policy: any matching rule selects a detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterSet {
    pub rules: Vec<FilterRule>,
    /// Set-wide minimum score applied when a rule has no `min_score`.
    pub min_score: f32,
}

impl Default for FilterSet {
    fn default() -> Self {
        // Sensible safe default: censor exposed genitalia, breasts, buttocks, anus.
        Self {
            rules: vec![
                FilterRule::part(Part::Genitalia),
                FilterRule::part(Part::Anus),
                FilterRule {
                    sex: Match::Any,
                    part: Match::Only(Part::Breasts),
                    state: Match::Only(State::Exposed),
                    min_score: None,
                },
                FilterRule {
                    sex: Match::Any,
                    part: Match::Only(Part::Buttocks),
                    state: Match::Only(State::Exposed),
                    min_score: None,
                },
            ],
            min_score: 0.2,
        }
    }
}

impl FilterSet {
    /// True if this detection should be censored under the current policy.
    pub fn selects(&self, det: &Detection) -> bool {
        for rule in &self.rules {
            if rule.matches_category(&det.category) {
                let floor = rule.min_score.unwrap_or(self.min_score);
                if det.score >= floor {
                    return true;
                }
            }
        }
        false
    }

    /// Filter a detection slice down to the ones this policy selects.
    pub fn select_all<'a>(&self, dets: &'a [Detection]) -> Vec<&'a Detection> {
        dets.iter().filter(|d| self.selects(d)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BBox;
    use crate::taxonomy::cat;

    fn det(cat: Category, score: f32) -> Detection {
        Detection {
            bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
            category: cat,
            score,
        }
    }

    #[test]
    fn default_selects_exposed_breast() {
        let f = FilterSet::default();
        assert!(f.selects(&det(cat::FEMALE_BREAST_EXPOSED, 0.9)));
    }

    #[test]
    fn default_ignores_covered_breast() {
        let f = FilterSet::default();
        assert!(!f.selects(&det(cat::FEMALE_BREAST_COVERED, 0.9)));
    }

    #[test]
    fn score_floor_is_respected() {
        let f = FilterSet::default();
        assert!(!f.selects(&det(cat::FEMALE_GENITALIA_EXPOSED, 0.1)));
        assert!(f.selects(&det(cat::FEMALE_GENITALIA_EXPOSED, 0.5)));
    }

    #[test]
    fn per_rule_min_score_overrides() {
        let f = FilterSet {
            rules: vec![FilterRule {
                min_score: Some(0.8),
                ..FilterRule::part(Part::Feet)
            }],
            min_score: 0.2,
        };
        assert!(!f.selects(&det(cat::FEET_EXPOSED, 0.5)));
        assert!(f.selects(&det(cat::FEET_EXPOSED, 0.85)));
    }
}
