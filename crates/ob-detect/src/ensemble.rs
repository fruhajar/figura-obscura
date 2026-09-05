//! Running several detectors over the same frame and merging their verdicts.
//!
//! The anime detectors in the registry share a taxonomy but not their weights,
//! so where they disagree is exactly where a single model is unsure. An
//! [`EnsembleDetector`] runs each of them and combines the results under one of
//! two opposite policies:
//!
//! * `min_votes == 1` (the default) — **union**. Anything any model sees gets
//!   censored. Maximises recall, which is the fail-safe direction for a
//!   censoring tool: a false positive costs a needlessly obscured patch, a false
//!   negative costs the thing the tool exists to prevent.
//! * `min_votes >= 2` — **consensus**. A region survives only if that many
//!   models independently found it. Maximises precision at the cost of recall;
//!   useful for auditing how much of a model's output is corroborated, and
//!   dangerous as a production censoring policy.
//!
//! **Votes are counted per category, among the members that could actually
//! cast one.** Members do not share a taxonomy: the three-class anime models
//! can never report buttocks, and a nipple specialist can never report anything
//! else. Counting a global threshold against members structurally unable to
//! vote would mean that adding a specialist *deletes* every category outside
//! its competence — a specialist would make coverage strictly worse, which is
//! the opposite of why anyone adds one.
//!
//! Like [`crate::tile::TiledDetector`] this is itself a [`Detector`], so it
//! composes: an ensemble of tiled detectors is just the two wrappers nested.

use crate::postprocess::nms;
use crate::{DetectError, Detector};
use ob_core::geometry::{Detection, Frame};
use ob_core::taxonomy::Category;

/// Several detectors voting on one frame.
pub struct EnsembleDetector {
    members: Vec<Box<dyn Detector>>,
    /// How many members must independently find a region for it to be kept.
    min_votes: usize,
    /// IoU above which two members' boxes count as the same region.
    agree_iou: f32,
}

impl EnsembleDetector {
    /// Build an ensemble. `min_votes` is clamped to at least 1, and to at most
    /// the member count (so a too-high value cannot silently censor nothing).
    pub fn new(members: Vec<Box<dyn Detector>>, min_votes: usize, agree_iou: f32) -> Self {
        let n = members.len().max(1);
        Self {
            members,
            min_votes: min_votes.clamp(1, n),
            agree_iou,
        }
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// How many members could possibly report `category`.
    fn eligible_voters(&self, category: Category) -> usize {
        self.members.iter().filter(|m| m.can_emit(category)).count()
    }

    /// Votes actually required for `category`, never more than the number of
    /// members able to give one.
    ///
    /// So `--min-votes 2` across a general model and a nipple specialist still
    /// means "two must agree" for nipples, but stays "one is enough" for
    /// genitalia — which only one of them can see. Without this clamp the
    /// genitalia boxes would be silently discarded.
    fn required_votes(&self, category: Category) -> usize {
        self.min_votes.min(self.eligible_voters(category).max(1))
    }
}

impl Detector for EnsembleDetector {
    /// The ensemble can report anything any member can.
    fn can_emit(&self, category: Category) -> bool {
        self.members.iter().any(|m| m.can_emit(category))
    }

    fn detect(&self, frame: &Frame) -> Result<Vec<Detection>, DetectError> {
        // Keep each member's output separate: votes are counted per member, so
        // one model firing twice on the same spot must not look like agreement.
        let mut per_member: Vec<Vec<Detection>> = Vec::with_capacity(self.members.len());
        for m in &self.members {
            per_member.push(m.detect(frame)?);
        }

        if self.min_votes <= 1 {
            // Union: merge everything, then de-duplicate overlapping boxes.
            let all: Vec<Detection> = per_member.into_iter().flatten().collect();
            return Ok(nms(all, self.agree_iou));
        }

        // Consensus: keep a detection only when enough *other* members also
        // found a same-category box overlapping it.
        let mut kept = Vec::new();
        for (i, dets) in per_member.iter().enumerate() {
            for d in dets {
                let votes = 1 + per_member
                    .iter()
                    .enumerate()
                    .filter(|(j, other)| {
                        *j != i
                            && other.iter().any(|o| {
                                o.category == d.category && o.bbox.iou(&d.bbox) >= self.agree_iou
                            })
                    })
                    .count();
                if votes >= self.required_votes(d.category) {
                    kept.push(*d);
                }
            }
        }
        Ok(nms(kept, self.agree_iou))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ob_core::geometry::BBox;
    use ob_core::taxonomy::cat;

    struct Fixed(Vec<Detection>);

    impl Detector for Fixed {
        fn detect(&self, _frame: &Frame) -> Result<Vec<Detection>, DetectError> {
            Ok(self.0.clone())
        }
    }

    /// A detector that can only ever report one category — the shape of a
    /// specialist, e.g. a nipple-only model.
    struct Specialist {
        only: Category,
        dets: Vec<Detection>,
    }

    impl Detector for Specialist {
        fn detect(&self, _frame: &Frame) -> Result<Vec<Detection>, DetectError> {
            Ok(self.dets.clone())
        }
        fn can_emit(&self, category: Category) -> bool {
            category == self.only
        }
    }

    fn det(x1: f32, score: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, 0.0, x1 + 20.0, 20.0),
            category: cat::FEMALE_BREAST_EXPOSED,
            score,
        }
    }

    fn frame() -> Frame {
        Frame::new(64, 64, vec![0u8; 64 * 64 * 3]).unwrap()
    }

    fn member(dets: Vec<Detection>) -> Box<dyn Detector> {
        Box::new(Fixed(dets))
    }

    #[test]
    fn union_keeps_what_only_one_model_saw() {
        // Model A finds a region at 0; model B finds a different one at 200.
        let e = EnsembleDetector::new(
            vec![member(vec![det(0.0, 0.9)]), member(vec![det(200.0, 0.8)])],
            1,
            0.45,
        );
        let out = e.detect(&frame()).unwrap();
        assert_eq!(out.len(), 2, "union must keep both singly-seen regions");
    }

    #[test]
    fn union_merges_the_same_region_seen_twice() {
        let e = EnsembleDetector::new(
            vec![member(vec![det(0.0, 0.9)]), member(vec![det(2.0, 0.7)])],
            1,
            0.45,
        );
        let out = e.detect(&frame()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].score, 0.9, "the more confident box should survive");
    }

    #[test]
    fn consensus_drops_what_only_one_model_saw() {
        let e = EnsembleDetector::new(
            vec![member(vec![det(0.0, 0.9)]), member(vec![det(200.0, 0.8)])],
            2,
            0.45,
        );
        assert!(
            e.detect(&frame()).unwrap().is_empty(),
            "neither region was corroborated, so consensus keeps nothing"
        );
    }

    #[test]
    fn consensus_keeps_a_corroborated_region() {
        let e = EnsembleDetector::new(
            vec![
                member(vec![det(0.0, 0.9), det(200.0, 0.5)]),
                member(vec![det(3.0, 0.8)]),
            ],
            2,
            0.45,
        );
        let out = e.detect(&frame()).unwrap();
        assert_eq!(out.len(), 1);
        assert!(
            out[0].bbox.x1 < 10.0,
            "the agreed-on region is the one at 0"
        );
    }

    #[test]
    fn one_model_firing_twice_is_not_agreement() {
        // A single member reporting two overlapping boxes must not self-certify.
        let e = EnsembleDetector::new(
            vec![member(vec![det(0.0, 0.9), det(1.0, 0.85)]), member(vec![])],
            2,
            0.45,
        );
        assert!(e.detect(&frame()).unwrap().is_empty());
    }

    #[test]
    fn min_votes_cannot_exceed_the_member_count() {
        // Asking for 5 votes from 2 members would otherwise censor nothing.
        let e = EnsembleDetector::new(
            vec![member(vec![det(0.0, 0.9)]), member(vec![det(1.0, 0.8)])],
            5,
            0.45,
        );
        assert_eq!(e.min_votes, 2);
        assert_eq!(e.detect(&frame()).unwrap().len(), 1);
    }

    /// A detection in a category the specialist cannot see.
    fn other_cat(x1: f32, score: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, 0.0, x1 + 20.0, 20.0),
            category: cat::FEMALE_GENITALIA_EXPOSED,
            score,
        }
    }

    #[test]
    fn a_specialist_cannot_veto_categories_it_cannot_see() {
        // The regression this whole mechanism exists for. A generalist finds a
        // breast and a genitalia region; a nipple specialist corroborates only
        // the breast. Under a naive global --min-votes 2 the genitalia box gets
        // one vote and is deleted — so adding a specialist would make coverage
        // strictly *worse* outside its competence.
        let generalist = Box::new(Fixed(vec![det(10.0, 0.9), other_cat(40.0, 0.9)]));
        let specialist = Box::new(Specialist {
            only: cat::FEMALE_BREAST_EXPOSED,
            dets: vec![det(10.0, 0.8)],
        });

        let ens = EnsembleDetector::new(vec![generalist, specialist], 2, 0.45);
        let out = ens.detect(&frame()).unwrap();

        // Two eligible voters for breasts, and both agreed: kept.
        assert!(
            out.iter().any(|d| d.category == cat::FEMALE_BREAST_EXPOSED),
            "corroborated breast detection was dropped"
        );
        // Only one member can see genitalia at all, so one vote is the most
        // that category can ever get — it must still be kept.
        assert!(
            out.iter().any(|d| d.category == cat::FEMALE_GENITALIA_EXPOSED),
            "a specialist deleted a category it is structurally unable to vote on"
        );
    }

    #[test]
    fn consensus_still_applies_where_members_are_genuinely_eligible() {
        // The clamp must not become a blanket exemption: where two members can
        // both see a category and only one found it, consensus still rejects.
        let a = Box::new(Fixed(vec![det(10.0, 0.9)]));
        let b = Box::new(Fixed(vec![])); // same taxonomy, saw nothing
        let ens = EnsembleDetector::new(vec![a, b], 2, 0.45);
        assert!(
            ens.detect(&frame()).unwrap().is_empty(),
            "an uncorroborated detection survived consensus"
        );
    }

    #[test]
    fn required_votes_is_clamped_per_category() {
        let generalist = Box::new(Fixed(vec![]));
        let specialist = Box::new(Specialist {
            only: cat::FEMALE_BREAST_EXPOSED,
            dets: vec![],
        });
        let ens = EnsembleDetector::new(vec![generalist, specialist], 2, 0.45);

        assert_eq!(ens.eligible_voters(cat::FEMALE_BREAST_EXPOSED), 2);
        assert_eq!(ens.required_votes(cat::FEMALE_BREAST_EXPOSED), 2);
        // Only the generalist can see this one.
        assert_eq!(ens.eligible_voters(cat::FEMALE_GENITALIA_EXPOSED), 1);
        assert_eq!(ens.required_votes(cat::FEMALE_GENITALIA_EXPOSED), 1);
        // The ensemble as a whole reports the union of its members.
        assert!(ens.can_emit(cat::FEMALE_GENITALIA_EXPOSED));
    }
}
