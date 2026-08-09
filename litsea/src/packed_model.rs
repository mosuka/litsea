//! Declarative feature-template table and packed integer feature keys.
//!
//! This module is the single source of truth for the segmentation feature
//! template (issue #136). Three consumers derive from the [`TEMPLATES`]
//! table:
//!
//! 1. The string writer ([`crate::segmenter::Segmenter`]'s `write_attributes`),
//!    which renders feature strings for training, extraction, and the POS
//!    path.
//! 2. The packed-key writer ([`Template::pack`]), used by `segment()`'s hot
//!    loop to score positions without building strings.
//! 3. The load-time parser ([`parse_feature_keys`]), which converts a trained
//!    model's string feature keys into packed keys once, when the model is
//!    compiled into a [`PackedModel`].
//!
//! The table order is load-bearing: `segment()` sums `f64` weights in
//! emission order and float addition is not associative, so the order must
//! stay byte-for-byte compatible with the historical `write_attributes`
//! emission sequence. The language-gated `WC1`..`WC4` templates sit last so
//! that [`templates_for`] can hand out a prefix slice.

use crate::language::Language;

/// One slot of a feature template. The payload is the position delta: a slot
/// with delta `d` reads context index `i - 3 + d` for character position `i`.
///
/// - `Tag`: a boundary tag ("U"/"B"/"O"); valid deltas are 0..=2 (positions
///   `i-3`..`i-1` — tags exist only for already-decided positions).
/// - `Chr`: a character (a real character or one of the [`SENTINELS`]);
///   valid deltas are 0..=5 (positions `i-3`..`i+2`).
/// - `Typ`: a character type code; valid deltas are 0..=5.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Slot {
    /// Boundary tag slot (delta 0..=2).
    Tag(u8),
    /// Character slot (delta 0..=5).
    Chr(u8),
    /// Character type slot (delta 0..=5).
    Typ(u8),
}

/// One feature template: a distinct prefix plus an ordered slot list.
/// A feature string renders as `{prefix}:{slot values concatenated}`.
#[derive(Debug)]
pub(crate) struct Template {
    /// Template id; equals the index in [`TEMPLATES`] (pinned by a test).
    /// Occupies the top byte of every packed key from this template.
    #[allow(dead_code)] // read by Template::pack, introduced with the packed scoring path
    pub(crate) id: u8,
    /// Feature-string prefix, e.g. `"UW4"`. All prefixes are distinct.
    pub(crate) prefix: &'static str,
    /// Ordered slots concatenated after the `:` separator.
    pub(crate) slots: &'static [Slot],
}

const fn t(id: u8, prefix: &'static str, slots: &'static [Slot]) -> Template {
    Template { id, prefix, slots }
}

use Slot::{Chr, Tag, Typ};

/// The full feature template in historical emission order. Slot deltas map
/// the classic template variables to context positions: `w1..w6` = `Chr(0..5)`,
/// `c1..c6` = `Typ(0..5)`, `p1..p3` = `Tag(0..2)`.
pub(crate) const TEMPLATES: [Template; 42] = [
    t(0, "UP1", &[Tag(0)]),
    t(1, "UP2", &[Tag(1)]),
    t(2, "UP3", &[Tag(2)]),
    t(3, "BP1", &[Tag(0), Tag(1)]),
    t(4, "BP2", &[Tag(1), Tag(2)]),
    t(5, "UW1", &[Chr(0)]),
    t(6, "UW2", &[Chr(1)]),
    t(7, "UW3", &[Chr(2)]),
    t(8, "UW4", &[Chr(3)]),
    t(9, "UW5", &[Chr(4)]),
    t(10, "UW6", &[Chr(5)]),
    t(11, "BW1", &[Chr(1), Chr(2)]),
    t(12, "BW2", &[Chr(2), Chr(3)]),
    t(13, "BW3", &[Chr(3), Chr(4)]),
    t(14, "UC1", &[Typ(0)]),
    t(15, "UC2", &[Typ(1)]),
    t(16, "UC3", &[Typ(2)]),
    t(17, "UC4", &[Typ(3)]),
    t(18, "UC5", &[Typ(4)]),
    t(19, "UC6", &[Typ(5)]),
    t(20, "BC1", &[Typ(1), Typ(2)]),
    t(21, "BC2", &[Typ(2), Typ(3)]),
    t(22, "BC3", &[Typ(3), Typ(4)]),
    t(23, "TC1", &[Typ(0), Typ(1), Typ(2)]),
    t(24, "TC2", &[Typ(1), Typ(2), Typ(3)]),
    t(25, "TC3", &[Typ(2), Typ(3), Typ(4)]),
    t(26, "TC4", &[Typ(3), Typ(4), Typ(5)]),
    t(27, "UQ1", &[Tag(0), Typ(0)]),
    t(28, "UQ2", &[Tag(1), Typ(1)]),
    t(29, "UQ3", &[Tag(2), Typ(2)]),
    t(30, "BQ1", &[Tag(1), Typ(1), Typ(2)]),
    t(31, "BQ2", &[Tag(1), Typ(2), Typ(3)]),
    t(32, "BQ3", &[Tag(2), Typ(1), Typ(2)]),
    t(33, "BQ4", &[Tag(2), Typ(2), Typ(3)]),
    t(34, "TQ1", &[Tag(1), Typ(0), Typ(1), Typ(2)]),
    t(35, "TQ2", &[Tag(1), Typ(1), Typ(2), Typ(3)]),
    t(36, "TQ3", &[Tag(2), Typ(0), Typ(1), Typ(2)]),
    t(37, "TQ4", &[Tag(2), Typ(1), Typ(2), Typ(3)]),
    // Language-specific char + char-type mixed features (Japanese/Chinese
    // only); kept last so templates_for can slice.
    t(38, "WC1", &[Chr(2), Typ(3)]),
    t(39, "WC2", &[Typ(2), Chr(3)]),
    t(40, "WC3", &[Chr(2), Typ(2)]),
    t(41, "WC4", &[Chr(3), Typ(3)]),
];

/// Number of templates shared by all languages (everything before `WC1`).
const BASE_TEMPLATE_COUNT: usize = 38;

/// Returns the templates applicable to `language`, in emission order.
///
/// Japanese and Chinese use all 42 templates; other languages use the 38
/// base templates (the `WC*` char/type mixed features are excluded for
/// Korean because its uniform syllable types would make them noise).
pub(crate) fn templates_for(language: Language) -> &'static [Template] {
    match language {
        Language::Japanese | Language::Chinese => &TEMPLATES[..],
        _ => &TEMPLATES[..BASE_TEMPLATE_COUNT],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_ids_match_indices() {
        for (i, template) in TEMPLATES.iter().enumerate() {
            assert_eq!(template.id as usize, i, "id mismatch at {}", template.prefix);
        }
    }

    #[test]
    fn test_template_prefixes_distinct() {
        for (i, a) in TEMPLATES.iter().enumerate() {
            for b in &TEMPLATES[i + 1..] {
                assert_ne!(a.prefix, b.prefix);
            }
        }
    }

    #[test]
    fn test_template_slot_deltas_in_range() {
        for template in &TEMPLATES {
            for slot in template.slots {
                match *slot {
                    Slot::Tag(d) => assert!(d <= 2, "{}: tag delta {d}", template.prefix),
                    Slot::Chr(d) | Slot::Typ(d) => {
                        assert!(d <= 5, "{}: delta {d}", template.prefix)
                    }
                }
            }
        }
    }

    #[test]
    fn test_templates_for_language_gating() {
        assert_eq!(templates_for(Language::Japanese).len(), 42);
        assert_eq!(templates_for(Language::Chinese).len(), 42);
        assert_eq!(templates_for(Language::Korean).len(), 38);
        // The gated tail is exactly the WC* templates.
        for template in &TEMPLATES[BASE_TEMPLATE_COUNT..] {
            assert!(template.prefix.starts_with("WC"));
        }
    }
}
