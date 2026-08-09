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

use rustc_hash::FxHashMap;

use crate::adaboost::AdaBoost;
use crate::language::Language;

/// Boundary-tag id for "U" (unknown / padding).
pub(crate) const TAG_U: u8 = 0;
/// Boundary-tag id for "B" (word start).
pub(crate) const TAG_B: u8 = 1;
/// Boundary-tag id for "O" (word continuation).
pub(crate) const TAG_O: u8 = 2;
/// Tag strings indexed by tag id.
const TAG_STRS: [&str; 3] = ["U", "B", "O"];

/// Padding sentinel strings in context order: B3/B2/B1 precede the text,
/// E1/E2/E3 follow it. Index `k` maps to char code `SENTINEL_BASE + k`.
pub(crate) const SENTINELS: [&str; 6] = ["B3", "B2", "B1", "E1", "E2", "E3"];
/// First sentinel char code, directly above the Unicode scalar range
/// (`char::MAX` is U+10FFFF), so sentinel codes can never collide with a
/// real character's code point.
pub(crate) const SENTINEL_BASE: u32 = 0x11_0000;

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

impl Template {
    /// Packs this template's feature at position `i` into a `u64` key.
    ///
    /// Layout: the template id occupies the top byte (bits 56..64); the slot
    /// values are shift-accumulated below it in slot order — 8 bits per
    /// `Tag`/`Typ` slot, 24 bits per `Chr` slot. The widest template (`BW*`,
    /// two `Chr` slots) uses 48 slot bits, so the payload never reaches the
    /// id byte, and every slot value is strictly below its field width
    /// (tag ids < 3, type ids < 256, char codes <= 0x110005 < 2^24). The
    /// encoding is therefore injective over (template, slot values).
    ///
    /// # Arguments
    /// * `i` - The character position (same convention as the string writer).
    /// * `tags` - Boundary-tag ids per position (`TAG_U`/`TAG_B`/`TAG_O`).
    /// * `chars` - Char codes per position (code points; sentinels are
    ///   `SENTINEL_BASE + k`).
    /// * `types` - Type ids per position ([`Language::char_type_id`]).
    ///
    /// # Returns
    /// The packed key for looking up this feature's weight in a
    /// [`PackedModel`].
    #[inline]
    pub(crate) fn pack(&self, i: usize, tags: &[u8], chars: &[u32], types: &[u8]) -> u64 {
        let mut acc = 0u64;
        for slot in self.slots {
            acc = match *slot {
                Slot::Tag(d) => (acc << 8) | u64::from(tags[i - 3 + d as usize]),
                Slot::Typ(d) => (acc << 8) | u64::from(types[i - 3 + d as usize]),
                Slot::Chr(d) => (acc << 24) | u64::from(chars[i - 3 + d as usize]),
            };
        }
        (u64::from(self.id) << 56) | acc
    }

    /// Returns true when every slot is a `Tag`/`Typ` slot, i.e. the
    /// template's key space is the small mixed-radix product of the tag and
    /// type domains and its weights can live in a direct-indexed dense
    /// table ([`PackedModel::dense`]) instead of the hash map. 29 of the 42
    /// templates qualify (all but `UW*`, `BW*`, `WC*`).
    pub(crate) fn is_dense(&self) -> bool {
        self.slots.iter().all(|slot| !matches!(slot, Slot::Chr(_)))
    }

    /// Number of entries in this template's dense table for a language with
    /// `type_radix` type codes: the mixed-radix product of the slot domains
    /// (3 per `Tag` slot, `type_radix` per `Typ` slot). Only meaningful for
    /// [`is_dense`](Self::is_dense) templates.
    pub(crate) fn dense_size(&self, type_radix: usize) -> usize {
        debug_assert!(self.is_dense());
        self.slots.iter().fold(1, |acc, slot| match slot {
            Slot::Tag(_) => acc * TAG_RADIX,
            Slot::Typ(_) => acc * type_radix,
            Slot::Chr(_) => acc, // unreachable: dense templates have no Chr slot
        })
    }

    /// Mixed-radix dense-table index of this template's feature at position
    /// `i`: `idx = idx * radix + value` over the slots in order. Shares its
    /// definition with [`dense_index_from_key`](Self::dense_index_from_key)
    /// so the scorer and the table builder agree by construction. Only
    /// meaningful for [`is_dense`](Self::is_dense) templates.
    ///
    /// # Arguments
    /// * `i` - The character position (same convention as [`pack`](Self::pack)).
    /// * `tags` - Boundary-tag ids per position.
    /// * `types` - Type ids per position.
    /// * `type_radix` - The language's type-code count.
    ///
    /// # Returns
    /// An index strictly below [`dense_size`](Self::dense_size).
    #[inline]
    pub(crate) fn dense_index(
        &self,
        i: usize,
        tags: &[u8],
        types: &[u8],
        type_radix: usize,
    ) -> usize {
        debug_assert!(self.is_dense());
        let mut idx = 0usize;
        for slot in self.slots {
            idx = match *slot {
                Slot::Tag(d) => idx * TAG_RADIX + tags[i - 3 + d as usize] as usize,
                Slot::Typ(d) => idx * type_radix + types[i - 3 + d as usize] as usize,
                Slot::Chr(_) => idx, // unreachable: dense templates have no Chr slot
            };
        }
        idx
    }

    /// Recomputes the dense-table index from a packed key produced by
    /// [`pack`](Self::pack) or [`parse_feature_keys`]. Dense templates carry
    /// only 8-bit fields, decoded here in slot order and re-accumulated with
    /// the same mixed radices as [`dense_index`](Self::dense_index).
    fn dense_index_from_key(&self, key: u64, type_radix: usize) -> usize {
        debug_assert!(self.is_dense());
        let n = self.slots.len();
        let mut idx = 0usize;
        for (j, slot) in self.slots.iter().enumerate() {
            let value = ((key >> (8 * (n - 1 - j))) & 0xFF) as usize;
            idx = match slot {
                Slot::Tag(_) => idx * TAG_RADIX + value,
                Slot::Typ(_) => idx * type_radix + value,
                Slot::Chr(_) => idx, // unreachable: dense templates have no Chr slot
            };
        }
        idx
    }
}

/// Radix of a tag slot (`TAG_U`/`TAG_B`/`TAG_O`).
const TAG_RADIX: usize = 3;

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

/// Parses a model feature string against `language`'s templates and pushes
/// the packed key of every slot-value tuple that renders exactly this
/// string. Strings no template can generate (unknown prefix, foreign type
/// codes, leftover input) push nothing — such features are unreachable from
/// the attribute writer for this language, so omitting them from the packed
/// map cannot change any score.
///
/// With the current grammar every parse is unique (tags are single
/// characters, type-code sets are prefix-free, and sentinel-vs-char choices
/// are forced by full consumption); the exhaustive collection is kept for
/// robustness against future template changes and is pinned by tests.
///
/// # Arguments
/// * `language` - The language whose templates and type codes to parse with.
/// * `feature` - The feature string from a trained model.
/// * `keys` - Output buffer; cleared by the caller, appended to here.
pub(crate) fn parse_feature_keys(language: Language, feature: &str, keys: &mut Vec<u64>) {
    for template in templates_for(language) {
        let Some(rest) = feature.strip_prefix(template.prefix).and_then(|r| r.strip_prefix(':'))
        else {
            continue;
        };
        let base = u64::from(template.id) << 56;
        parse_slots(language, template.slots, rest, base, 0, keys);
    }
}

/// Recursive helper for [`parse_feature_keys`]: tries every way the next
/// slot could have rendered the head of `rest`, and pushes `base | acc` as a
/// complete key when all slots and all input are consumed. `base` carries
/// the fixed template-id byte (bits 56..64) and is never shifted; `acc`
/// accumulates the slot values exactly as [`Template::pack`] does.
fn parse_slots(
    language: Language,
    slots: &[Slot],
    rest: &str,
    base: u64,
    acc: u64,
    keys: &mut Vec<u64>,
) {
    let Some((slot, tail)) = slots.split_first() else {
        if rest.is_empty() {
            keys.push(base | acc);
        }
        return;
    };
    match slot {
        Slot::Tag(_) => {
            for (id, tag) in TAG_STRS.iter().enumerate() {
                if let Some(r) = rest.strip_prefix(tag) {
                    parse_slots(language, tail, r, base, (acc << 8) | id as u64, keys);
                }
            }
        }
        Slot::Typ(_) => {
            for (id, code) in language.type_codes().iter().enumerate() {
                if let Some(r) = rest.strip_prefix(code) {
                    parse_slots(language, tail, r, base, (acc << 8) | id as u64, keys);
                }
            }
        }
        Slot::Chr(_) => {
            for (k, sentinel) in SENTINELS.iter().enumerate() {
                if let Some(r) = rest.strip_prefix(sentinel) {
                    let code = SENTINEL_BASE + k as u32;
                    parse_slots(language, tail, r, base, (acc << 24) | u64::from(code), keys);
                }
            }
            if let Some(c) = rest.chars().next() {
                let r = &rest[c.len_utf8()..];
                parse_slots(language, tail, r, base, (acc << 24) | u64::from(u32::from(c)), keys);
            }
        }
    }
}

/// A trained AdaBoost model compiled to integer feature keys for
/// allocation-free scoring in `segment()`'s hot loop. Tag/type-only
/// templates live in direct-indexed dense tables (single array load per
/// feature); char-bearing templates use a hash map probe on the packed
/// key. The bias is read from the learner.
#[derive(Debug)]
pub(crate) struct PackedModel {
    /// Char-bearing templates (`UW*`, `BW*`, `WC*`): packed key -> weight.
    pub(crate) weights: FxHashMap<u64, f64>,
    /// Dense weight tables indexed by template id, one per
    /// [`Template::is_dense`] template (empty `Vec` for map-scored
    /// templates, so the scorer dispatches on `is_empty()`). Entry order is
    /// the mixed-radix index of [`Template::dense_index`]; unset entries
    /// stay `0.0`, reproducing a hash-map miss exactly.
    pub(crate) dense: Vec<Vec<f64>>,
}

impl Default for PackedModel {
    /// An empty model: no map entries and zero-length dense tables. Every
    /// lookup dispatches to the map path and misses (adds 0.0), matching an
    /// untrained learner.
    fn default() -> Self {
        PackedModel {
            weights: FxHashMap::default(),
            dense: vec![Vec::new(); TEMPLATES.len()],
        }
    }
}

impl PackedModel {
    /// Compiles the learner's string-keyed weights into integer keys for
    /// `language`. Called once per model (re)load, not on the hot path.
    ///
    /// Dense-eligible templates get a 0.0-initialized table sized by
    /// [`Template::dense_size`]; their parsed keys are decoded into
    /// mixed-radix indices. All other keys go into the hash map.
    ///
    /// # Arguments
    /// * `language` - The language whose templates to compile for.
    /// * `learner` - The learner whose feature weights to compile.
    ///
    /// # Returns
    /// A `PackedModel` mirroring every feature of `learner` that the
    /// attribute writer can generate for `language`.
    pub(crate) fn build(language: Language, learner: &AdaBoost) -> Self {
        let type_radix = language.type_codes().len();
        let mut weights = FxHashMap::default();
        let mut dense: Vec<Vec<f64>> = TEMPLATES
            .iter()
            .map(|t| if t.is_dense() { vec![0.0; t.dense_size(type_radix)] } else { Vec::new() })
            .collect();
        let mut keys = Vec::new();
        for (feature, weight) in learner.feature_weights() {
            keys.clear();
            parse_feature_keys(language, feature, &mut keys);
            for &key in &keys {
                let template = &TEMPLATES[(key >> 56) as usize];
                if template.is_dense() {
                    let idx = template.dense_index_from_key(key, type_radix);
                    dense[template.id as usize][idx] = weight;
                } else {
                    weights.insert(key, weight);
                }
            }
        }
        PackedModel { weights, dense }
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

    /// Test context: parallel string/numeric arrays mirroring what the
    /// segmenter builds. Position i=4 is used, so Tag slots read indices
    /// 1..=3 and Chr/Typ slots read indices 1..=6. Characters include the
    /// sentinel-lookalike real chars 'B' and '1' on purpose.
    struct Ctx {
        tag_strs: [&'static str; 7],
        tag_ids: [u8; 7],
        char_strs: [&'static str; 7],
        char_codes: [u32; 7],
        type_ids: [u8; 7],
    }

    fn ctx_for(language: Language) -> Ctx {
        // Type ids 1..=6 exist for every language (all tables have >= 8
        // codes); index 0 stays "O".
        let type_ids = [0u8, 1, 2, 3, 4, 5, 6];
        Ctx {
            tag_strs: ["U", "U", "B", "O", "U", "U", "U"],
            tag_ids: [TAG_U, TAG_U, TAG_B, TAG_O, TAG_U, TAG_U, TAG_U],
            char_strs: ["B3", "B2", "B1", "B", "1", "漢", "E1"],
            char_codes: [
                SENTINEL_BASE,
                SENTINEL_BASE + 1,
                SENTINEL_BASE + 2,
                u32::from('B'),
                u32::from('1'),
                u32::from('漢'),
                SENTINEL_BASE + 3,
            ],
            type_ids: {
                let codes = language.type_codes();
                assert!(codes.len() >= 7);
                type_ids
            },
        }
    }

    fn render(language: Language, template: &Template, ctx: &Ctx) -> String {
        let mut s = String::new();
        s.push_str(template.prefix);
        s.push(':');
        for slot in template.slots {
            match *slot {
                Slot::Tag(d) => s.push_str(ctx.tag_strs[1 + d as usize]),
                Slot::Chr(d) => s.push_str(ctx.char_strs[1 + d as usize]),
                Slot::Typ(d) => {
                    s.push_str(language.type_codes()[ctx.type_ids[1 + d as usize] as usize])
                }
            }
        }
        s
    }

    #[test]
    fn test_pack_parse_roundtrip_unique_and_injective() {
        // For every language and template: rendering the feature string and
        // parsing it back must yield exactly the packed key (unique parse —
        // the current grammar is unambiguous), and distinct rendered strings
        // must map to distinct keys within a language.
        for language in [Language::Japanese, Language::Chinese, Language::Korean] {
            let ctx = ctx_for(language);
            let mut seen: FxHashMap<u64, String> = FxHashMap::default();
            for template in templates_for(language) {
                let rendered = render(language, template, &ctx);
                let key = template.pack(4, &ctx.tag_ids, &ctx.char_codes, &ctx.type_ids);
                let mut keys = Vec::new();
                parse_feature_keys(language, &rendered, &mut keys);
                assert_eq!(keys, vec![key], "{language}: {rendered}");
                if let Some(other) = seen.insert(key, rendered.clone()) {
                    panic!("{language}: key collision between {other} and {rendered}");
                }
            }
        }
    }

    #[test]
    fn test_parse_sentinel_vs_real_char_cases() {
        // BW1 = [Chr(1), Chr(2)] (id 11), BW2 = [Chr(2), Chr(3)] (id 12),
        // UW1 = [Chr(0)] (id 5). Full consumption forces a unique parse in
        // every sentinel-lookalike case.
        let cases: [(&str, u64); 5] = [
            // (sentinel B1, 'x')
            ("BW1:B1x", (11u64 << 56) | (u64::from(SENTINEL_BASE + 2) << 24) | u64::from('x')),
            // real chars 'B' then '1'
            ("BW2:B1", (12u64 << 56) | (u64::from('B') << 24) | u64::from('1')),
            // (sentinel B1, sentinel E1)
            (
                "BW1:B1E1",
                (11u64 << 56) | (u64::from(SENTINEL_BASE + 2) << 24) | u64::from(SENTINEL_BASE + 3),
            ),
            // single slot, two chars: must be the sentinel
            ("UW1:B1", (5u64 << 56) | u64::from(SENTINEL_BASE + 2)),
            // single slot, one char: real 'B'
            ("UW1:B", (5u64 << 56) | u64::from('B')),
        ];
        for (feature, expected) in cases {
            let mut keys = Vec::new();
            parse_feature_keys(Language::Japanese, feature, &mut keys);
            assert_eq!(keys, vec![expected], "{feature}");
        }
    }

    #[test]
    fn test_parse_korean_multichar_type_codes() {
        // Korean ids: E=4, SN=5, SF=6. BC2 = [Typ(2), Typ(3)] (id 21),
        // TC1 = [Typ(0), Typ(1), Typ(2)] (id 23), UQ1 = [Tag(0), Typ(0)]
        // (id 27).
        let cases: [(&str, u64); 3] = [
            ("BC2:SFN", (21u64 << 56) | (6 << 8) | 3),
            ("TC1:SNSFE", (23u64 << 56) | (5 << 16) | (6 << 8) | 4),
            ("UQ1:USN", (27u64 << 56) | (u64::from(TAG_U) << 8) | 5),
        ];
        for (feature, expected) in cases {
            let mut keys = Vec::new();
            parse_feature_keys(Language::Korean, feature, &mut keys);
            assert_eq!(keys, vec![expected], "{feature}");
        }
        // "S" alone is not a Korean code: no parse.
        let mut keys = Vec::new();
        parse_feature_keys(Language::Korean, "UC1:S", &mut keys);
        assert!(keys.is_empty());
    }

    #[test]
    fn test_parse_skips_unreachable_features() {
        let unparseable = [
            // Hiragana type code under Korean
            (Language::Korean, "UC2:I"),
            // WC templates are not part of the Korean template set
            (Language::Korean, "WC1:하H"),
            // Unknown prefix
            (Language::Japanese, "ZZZ:x"),
            // The bias bucket has no prefix at all
            (Language::Japanese, ""),
            // Chr slot with no input left
            (Language::Japanese, "UW1:"),
            // Trailing input after all slots
            (Language::Japanese, "UP1:UU"),
        ];
        for (language, feature) in unparseable {
            let mut keys = Vec::new();
            parse_feature_keys(language, feature, &mut keys);
            assert!(keys.is_empty(), "{language}: {feature:?} should not parse");
        }
    }

    #[test]
    fn test_packed_model_build_skips_foreign_features() {
        let model = "UW4:い\t0.5\nBC2:OI\t-0.25\nUC1:SN\t1.0\nZZZ:x\t2.0\n0.0\n";
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_reader(model.as_bytes()).unwrap();

        let packed = PackedModel::build(Language::Japanese, &learner);
        // UW4:い (char-bearing) lands in the map; BC2:OI (tag/type-only)
        // lands in its dense table; UC1:SN (Korean code) and ZZZ:x parse for
        // no Japanese template and are skipped entirely.
        assert_eq!(packed.weights.len(), 1);
        // UW4 = [Chr(3)] (id 8).
        let uw4 = (8u64 << 56) | u64::from('い');
        assert_eq!(packed.weights.get(&uw4), Some(&0.5));
        // BC2 = [Typ(2), Typ(3)] (id 21) with Japanese ids O=0, I=6:
        // mixed-radix index = 0 * 8 + 6.
        assert_eq!(packed.dense[21][6], -0.25);
        // UC1:SN was skipped: its dense table stays all-zero.
        assert!(packed.dense[14].iter().all(|&w| w == 0.0));
    }

    // --- Dense-table tests (#138) ---

    #[test]
    fn test_dense_partition() {
        // Exactly the char-bearing templates (UW*, BW*, WC*) are map-scored;
        // the other 29 are dense-eligible.
        let mut dense_count = 0;
        for template in &TEMPLATES {
            let char_bearing = template.prefix.starts_with("UW")
                || template.prefix.starts_with("BW")
                || template.prefix.starts_with("WC");
            assert_eq!(template.is_dense(), !char_bearing, "{}", template.prefix);
            if template.is_dense() {
                dense_count += 1;
            }
        }
        assert_eq!(dense_count, 29);
    }

    #[test]
    fn test_dense_sizes_japanese() {
        // Japanese has 8 type codes; expected mixed-radix products per
        // template family.
        let expected = [
            ("UP1", 3),
            ("BP1", 9),
            ("UC1", 8),
            ("BC1", 64),
            ("TC1", 512),
            ("UQ1", 24),
            ("BQ1", 192),
            ("TQ1", 1536),
        ];
        for (prefix, size) in expected {
            let template = TEMPLATES.iter().find(|t| t.prefix == prefix).unwrap();
            assert_eq!(template.dense_size(8), size, "{prefix}");
        }
    }

    #[test]
    fn test_dense_index_consistent_with_key_decode() {
        // For every dense template and language: the scorer's directly
        // computed index equals the index decoded from the packed key (the
        // builder's path), and both are within the table.
        for language in [Language::Japanese, Language::Chinese, Language::Korean] {
            let type_radix = language.type_codes().len();
            let ctx = ctx_for(language);
            for template in templates_for(language) {
                if !template.is_dense() {
                    continue;
                }
                let key = template.pack(4, &ctx.tag_ids, &ctx.char_codes, &ctx.type_ids);
                let from_key = template.dense_index_from_key(key, type_radix);
                let direct = template.dense_index(4, &ctx.tag_ids, &ctx.type_ids, type_radix);
                assert_eq!(from_key, direct, "{language}: {}", template.prefix);
                assert!(direct < template.dense_size(type_radix), "{}", template.prefix);
            }
        }
    }

    #[test]
    fn test_build_dense_storage_roundtrip() {
        // A weight stored through build's key-decode path must be readable
        // at the scorer's directly computed index. TQ4 = [Tag(2), Typ(1),
        // Typ(2), Typ(3)] (id 37) exercises the deepest mixed-radix case;
        // rendered for Japanese with p3=O, types A,N,I.
        let model = "TQ4:OANI\t0.75\n0.0\n";
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_reader(model.as_bytes()).unwrap();
        let packed = PackedModel::build(Language::Japanese, &learner);

        let template = TEMPLATES.iter().find(|t| t.prefix == "TQ4").unwrap();
        // Context where position 4 yields exactly p3=O ("O" at tags[3]) and
        // c2..c4 = A,N,I (type ids 2,3,6 at indices 2..=4).
        let tags = [TAG_U, TAG_U, TAG_U, TAG_O, TAG_U, TAG_U, TAG_U];
        let types = [0u8, 0, 2, 3, 6, 0, 0];
        let idx = template.dense_index(4, &tags, &types, 8);
        assert_eq!(packed.dense[37][idx], 0.75);
        assert!(packed.weights.is_empty());
    }
}
