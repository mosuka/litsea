//! Declarative feature-template table and the compiled two-pass scoring
//! model.
//!
//! This module is the single source of truth for the segmentation feature
//! template (issues #136/#138/#139). Four consumers derive from the
//! [`TEMPLATES`] table:
//!
//! 1. The string writer ([`crate::segmenter::Segmenter`]'s `write_attributes`),
//!    which renders feature strings for training and extraction in the
//!    table's emission order.
//! 2. The load-time parser ([`parse_feature_keys`]), which converts a
//!    trained model's string feature keys into packed integer keys once,
//!    when the model is compiled into a [`PackedModel`].
//! 3. The two-pass scorer in `segment()`, which reads the compiled tables:
//!    a static pass scatter-adds every tag-free feature (merged `UW`/`BW`
//!    probes, `WC` probes, `UC`/`BC`/`TC` dense loads) into a per-position
//!    buffer, and a sequential pass adds the 16 tag-dependent dense loads.
//! 4. The multiclass twin of 2 and 3 for the POS path
//!    ([`crate::packed_pos_model::PackedPosModel`], issue #143), which
//!    compiles the Averaged Perceptron's per-class weight rows into the
//!    same two-pass table structure for `segment_with_pos()`.
//!
//! The table order still defines the string writer's emission sequence
//! (model files and training data depend on it). Scoring accumulates in
//! two-pass order instead, so segmentation output is not bit-for-bit
//! guaranteed against the string-keyed reference; it is pinned empirically
//! by the exact-equality differential tests (zero divergence across all
//! bundled models and corpora). The language-gated `WC1`..`WC4` templates
//! sit last so that [`templates_for`] can hand out a prefix slice.

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
    /// The packed key for this feature.
    ///
    /// Test-only since the two-pass scorers: production code (both the
    /// AdaBoost path and the packed POS path) derives keys at build time
    /// via [`parse_feature_keys`] and scores through the merged tables, but
    /// the pack/parse roundtrip tests keep pinning the key encoding that
    /// the builders decode.
    #[cfg(test)]
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
    /// table ([`PackedModel::dense`]). 29 of the 42 templates qualify (all
    /// but `UW*`, `BW*`, `WC*`, whose weights live in the merged-vector
    /// hash tables).
    pub(crate) fn is_dense(&self) -> bool {
        self.slots.iter().all(|slot| !matches!(slot, Slot::Chr(_)))
    }

    /// Returns true when the template reads at least one boundary tag, i.e.
    /// it depends on earlier segmentation decisions and must be scored in
    /// the sequential pass (`UP*`, `BP*`, `UQ*`, `BQ*`, `TQ*` — 16
    /// templates). Tag-free templates depend only on the input text and are
    /// scored in the static pass. The hot loop uses hard-coded indices
    /// pinned by tests against the id ranges (`TAG_HEAD_IDS` etc.);
    /// production calls this only once per model compilation, to decide
    /// [`PackedModel::has_tag_features`] (issue #183).
    pub(crate) fn has_tag_slot(&self) -> bool {
        self.slots.iter().any(|slot| matches!(slot, Slot::Tag(_)))
    }

    /// Decodes the (char code, type id) pair out of a packed key of a
    /// template with exactly one `Chr` and one `Typ` slot (the `WC*`
    /// templates), walking the slots with their pack widths (24 bits per
    /// `Chr`, 8 per `Tag`/`Typ`). Shared by both packed-model builders.
    pub(crate) fn decode_chr_typ(&self, key: u64) -> (u32, u8) {
        let mut shift: u32 = self
            .slots
            .iter()
            .map(|slot| if matches!(slot, Slot::Chr(_)) { 24 } else { 8 })
            .sum();
        let (mut chr, mut typ) = (0u32, 0u8);
        for slot in self.slots {
            let width = if matches!(slot, Slot::Chr(_)) { 24 } else { 8 };
            shift -= width;
            let value = (key >> shift) & ((1u64 << width) - 1);
            match slot {
                Slot::Chr(_) => chr = value as u32,
                Slot::Typ(_) => typ = value as u8,
                Slot::Tag(_) => {}
            }
        }
        (chr, typ)
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
    ///
    /// Test-only since the two-pass scorers hard-code the mixed-radix
    /// expressions per family (both `segment()` and the packed POS scorer's
    /// `predict_seq`): this remains the canonical definition the hard-coded
    /// indices are pinned against.
    #[cfg(test)]
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
    pub(crate) fn dense_index_from_key(&self, key: u64, type_radix: usize) -> usize {
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

/// Template-id ranges of the template families, used to route packed keys
/// into the per-family tables and to partition scoring into the static and
/// sequential passes. The ids are pinned by `test_template_ids_match_indices`
/// and the partition by `test_family_ranges_match_predicates`.
/// `UW*`: sparse (char keyed), tag-free — scored in the static pass.
pub(crate) const UW_IDS: std::ops::Range<usize> = 5..11;
/// `BW*`: sparse (char-bigram keyed), tag-free — scored in the static pass.
pub(crate) const BW_IDS: std::ops::Range<usize> = 11..14;
/// `UC*`, `BC*`, `TC*`: dense, tag-free — scored in the static pass.
pub(crate) const TYPE_ONLY_IDS: std::ops::Range<usize> = 14..27;
/// First template id of the `BC*` family (within [`TYPE_ONLY_IDS`]).
pub(crate) const BC_FIRST_ID: usize = 20;
/// First template id of the `TC*` family (within [`TYPE_ONLY_IDS`]).
pub(crate) const TC_FIRST_ID: usize = 23;
/// `UP*`, `BP*`: dense, tag-dependent — scored in the sequential pass
/// (via hard-coded indices pinned by tests against these ranges).
#[cfg(test)]
pub(crate) const TAG_HEAD_IDS: std::ops::Range<usize> = 0..5;
/// `UQ*`, `BQ*`, `TQ*`: dense, tag-dependent — scored in the sequential
/// pass (via hard-coded indices pinned by tests against these ranges).
#[cfg(test)]
pub(crate) const TAG_TAIL_IDS: std::ops::Range<usize> = 27..38;
/// `WC*`: sparse (char/type keyed), tag-free — scored in the static pass;
/// emitted only for languages using all templates (see [`templates_for`]).
pub(crate) const WC_IDS: std::ops::Range<usize> = 38..42;

/// Builds the slot-order-normalized key of a `WC*` weight in
/// [`PackedModel::wc`]: the template's index within the `WC` family plus
/// its (char code, type id) pair, independent of which slot renders first.
#[inline]
pub(crate) fn wc_key(wc_index: usize, chr: u32, typ: u8) -> u64 {
    ((wc_index as u64) << 32) | (u64::from(chr) << 8) | u64::from(typ)
}

/// Returns true when `feature` (a rendered attribute string such as
/// `"UP1:U"`, or a model-file line starting with one) belongs to one of the
/// 16 tag-dependent templates (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`). Used by
/// [`Extractor`](crate::extractor::Extractor)'s tag-free extraction
/// (issue #183) to drop those features so the trained model is pointwise
/// and [`PackedModel::has_tag_features`] stays false.
pub(crate) fn is_tag_dependent_feature(feature: &str) -> bool {
    TEMPLATES.iter().any(|t| {
        t.has_tag_slot()
            && feature.starts_with(t.prefix)
            && feature.as_bytes().get(t.prefix.len()) == Some(&b':')
    })
}

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

/// An [`AdaBoost`]-format model compiled for two-pass scoring in
/// `segment()`.
///
/// "AdaBoost model" here means the on-disk format/struct, not necessarily
/// AdaBoost boosting as the training algorithm: this also compiles
/// two-stage models' stage-1 boundary classifier, which is trained as a
/// 2-class Averaged Perceptron and losslessly collapsed into `AdaBoost`
/// format (see `crate::trainer`'s module docs) before it ever reaches this
/// builder.
///
/// Char-bearing templates live in merged-vector hash tables — one probe
/// covers a whole family at a text position, and the contributions are
/// scatter-added into a per-position static score in a single pass over the
/// sentence. Tag/type-only templates keep the direct-indexed dense tables
/// from the previous design. The bias is read from the learner.
#[derive(Debug)]
pub(crate) struct PackedModel {
    /// Char code -> `[UW1..UW6]` weights: one probe per text position
    /// covers all six unigram-word templates (scatter-added to the six
    /// neighboring decision positions).
    pub(crate) uw: FxHashMap<u32, [f64; 6]>,
    /// `(a << 24) | b` (adjacent char codes) -> `[BW1..BW3]` weights: one
    /// probe per adjacent pair covers all three bigram-word templates.
    pub(crate) bw: FxHashMap<u64, [f64; 3]>,
    /// Char code -> merged `WC*` row (Japanese/Chinese only): a flat
    /// `[slot 0..4][type_id 0..type_radix]` array (length
    /// `4 * type_radix`), so one probe per text position covers all four
    /// char/type templates; the type dimension is direct-indexed. Slot
    /// order follows the `WC1`..`WC4` template order (#157).
    pub(crate) wc: FxHashMap<u32, Box<[f64]>>,
    /// Type id -> `[UC1..UC6]` weights: the scatter twin of the `UC*`
    /// dense tables (derived from `dense`, one direct index per position).
    pub(crate) uc: Vec<[f64; 6]>,
    /// Type-id pair (`t1 * T + t2`) -> `[BC1..BC3]` weights.
    pub(crate) bc: Vec<[f64; 3]>,
    /// Type-id triple (`(t1 * T + t2) * T + t3`) -> `[TC1..TC4]` weights.
    pub(crate) tc: Vec<[f64; 4]>,
    /// Dense weight tables indexed by template id, one per
    /// [`Template::is_dense`] template (empty `Vec` for the char-bearing
    /// templates, which live in the maps above). Entry order is the
    /// mixed-radix index of [`Template::dense_index`]; unset entries stay
    /// `0.0`, equivalent to a hash-map miss. The canonical store for the
    /// tag-dependent sequential pass; the `uc`/`bc`/`tc` scatter vectors
    /// above are derived views of ids 14..27.
    pub(crate) dense: Vec<Vec<f64>>,
    /// True iff any tag-dependent (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`) table
    /// holds a non-zero weight. When false, the model is pointwise: the 16
    /// tag-dependent loads all add `0.0`, so `segment()` skips the
    /// sequential pass and its tag bookkeeping entirely (issue #183). The
    /// two paths are exactly equivalent — the skipped loads contribute
    /// nothing — so the gate cannot change output.
    pub(crate) has_tag_features: bool,
}

impl PackedModel {
    /// Compiles the learner's string-keyed weights into the two-pass
    /// scoring tables for `language`. Called once per model (re)load, not
    /// on the hot path.
    ///
    /// Dense-eligible templates get a 0.0-initialized table sized by
    /// [`Template::dense_size`]; `UW*`/`BW*` keys are decoded into the
    /// merged-vector maps (family slot = template id offset within the
    /// family); `WC*` keys are decoded into per-character rows laid out
    /// `[slot][type_id]` (see the `wc` field docs).
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
        let mut uw: FxHashMap<u32, [f64; 6]> = FxHashMap::default();
        let mut bw: FxHashMap<u64, [f64; 3]> = FxHashMap::default();
        let mut wc: FxHashMap<u32, Box<[f64]>> = FxHashMap::default();
        let mut dense: Vec<Vec<f64>> = TEMPLATES
            .iter()
            .map(|t| if t.is_dense() { vec![0.0; t.dense_size(type_radix)] } else { Vec::new() })
            .collect();
        let mut keys = Vec::new();
        for (feature, weight) in learner.feature_weights() {
            keys.clear();
            parse_feature_keys(language, feature, &mut keys);
            for &key in &keys {
                let tid = (key >> 56) as usize;
                let template = &TEMPLATES[tid];
                if template.is_dense() {
                    let idx = template.dense_index_from_key(key, type_radix);
                    dense[tid][idx] = weight;
                } else if UW_IDS.contains(&tid) {
                    // Single Chr slot: the char code is the low 24 bits.
                    let code = (key & 0xFF_FFFF) as u32;
                    uw.entry(code).or_insert([0.0; 6])[tid - UW_IDS.start] = weight;
                } else if BW_IDS.contains(&tid) {
                    // Two Chr slots: (a << 24) | b in the low 48 bits, the
                    // same layout the scatter pass builds from adjacent
                    // char codes.
                    bw.entry(key & 0xFFFF_FFFF_FFFF).or_insert([0.0; 3])[tid - BW_IDS.start] =
                        weight;
                } else {
                    let (chr, typ) = template.decode_chr_typ(key);
                    let row = wc
                        .entry(chr)
                        .or_insert_with(|| vec![0.0; 4 * type_radix].into_boxed_slice());
                    row[(tid - WC_IDS.start) * type_radix + typ as usize] = weight;
                }
            }
        }
        // Derive the scatter twins of the type-only dense tables: family
        // slot k of entry v holds dense[first_id + k][v]. The dense tables
        // stay canonical; these views trade memory (a few KB) for one direct
        // index per position in the static pass.
        let t = type_radix;
        let uc: Vec<[f64; 6]> = (0..t)
            .map(|v| std::array::from_fn(|k| dense[TYPE_ONLY_IDS.start + k][v]))
            .collect();
        let bc: Vec<[f64; 3]> =
            (0..t * t).map(|v| std::array::from_fn(|k| dense[BC_FIRST_ID + k][v])).collect();
        let tc: Vec<[f64; 4]> = (0..t * t * t)
            .map(|v| std::array::from_fn(|k| dense[TC_FIRST_ID + k][v]))
            .collect();
        // One linear scan at build time decides the pointwise fast path
        // (#183): a model whose tag-dependent tables are all zero scores
        // identically without the sequential pass.
        let has_tag_features = TEMPLATES
            .iter()
            .enumerate()
            .any(|(tid, tpl)| tpl.has_tag_slot() && dense[tid].iter().any(|&w| w != 0.0));
        PackedModel {
            uw,
            bw,
            wc,
            uc,
            bc,
            tc,
            dense,
            has_tag_features,
        }
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
        // UW4:い (char-bearing) lands in the merged uw table at family slot
        // 3; BC2:OI (tag/type-only) lands in its dense table; UC1:SN (Korean
        // code) and ZZZ:x parse for no Japanese template and are skipped.
        assert_eq!(packed.uw.len(), 1);
        assert_eq!(packed.uw.get(&u32::from('い')), Some(&[0.0, 0.0, 0.0, 0.5, 0.0, 0.0]));
        assert!(packed.bw.is_empty());
        assert!(packed.wc.is_empty());
        // BC2 = [Typ(2), Typ(3)] (id 21) with Japanese ids O=0, I=6:
        // mixed-radix index = 0 * 8 + 6.
        assert_eq!(packed.dense[21][6], -0.25);
        // UC1:SN was skipped: its dense table stays all-zero.
        assert!(packed.dense[14].iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_is_tag_dependent_feature() {
        // Every tag-dependent template prefix matches; every tag-free
        // prefix does not; near-misses without the ':' separator or with a
        // longer prefix do not.
        for f in ["UP1:U", "BP2:BO", "UQ3:BH", "BQ4:OII", "TQ1:UOOI"] {
            assert!(is_tag_dependent_feature(f), "{f} should be tag-dependent");
        }
        // Model-file lines (feature + tab + weight) match on their prefix.
        assert!(is_tag_dependent_feature("UP1:U\t0.5"));
        for f in ["UW1:あ", "BW2:xy", "UC1:H", "BC2:OI", "TC1:III", "WC1:あI", "UP1", "UPX:U", ""]
        {
            assert!(!is_tag_dependent_feature(f), "{f} should not be tag-dependent");
        }
    }

    #[test]
    fn test_has_tag_features_flag() {
        fn build(model: &str) -> PackedModel {
            let mut learner = AdaBoost::new(0.01, 100);
            learner.load_model_from_reader(model.as_bytes()).unwrap();
            PackedModel::build(Language::Japanese, &learner)
        }

        // Tag-free model: only static-pass families -> pointwise.
        assert!(!build("UW3:あ\t0.5\nBC2:OI\t-0.25\n0.0\n").has_tag_features);
        // Any non-zero tag-dependent weight flips the flag, family by family.
        assert!(build("UW3:あ\t0.5\nUP1:U\t0.1\n0.0\n").has_tag_features);
        assert!(build("BQ1:BII\t-0.1\n0.0\n").has_tag_features);
        assert!(build("TQ4:OIII\t0.1\n0.0\n").has_tag_features);
        // A tag feature carrying an explicit zero weight contributes
        // nothing to any score, so the model is still pointwise.
        assert!(!build("UW3:あ\t0.5\nUP1:U\t0\n0.0\n").has_tag_features);
        // An empty model is pointwise (all tables zero).
        assert!(!build("0.0\n").has_tag_features);
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
        assert!(packed.uw.is_empty() && packed.bw.is_empty() && packed.wc.is_empty());
    }

    // --- Two-pass scoring tests (#139) ---

    #[test]
    fn test_family_ranges_match_predicates() {
        // The pinned id ranges used for build routing and pass partitioning
        // must agree with the slot-derived predicates.
        for (i, template) in TEMPLATES.iter().enumerate() {
            assert_eq!(UW_IDS.contains(&i), template.prefix.starts_with("UW"));
            assert_eq!(BW_IDS.contains(&i), template.prefix.starts_with("BW"));
            assert_eq!(WC_IDS.contains(&i), template.prefix.starts_with("WC"));
            assert_eq!(
                TYPE_ONLY_IDS.contains(&i),
                template.is_dense() && !template.has_tag_slot(),
                "{}",
                template.prefix
            );
            assert_eq!(
                TAG_HEAD_IDS.contains(&i) || TAG_TAIL_IDS.contains(&i),
                template.is_dense() && template.has_tag_slot(),
                "{}",
                template.prefix
            );
        }
        assert_eq!(TYPE_ONLY_IDS.len(), 13);
        assert_eq!(TAG_HEAD_IDS.len() + TAG_TAIL_IDS.len(), 16);
    }

    #[test]
    fn test_sequential_pass_indices_match_dense_index() {
        // The two-pass scorer hard-codes the mixed-radix index expressions
        // for the 16 tag-dependent templates. Pin them against the
        // canonical Template::dense_index over every (tags, types)
        // combination for the largest type radix in use.
        let t = 10usize; // Korean type-code count (largest)
        for p1 in 0..3usize {
            for p2 in 0..3usize {
                for p3 in 0..3usize {
                    for c1 in 0..t {
                        for c2 in 0..t {
                            for c3 in 0..t {
                                for c4 in 0..t {
                                    // Position i = 4 reads context indices
                                    // 1..=3 (tags) and 1..=4 (types).
                                    let tags = [0, p1 as u8, p2 as u8, p3 as u8, 0, 0, 0];
                                    let types = [0, c1 as u8, c2 as u8, c3 as u8, c4 as u8, 0, 0];
                                    let expected: [usize; 16] = [
                                        p1,
                                        p2,
                                        p3,
                                        p1 * 3 + p2,
                                        p2 * 3 + p3,
                                        p1 * t + c1,
                                        p2 * t + c2,
                                        p3 * t + c3,
                                        (p2 * t + c2) * t + c3,
                                        (p2 * t + c3) * t + c4,
                                        (p3 * t + c2) * t + c3,
                                        (p3 * t + c3) * t + c4,
                                        ((p2 * t + c1) * t + c2) * t + c3,
                                        ((p2 * t + c2) * t + c3) * t + c4,
                                        ((p3 * t + c1) * t + c2) * t + c3,
                                        ((p3 * t + c2) * t + c3) * t + c4,
                                    ];
                                    let ids = TAG_HEAD_IDS.chain(TAG_TAIL_IDS);
                                    for (tid, exp) in ids.zip(expected) {
                                        assert_eq!(
                                            TEMPLATES[tid].dense_index(4, &tags, &types, t),
                                            exp,
                                            "{}",
                                            TEMPLATES[tid].prefix
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_static_scatter_offsets_match_slots() {
        // The static pass scatter-adds with hard-coded position offsets:
        // unigram families (UW/UC) at i = q + 3 - k, bigram families
        // (BW/BC) at i = q + 2 - k, TC at i = q + 3 - k, and WC gathered at
        // (w3,c4), (c3,w4), (w3,c3), (w4,c4). Derive each template's true
        // offset from its slot deltas and pin the hard-coded values.
        for (k, tid) in UW_IDS.enumerate() {
            let Slot::Chr(d) = TEMPLATES[tid].slots[0] else { panic!() };
            // Slot reads context q = i - 3 + d, so i = q + 3 - d and d == k.
            assert_eq!(d as usize, k, "{}", TEMPLATES[tid].prefix);
        }
        for (k, tid) in (TYPE_ONLY_IDS.start..BC_FIRST_ID).enumerate() {
            let Slot::Typ(d) = TEMPLATES[tid].slots[0] else { panic!() };
            assert_eq!(d as usize, k, "{}", TEMPLATES[tid].prefix);
        }
        for (k, tid) in BW_IDS.enumerate() {
            let [Slot::Chr(a), Slot::Chr(b)] = TEMPLATES[tid].slots else { panic!() };
            // Pair (q, q+1) with q = i - 3 + a, adjacency b = a + 1;
            // i = q + 2 - k requires a == k + 1.
            assert_eq!((*a as usize, *b as usize), (k + 1, k + 2), "{}", TEMPLATES[tid].prefix);
        }
        for (k, tid) in (BC_FIRST_ID..TC_FIRST_ID).enumerate() {
            let [Slot::Typ(a), Slot::Typ(b)] = TEMPLATES[tid].slots else { panic!() };
            assert_eq!((*a as usize, *b as usize), (k + 1, k + 2), "{}", TEMPLATES[tid].prefix);
        }
        for (k, tid) in (TC_FIRST_ID..TYPE_ONLY_IDS.end).enumerate() {
            let [Slot::Typ(a), Slot::Typ(b), Slot::Typ(c)] = TEMPLATES[tid].slots else {
                panic!()
            };
            // Triple (q, q+1, q+2) with q = i - 3 + a; i = q + 3 - k
            // requires a == k.
            assert_eq!(
                (*a as usize, *b as usize, *c as usize),
                (k, k + 1, k + 2),
                "{}",
                TEMPLATES[tid].prefix
            );
        }
        // WC context deltas: (Chr, Typ) pairs per family index.
        let expected_wc = [(2u8, 3u8), (3, 2), (2, 2), (3, 3)];
        for (k, tid) in WC_IDS.enumerate() {
            let (mut chr_d, mut typ_d) = (0u8, 0u8);
            for slot in TEMPLATES[tid].slots {
                match *slot {
                    Slot::Chr(d) => chr_d = d,
                    Slot::Typ(d) => typ_d = d,
                    Slot::Tag(_) => panic!(),
                }
            }
            assert_eq!((chr_d, typ_d), expected_wc[k], "{}", TEMPLATES[tid].prefix);
        }
    }

    #[test]
    fn test_scatter_vectors_match_dense_tables() {
        // The uc/bc/tc scatter vectors are derived views of the dense
        // tables; verify the derivation on a model with features in each
        // family.
        let model = "UC3:I\t0.1\nBC2:HI\t0.2\nTC4:IHK\t0.3\n0.0\n";
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_reader(model.as_bytes()).unwrap();
        let packed = PackedModel::build(Language::Japanese, &learner);
        let t = 8usize;
        // Japanese ids: I=6, H=5, K=7. UC3 = family slot 2; BC2 = slot 1;
        // TC4 = slot 3.
        assert_eq!(packed.uc[6][2], 0.1);
        assert_eq!(packed.bc[5 * t + 6][1], 0.2);
        assert_eq!(packed.tc[(6 * t + 5) * t + 7][3], 0.3);
        for (v, arr) in packed.uc.iter().enumerate() {
            for (k, w) in arr.iter().enumerate() {
                assert_eq!(*w, packed.dense[TYPE_ONLY_IDS.start + k][v]);
            }
        }
    }

    #[test]
    fn test_build_merged_uw_bw_wc_tables() {
        // Family slots and key layouts of the merged tables, including
        // sentinel chars and both WC slot orders (WC2 renders type first).
        let model = "UW1:B2\t0.1\nUW6:あ\t0.2\nBW1:B1あ\t0.3\nBW3:あい\t0.4\nWC2:Iい\t0.5\nWC4:いI\t0.6\n0.0\n";
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_reader(model.as_bytes()).unwrap();
        let packed = PackedModel::build(Language::Japanese, &learner);

        let b1 = SENTINEL_BASE + 2;
        let b2 = SENTINEL_BASE + 1;
        assert_eq!(packed.uw.get(&b2), Some(&[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(packed.uw.get(&u32::from('あ')), Some(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]));
        let bw1_key = (u64::from(b1) << 24) | u64::from('あ');
        let bw3_key = (u64::from('あ') << 24) | u64::from('い');
        assert_eq!(packed.bw.get(&bw1_key), Some(&[0.3, 0.0, 0.0]));
        assert_eq!(packed.bw.get(&bw3_key), Some(&[0.0, 0.0, 0.4]));
        // Japanese type id I = 6; both WC features share the char 'い', so
        // they land in one row, laid out [slot 0..4][type_id 0..radix] with
        // WC2 at slot 1 and WC4 at slot 3.
        let radix = Language::Japanese.type_codes().len();
        let row = packed.wc.get(&u32::from('い')).expect("row for い");
        assert_eq!(row.len(), 4 * radix);
        assert_eq!(row[radix + 6], 0.5);
        assert_eq!(row[3 * radix + 6], 0.6);
        assert_eq!(row.iter().filter(|w| **w != 0.0).count(), 2);
        assert_eq!(packed.wc.len(), 1);
    }
}
