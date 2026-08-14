//! Packed scoring tables for the two-stage POS tagger (issue #147).
//!
//! [`PackedTwoStageModel`] compiles a two-stage model's stage-2 tagger
//! (a string-keyed [`AveragedPerceptron`] over the word-level templates of
//! [`crate::word_features`]) and its candidate-tag lexicon into the
//! structures the tagging hot loop needs:
//!
//! - a **surface map**: one `FxHashMap<String, _>` probe per word covering
//!   the lexicon (candidate restriction, dominance-fixed tags) *and* the
//!   `WS` surface-feature weight row;
//! - **sparse class rows** keyed by packed integers for the char-valued
//!   templates, with per-template gating (a template with no weights costs
//!   nothing at runtime, so feature-subset models are automatically
//!   cheaper);
//! - **dense tables** for the type-valued and word-length templates
//!   (direct indexing, no hashing).
//!
//! Words whose surface has a single observed tag — or one that covers at
//! least the `dominance` fraction of its training occurrences — are tagged
//! without touching the classifier at all; this skip is the main cost
//! lever of the two-stage design (see #147). Ambiguous known words get a
//! candidate-masked argmax; unknown words fall back to the full argmax
//! over all classes.

use rustc_hash::FxHashMap;

use crate::language::Language;
use crate::perceptron::AveragedPerceptron;
use crate::upos::Upos;
use crate::word_features::{
    BOS_CODE, CONTEXT_WINDOW, EOS_CODE, F_CL1, F_CR1, F_FT, F_LT, N_TYPE_FAMILIES,
    N_WORD_TEMPLATES, T_FC, T_L1, T_LB, T_LC, T_P2, T_R1, T_RB, T_S2, T_TS, TS_CAP, WL_CAP,
    WordFeature, hash_key, parse_word_feature, ts_payload,
};

/// Build-time accumulator for one surface: fixed tag, candidate class
/// indices, and the `WS` weight row (frozen into a [`WordEntry`]).
type WordEntryBuild = (Option<Upos>, Vec<u16>, Vec<(u16, f64)>);

/// Per-surface entry of the packed surface map.
#[derive(Debug, Default)]
struct WordEntry {
    /// Tag assigned without scoring: present when the surface has a single
    /// observed tag or a dominant one (see `dominance`). May be a tag the
    /// classifier does not know (a tag observed only on unambiguous
    /// words).
    fixed: Option<Upos>,
    /// Candidate class indices for the masked argmax, sorted ascending
    /// (the perceptron's first-wins tie-break order). Lexicon tags the
    /// classifier does not know are dropped; if none remain the word is
    /// scored as unknown.
    candidates: Box<[u16]>,
    /// Sparse `(class, weight)` row of the `WS` surface feature.
    ws_row: Box<[(u16, f64)]>,
}

/// The compiled two-stage tagging tables. Built once per model (re)load by
/// [`build`](Self::build); consulted by
/// [`crate::segmenter::Segmenter::segment_with_pos`].
#[derive(Debug)]
pub(crate) struct PackedTwoStageModel {
    /// Class index -> UPOS tag (the stage-2 class order).
    classes: Box<[Upos]>,
    /// Number of stage-2 classes.
    n_classes: usize,
    /// `language.type_codes().len() + 2` (the two extra slots are the
    /// BOS/EOS sentinel type indices).
    type_stride: usize,
    /// Surface -> lexicon + `WS` data (single String probe per word).
    words: FxHashMap<String, WordEntry>,
    /// Packed integer key -> sparse `(class, weight)` row for the
    /// char-valued templates.
    hash: FxHashMap<u64, Box<[(u16, f64)]>>,
    /// Per-template gating for the hash-keyed templates.
    has: [bool; N_WORD_TEMPLATES],
    /// Dense word-length rows: `(WL_CAP + 1) * n_classes`.
    dense_wl: Box<[f64]>,
    /// Whether any `WL` weight exists.
    wl_used: bool,
    /// Dense type rows: `N_TYPE_FAMILIES * type_stride * n_classes`.
    dense_t: Box<[f64]>,
    /// Per-family gating for the dense type families.
    t_used: [bool; N_TYPE_FAMILIES],
}

impl PackedTwoStageModel {
    /// Compiles the stage-2 tagger and lexicon into packed tables for
    /// `language`. Called once per model (re)load, not on the hot path.
    ///
    /// # Arguments
    /// * `language` - The language whose type codes to compile for.
    /// * `stage2` - The stage-2 word-level tagger (classes are UPOS tags,
    ///   validated by [`crate::two_stage::TwoStageLearner`]).
    /// * `lexicon` - Surface -> observed `(tag, count)` candidates, most
    ///   frequent first (the [`crate::two_stage::TwoStageLearner`]
    ///   invariant).
    /// * `dominance` - The classifier-skip threshold in `(0.5, 1.0]`.
    ///
    /// # Returns
    /// The compiled model.
    pub(crate) fn build(
        language: Language,
        stage2: &AveragedPerceptron,
        lexicon: &FxHashMap<String, Vec<(Upos, u32)>>,
        dominance: f64,
    ) -> Self {
        let class_names = stage2.class_names();
        let n = class_names.len();
        // TwoStageLearner validates every class name as a UPOS tag; the
        // fallback mirrors PackedPosModel::build's defensive parse.
        let classes: Box<[Upos]> =
            class_names.iter().map(|c| c.parse().unwrap_or(Upos::X)).collect();
        let type_stride = language.type_codes().len() + 2;

        let mut words: FxHashMap<String, WordEntryBuild> = FxHashMap::default();
        let mut hash: FxHashMap<u64, Vec<(u16, f64)>> = FxHashMap::default();
        let mut has = [false; N_WORD_TEMPLATES];
        let mut dense_wl = vec![0.0f64; (WL_CAP + 1) * n].into_boxed_slice();
        let mut wl_used = false;
        let mut dense_t = vec![0.0f64; N_TYPE_FAMILIES * type_stride * n].into_boxed_slice();
        let mut t_used = [false; N_TYPE_FAMILIES];

        for (feature, class_weights) in stage2.feature_class_weights() {
            let sparse = || {
                class_weights
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| **w != 0.0)
                    .map(|(c, &w)| (c as u16, w))
            };
            match parse_word_feature(language, feature) {
                Some(WordFeature::Surface(surface)) => {
                    words.entry(surface.to_string()).or_default().2.extend(sparse());
                }
                Some(WordFeature::WordLen(len)) => {
                    dense_wl[len * n..][..n].copy_from_slice(class_weights);
                    wl_used = true;
                }
                Some(WordFeature::TypeDense { family, type_idx }) => {
                    dense_t[(family * type_stride + type_idx) * n..][..n]
                        .copy_from_slice(class_weights);
                    t_used[family] = true;
                }
                Some(WordFeature::Hash(key)) => {
                    let row = hash.entry(key).or_default();
                    row.extend(sparse());
                    if !row.is_empty() {
                        has[(key >> 56) as usize] = true;
                    }
                }
                // Features no word template of this language can render
                // are unreachable at inference and skipped, mirroring the
                // other packed builders.
                None => {}
            }
        }

        for (surface, entry) in lexicon {
            let slot = words.entry(surface.clone()).or_default();
            // The entry is sorted most-frequent-first with a deterministic
            // tie-break (TwoStageLearner invariant), so entry[0] is the
            // dominant candidate.
            let total: u32 = entry.iter().map(|(_, count)| count).sum();
            if entry.len() == 1 || f64::from(entry[0].1) / f64::from(total) >= dominance {
                slot.0 = Some(entry[0].0);
            }
            let mut candidates: Vec<u16> = entry
                .iter()
                .filter_map(|(tag, _)| classes.iter().position(|c| c == tag).map(|i| i as u16))
                .collect();
            candidates.sort_unstable();
            slot.1 = candidates;
        }

        let words = words
            .into_iter()
            .filter(|(_, (fixed, candidates, ws_row))| {
                fixed.is_some() || !candidates.is_empty() || !ws_row.is_empty()
            })
            .map(|(surface, (fixed, candidates, mut ws_row))| {
                ws_row.sort_unstable_by_key(|&(c, _)| c);
                (
                    surface,
                    WordEntry {
                        fixed,
                        candidates: candidates.into_boxed_slice(),
                        ws_row: ws_row.into_boxed_slice(),
                    },
                )
            })
            .collect();
        let hash = hash
            .into_iter()
            .filter(|(_, row)| !row.is_empty())
            .map(|(key, mut row)| {
                row.sort_unstable_by_key(|&(c, _)| c);
                (key, row.into_boxed_slice())
            })
            .collect();

        PackedTwoStageModel {
            classes,
            n_classes: n,
            type_stride,
            words,
            hash,
            has,
            dense_wl,
            wl_used,
            dense_t,
            t_used,
        }
    }

    /// Adds a sparse hash row to the score vector.
    #[inline]
    fn add_hash(&self, scores: &mut [f64], key: u64) {
        if let Some(row) = self.hash.get(&key) {
            for &(c, w) in row.iter() {
                scores[c as usize] += w;
            }
        }
    }

    /// Adds a dense type row to the score vector.
    #[inline]
    fn add_type(&self, scores: &mut [f64], family: usize, type_idx: usize) {
        if self.t_used[family] {
            let row = &self.dense_t[(family * self.type_stride + type_idx) * self.n_classes..]
                [..self.n_classes];
            for (s, w) in scores.iter_mut().zip(row) {
                *s += w;
            }
        }
    }

    /// Tags every word of a segmented sentence.
    ///
    /// The words must concatenate to the original sentence (the shape
    /// produced by [`crate::segmenter::Segmenter::segment`]); context
    /// features read the neighboring characters across word boundaries.
    ///
    /// # Arguments
    /// * `language` - The language for character type classification.
    /// * `words` - The segmented words, in order.
    ///
    /// # Returns
    /// One UPOS tag per word. Words the classifier cannot decide (an empty
    /// stage-2 model and no lexicon answer) receive [`Upos::X`].
    pub(crate) fn tag_words(&self, language: Language, words: &[String]) -> Vec<Upos> {
        let mut sent: Vec<char> = Vec::new();
        let mut type_ids: Vec<u8> = Vec::new();
        for word in words {
            for c in word.chars() {
                sent.push(c);
                type_ids.push(language.char_type_id(c));
            }
        }
        let radix = self.type_stride - 2;
        let n = self.n_classes;
        let mut scores = vec![0.0f64; n];
        let mut out = Vec::with_capacity(words.len());
        let mut start = 0usize;

        for word in words {
            let wlen = word.chars().count();
            if wlen == 0 {
                // Not producible by segment(); kept total for safety.
                out.push(Upos::X);
                continue;
            }
            let end = start + wlen;
            let entry = self.words.get(word.as_str());
            if let Some(e) = entry {
                if let Some(tag) = e.fixed {
                    out.push(tag);
                    start = end;
                    continue;
                }
            }
            if n == 0 {
                out.push(Upos::X);
                start = end;
                continue;
            }

            scores.iter_mut().for_each(|s| *s = 0.0);
            if let Some(e) = entry {
                for &(c, w) in e.ws_row.iter() {
                    scores[c as usize] += w;
                }
            }
            if self.wl_used {
                let row = &self.dense_wl[wlen.min(WL_CAP) * n..][..n];
                for (s, w) in scores.iter_mut().zip(row) {
                    *s += w;
                }
            }
            self.add_type(&mut scores, F_FT, type_ids[start] as usize);
            self.add_type(&mut scores, F_LT, type_ids[end - 1] as usize);
            if self.has[T_FC] {
                self.add_hash(&mut scores, hash_key(T_FC, sent[start] as u64));
            }
            if self.has[T_LC] {
                self.add_hash(&mut scores, hash_key(T_LC, sent[end - 1] as u64));
            }
            if self.has[T_TS] {
                let payload = ts_payload(&type_ids[start..end.min(start + TS_CAP)]);
                self.add_hash(&mut scores, hash_key(T_TS, payload));
            }
            // Context characters at distance k, as packed codes and dense
            // type indices (BOS/EOS sentinels beyond the sentence).
            let lc = |k: usize| if start >= k { sent[start - k] as u64 } else { BOS_CODE };
            let rc = |k: usize| sent.get(end + k - 1).map_or(EOS_CODE, |&c| c as u64);
            let lt = |k: usize| if start >= k { type_ids[start - k] as usize } else { radix };
            let rt = |k: usize| type_ids.get(end + k - 1).map_or(radix + 1, |&t| t as usize);
            for k in 1..=CONTEXT_WINDOW {
                if self.has[T_L1 + k - 1] {
                    self.add_hash(&mut scores, hash_key(T_L1 + k - 1, lc(k)));
                }
                if self.has[T_R1 + k - 1] {
                    self.add_hash(&mut scores, hash_key(T_R1 + k - 1, rc(k)));
                }
                self.add_type(&mut scores, F_CL1 + k - 1, lt(k));
                self.add_type(&mut scores, F_CR1 + k - 1, rt(k));
            }
            if self.has[T_LB] {
                self.add_hash(&mut scores, hash_key(T_LB, (lc(2) << 24) | lc(1)));
            }
            if self.has[T_RB] {
                self.add_hash(&mut scores, hash_key(T_RB, (rc(1) << 24) | rc(2)));
            }
            if wlen >= 2 {
                if self.has[T_P2] {
                    let payload = ((sent[start] as u64) << 24) | sent[start + 1] as u64;
                    self.add_hash(&mut scores, hash_key(T_P2, payload));
                }
                if self.has[T_S2] {
                    let payload = ((sent[end - 2] as u64) << 24) | sent[end - 1] as u64;
                    self.add_hash(&mut scores, hash_key(T_S2, payload));
                }
            }

            // Argmax with the perceptron's first-wins tie-break (lowest
            // class index), restricted to the candidates when the surface
            // has usable ones.
            let best = match entry {
                Some(e) if !e.candidates.is_empty() => {
                    let mut best = e.candidates[0] as usize;
                    let mut best_score = scores[best];
                    for &c in &e.candidates[1..] {
                        if scores[c as usize] > best_score {
                            best = c as usize;
                            best_score = scores[best];
                        }
                    }
                    best
                }
                _ => {
                    let mut best = 0usize;
                    let mut best_score = scores[0];
                    for (c, &s) in scores.iter().enumerate().skip(1) {
                        if s > best_score {
                            best = c;
                            best_score = s;
                        }
                    }
                    best
                }
            };
            out.push(self.classes[best]);
            start = end;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::adaboost::AdaBoost;
    use crate::segmenter::Segmenter;
    use crate::two_stage::TwoStageLearner;

    fn stage2(model: &str) -> AveragedPerceptron {
        let mut p = AveragedPerceptron::new();
        p.load_model_from_reader(model.as_bytes()).unwrap();
        p
    }

    fn lexicon(entries: &[(&str, &[(Upos, u32)])]) -> FxHashMap<String, Vec<(Upos, u32)>> {
        entries.iter().map(|(s, e)| (s.to_string(), e.to_vec())).collect()
    }

    const MODEL: &str = "2\nNOUN\nVERB\nL1:あ\tVERB\t1\nWS:x\tNOUN\t0.6\nWS:x\tVERB\t0.5";

    fn tag(model: &PackedTwoStageModel, words: &[&str]) -> Vec<Upos> {
        let words: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        model.tag_words(Language::Japanese, &words)
    }

    #[test]
    fn test_fixed_tags_skip_scoring() {
        let lex = lexicon(&[
            // Single candidate: always fixed.
            ("y", &[(Upos::NOUN, 1)]),
            // Dominant candidate: 100/101 >= 0.99.
            ("w", &[(Upos::VERB, 100), (Upos::NOUN, 1)]),
            // Non-dominant: 3/5 < 0.99, classifier decides.
            ("x", &[(Upos::NOUN, 3), (Upos::VERB, 2)]),
        ]);
        let model = PackedTwoStageModel::build(Language::Japanese, &stage2(MODEL), &lex, 0.99);
        // "y" and "w" are fixed; "x" is scored: WS gives NOUN 0.6 > VERB 0.5.
        assert_eq!(tag(&model, &["y", "w", "x"]), [Upos::NOUN, Upos::VERB, Upos::NOUN]);
    }

    #[test]
    fn test_context_feature_flips_masked_argmax() {
        let lex = lexicon(&[("x", &[(Upos::NOUN, 3), (Upos::VERB, 2)])]);
        let model = PackedTwoStageModel::build(Language::Japanese, &stage2(MODEL), &lex, 0.99);
        // Preceded by 'あ', the L1 context feature adds VERB +1:
        // VERB 1.5 > NOUN 0.6. ('あ' itself is unknown: all-zero scores,
        // first-wins tie-break picks class 0 = NOUN.)
        assert_eq!(tag(&model, &["あ", "x"]), [Upos::NOUN, Upos::VERB]);
        // Without that context the WS row decides: NOUN 0.6 > VERB 0.5.
        assert_eq!(tag(&model, &["x"]), [Upos::NOUN]);
    }

    #[test]
    fn test_lexicon_tags_unknown_to_classifier() {
        // Candidates outside the classifier's classes are unusable for
        // masking: with none left the word falls back to the full argmax,
        // but a *dominant* out-of-class tag is still assigned via `fixed`.
        let lex = lexicon(&[("z", &[(Upos::ADP, 5), (Upos::PART, 4)]), ("q", &[(Upos::SYM, 1)])]);
        let model = PackedTwoStageModel::build(Language::Japanese, &stage2(MODEL), &lex, 0.99);
        // "z": both candidates unknown to the classifier, not dominant ->
        // full argmax over all-zero scores -> class 0 (NOUN).
        // "q": single candidate -> fixed SYM even though the classifier
        // does not know it.
        assert_eq!(tag(&model, &["z", "q"]), [Upos::NOUN, Upos::SYM]);
    }

    #[test]
    fn test_empty_stage2_yields_x_for_unknown() {
        // A single-class degenerate model cannot be built (the perceptron
        // format requires classes), so exercise the n_classes == 0 guard
        // through an empty perceptron.
        let lex = lexicon(&[("y", &[(Upos::NOUN, 1)])]);
        let model =
            PackedTwoStageModel::build(Language::Japanese, &AveragedPerceptron::new(), &lex, 0.99);
        assert_eq!(tag(&model, &["y", "??"]), [Upos::NOUN, Upos::X]);
    }

    #[test]
    fn test_segmenter_two_stage_integration() {
        // The default (empty) stage-1 AdaBoost segments one char per word,
        // which makes the boundary behavior deterministic for this test.
        let lex = vec![("こ".to_string(), vec![(Upos::NOUN, 1)])];
        let learner = TwoStageLearner::from_parts(
            AdaBoost::default(),
            stage2("2\nNOUN\nVERB\nL1:こ\tVERB\t1"),
            lex,
            0.99,
        )
        .unwrap();
        let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);

        // segment() runs stage-1 only.
        assert_eq!(segmenter.segment("これ"), ["こ", "れ"]);
        // segment_with_pos(): "こ" is lexicon-fixed NOUN; "れ" is unknown
        // and the L1 context feature ('こ') pushes it to VERB.
        let tagged = segmenter.segment_with_pos("これ").unwrap();
        assert_eq!(tagged, [("こ".to_string(), Upos::NOUN), ("れ".to_string(), Upos::VERB)]);
        // Words always concatenate back to the input.
        let text: String = tagged.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(text, "これ");
        // Empty input stays empty.
        assert!(segmenter.segment_with_pos("").unwrap().is_empty());
        // The two-stage segmenter has no joint POS learner.
        assert!(segmenter.pos_learner().is_none());
    }
}
