//! Packed multiclass scoring model for the joint segmentation + POS path
//! (issue #143).
//!
//! [`PackedPosModel`] is the Averaged Perceptron counterpart of
//! [`crate::packed_model::PackedModel`]: the perceptron's string-keyed
//! per-class weight rows are compiled once (at model (re)load, not on the
//! hot path) into packed-integer-keyed tables, and
//! `Segmenter::segment_with_pos` scores each character position against
//! them without building a single feature string.
//!
//! Scoring walks the templates in emission order, so the per-class addition
//! sequence is bit-identical to the string-keyed reference:
//!
//! - The 29 tag/type-only templates load an `n_classes`-wide row from a
//!   direct-indexed dense table ([`Template::dense_index`]). Rows of absent
//!   features stay all-zero, and adding `0.0` to a score is an exact no-op
//!   (scores never become `-0.0`: they start at `+0.0` and accumulate sums
//!   of finite weights, which cannot round to `-0.0`).
//! - The char-bearing templates (`UW*`/`BW*`/`WC*`) probe one hash map of
//!   [`Template::pack`] keys and add only the non-zero class weights in
//!   ascending class order — the skipped additions are exact zeros and the
//!   surviving ones keep their relative order, so each per-class sum is
//!   unchanged bit for bit.
//!
//! The argmax loop mirrors `AveragedPerceptron::predict_idx_into` (first
//! strictly-greater class wins), so tie-breaking is identical as well.

use rustc_hash::FxHashMap;

use crate::language::Language;
use crate::packed_model::{TEMPLATES, templates_for};
use crate::perceptron::AveragedPerceptron;
use crate::upos::SegmentLabel;

/// A trained Averaged Perceptron compiled for packed-key scoring in
/// `segment_with_pos`.
///
/// Char-bearing templates live in one packed-key hash map with sparse
/// per-class rows (perceptron updates touch only the gold/predicted class
/// pair, so most features carry just 2-4 non-zero classes); tag/type-only
/// templates live in direct-indexed dense tables with `n_classes`-wide
/// rows. Class labels are parsed once at build time.
#[derive(Debug)]
pub(crate) struct PackedPosModel {
    /// Number of classes (the width of every dense row and the length of
    /// the score buffer). Zero when the perceptron has no classes.
    n_classes: usize,
    /// Class index -> parsed label, parallel to the perceptron's sorted
    /// class list. Unparseable class names map to [`SegmentLabel::O`],
    /// matching the `predict`-then-`parse().unwrap_or(O)` fallback of the
    /// string-keyed path.
    labels: Vec<SegmentLabel>,
    /// Packed key ([`crate::packed_model::Template::pack`]) -> sparse
    /// `(class index, weight)` row for the char-bearing templates
    /// (`UW*`/`BW*`/`WC*`), non-zero entries only, in ascending class order.
    map: FxHashMap<u64, Box<[(u16, f64)]>>,
    /// Dense weight tables indexed by template id, one per
    /// [`crate::packed_model::Template::is_dense`] template (empty for the
    /// char-bearing templates). `dense[tid][idx * n_classes..][..n_classes]`
    /// is the per-class row of the feature with mixed-radix index `idx`;
    /// rows of absent features stay all-zero.
    dense: Vec<Vec<f64>>,
}

impl PackedPosModel {
    /// Compiles the perceptron's string-keyed per-class weights into packed
    /// scoring tables for `language`. Called once per model (re)load, not
    /// on the hot path.
    ///
    /// # Arguments
    /// * `language` - The language whose templates and type codes to compile
    ///   for.
    /// * `learner` - The perceptron whose feature weights to compile.
    ///
    /// # Returns
    /// A `PackedPosModel` mirroring every feature of `learner` that the
    /// attribute writer can generate for `language` (features no template
    /// can render are unreachable at inference and are skipped, exactly as
    /// in [`crate::packed_model::PackedModel::build`]).
    pub(crate) fn build(language: Language, learner: &AveragedPerceptron) -> Self {
        let type_radix = language.type_codes().len();
        let class_names = learner.class_names();
        let n_classes = class_names.len();
        let labels: Vec<SegmentLabel> =
            class_names.iter().map(|c| c.parse().unwrap_or(SegmentLabel::O)).collect();

        let mut map: FxHashMap<u64, Box<[(u16, f64)]>> = FxHashMap::default();
        let mut dense: Vec<Vec<f64>> = TEMPLATES
            .iter()
            .map(|t| {
                if t.is_dense() {
                    vec![0.0; t.dense_size(type_radix) * n_classes]
                } else {
                    Vec::new()
                }
            })
            .collect();

        let mut keys = Vec::new();
        for (feature, class_weights) in learner.feature_class_weights() {
            keys.clear();
            crate::packed_model::parse_feature_keys(language, feature, &mut keys);
            for &key in &keys {
                let tid = (key >> 56) as usize;
                let template = &TEMPLATES[tid];
                if template.is_dense() {
                    let idx = template.dense_index_from_key(key, type_radix);
                    dense[tid][idx * n_classes..][..n_classes].copy_from_slice(class_weights);
                } else {
                    let row: Box<[(u16, f64)]> = class_weights
                        .iter()
                        .enumerate()
                        .filter(|(_, w)| **w != 0.0)
                        .map(|(c, &w)| (c as u16, w))
                        .collect();
                    if !row.is_empty() {
                        map.insert(key, row);
                    }
                }
            }
        }

        PackedPosModel {
            n_classes,
            labels,
            map,
            dense,
        }
    }

    /// Scores position `i` against every template and returns the index of
    /// the highest-scoring class, reusing `scores` as a scratch buffer.
    /// Returns `None` when no classes are registered (mirroring
    /// `AveragedPerceptron::predict_idx_into`).
    ///
    /// # Arguments
    /// * `language` - The language whose templates to score with (must match
    ///   the language this model was built for).
    /// * `i` - The character position (same convention as the attribute
    ///   writer: valid range is `[3, chars.len() - 3)`).
    /// * `tags` - Boundary-tag ids per already-decided position
    ///   (`TAG_U`/`TAG_B`/`TAG_O`).
    /// * `char_codes` - Char codes per context position (code points;
    ///   sentinels are `SENTINEL_BASE + k`).
    /// * `type_ids` - Type ids per context position
    ///   ([`Language::char_type_id`]).
    /// * `scores` - Scratch buffer, cleared and resized here.
    ///
    /// # Returns
    /// The argmax class index (first strictly-greater class wins ties), or
    /// `None` if the model has no classes.
    pub(crate) fn predict_idx(
        &self,
        language: Language,
        i: usize,
        tags: &[u8],
        char_codes: &[u32],
        type_ids: &[u8],
        scores: &mut Vec<f64>,
    ) -> Option<usize> {
        if self.n_classes == 0 {
            return None;
        }
        scores.clear();
        scores.resize(self.n_classes, 0.0);
        let type_radix = language.type_codes().len();
        for template in templates_for(language) {
            if template.is_dense() {
                let idx = template.dense_index(i, tags, type_ids, type_radix);
                let row =
                    &self.dense[template.id as usize][idx * self.n_classes..][..self.n_classes];
                for (s, w) in scores.iter_mut().zip(row) {
                    *s += *w;
                }
            } else {
                let key = template.pack(i, tags, char_codes, type_ids);
                if let Some(row) = self.map.get(&key) {
                    for &(c, w) in row.iter() {
                        scores[c as usize] += w;
                    }
                }
            }
        }
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, s) in scores.iter().enumerate() {
            if *s > best_score {
                best_score = *s;
                best = idx;
            }
        }
        Some(best)
    }

    /// Returns the parsed label of class index `idx` (as produced by
    /// [`predict_idx`](Self::predict_idx)).
    pub(crate) fn label(&self, idx: usize) -> &SegmentLabel {
        &self.labels[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::language::OTHER_TYPE_ID;
    use crate::packed_model::{SENTINEL_BASE, TAG_U};
    use crate::upos::Upos;

    fn perceptron_from(model: &str) -> AveragedPerceptron {
        let mut p = AveragedPerceptron::new();
        p.load_model_from_reader(model.as_bytes()).unwrap();
        p
    }

    /// Context arrays for the single-character sentence "あ" (three head
    /// sentinels, the character, three tail sentinels), predicting at i=3.
    fn single_char_context(c: char, type_id: u8) -> ([u8; 4], [u32; 7], [u8; 7]) {
        let tags = [TAG_U; 4];
        let char_codes = [
            SENTINEL_BASE,
            SENTINEL_BASE + 1,
            SENTINEL_BASE + 2,
            u32::from(c),
            SENTINEL_BASE + 3,
            SENTINEL_BASE + 4,
            SENTINEL_BASE + 5,
        ];
        let type_ids = [
            OTHER_TYPE_ID,
            OTHER_TYPE_ID,
            OTHER_TYPE_ID,
            type_id,
            OTHER_TYPE_ID,
            OTHER_TYPE_ID,
            OTHER_TYPE_ID,
        ];
        (tags, char_codes, type_ids)
    }

    #[test]
    fn test_build_sparse_and_dense_rows() {
        // Classes sorted: B-NOUN (0), B-VERB (1), O (2). UW4 (id 8) is
        // char-bearing -> sparse row; UC1 (id 14) is dense (Japanese I = 6);
        // ZZZ parses for no template and is skipped.
        let model = "3\nB-NOUN\nB-VERB\nO\n\
                     UW4:い\tB-NOUN\t0.5\nUW4:い\tO\t-0.25\nUC1:I\tB-VERB\t1.5\nZZZ:x\tO\t2.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));

        assert_eq!(packed.n_classes, 3);
        assert_eq!(
            packed.labels,
            vec![SegmentLabel::B(Upos::NOUN), SegmentLabel::B(Upos::VERB), SegmentLabel::O]
        );
        let uw4_key = (8u64 << 56) | u64::from('い');
        assert_eq!(
            packed.map.get(&uw4_key).map(|r| r.as_ref()),
            Some(&[(0u16, 0.5), (2u16, -0.25)][..])
        );
        assert_eq!(packed.map.len(), 1);
        assert_eq!(&packed.dense[14][6 * 3..6 * 3 + 3], &[0.0, 1.5, 0.0]);
    }

    #[test]
    fn test_build_labels_fallback_to_o() {
        // Class names that are not valid segment labels fall back to O,
        // matching the string path's parse().unwrap_or(O).
        let model = "2\nB-NOUN\nWEIRD\nUW4:い\tB-NOUN\t1.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        assert_eq!(packed.labels, vec![SegmentLabel::B(Upos::NOUN), SegmentLabel::O]);
    }

    #[test]
    fn test_predict_idx_matches_feature_hit() {
        let model = "2\nB-NOUN\nO\nUW4:あ\tO\t1.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        let (tags, char_codes, type_ids) = single_char_context('あ', 6);
        let mut scores = Vec::new();
        let idx = packed
            .predict_idx(Language::Japanese, 3, &tags, &char_codes, &type_ids, &mut scores)
            .unwrap();
        assert_eq!(packed.label(idx), &SegmentLabel::O);
        assert_eq!(scores, vec![0.0, 1.0]);
    }

    #[test]
    fn test_predict_idx_tie_breaks_to_first_class() {
        // No feature of the model fires on this input: all scores stay 0.0
        // and the first class must win, exactly like predict_idx_into.
        let model = "2\nB-NOUN\nO\nUW4:い\tO\t1.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        let (tags, char_codes, type_ids) = single_char_context('あ', 6);
        let mut scores = Vec::new();
        let idx = packed
            .predict_idx(Language::Japanese, 3, &tags, &char_codes, &type_ids, &mut scores)
            .unwrap();
        assert_eq!(idx, 0);
        assert_eq!(packed.label(idx), &SegmentLabel::B(Upos::NOUN));
    }

    #[test]
    fn test_predict_idx_empty_classes() {
        let packed = PackedPosModel::build(Language::Japanese, &AveragedPerceptron::new());
        let (tags, char_codes, type_ids) = single_char_context('あ', 6);
        let mut scores = Vec::new();
        assert_eq!(
            packed.predict_idx(Language::Japanese, 3, &tags, &char_codes, &type_ids, &mut scores),
            None
        );
    }
}
