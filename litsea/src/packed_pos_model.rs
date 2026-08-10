//! Packed multiclass two-pass scoring model for the joint segmentation +
//! POS path (issue #143).
//!
//! [`PackedPosModel`] is the Averaged Perceptron counterpart of
//! [`crate::packed_model::PackedModel`]: the perceptron's string-keyed
//! per-class weight rows are compiled once (at model (re)load, not on the
//! hot path) into merged scatter tables, and `Segmenter::segment_with_pos`
//! scores each sentence in the same two passes as `segment()` — a static
//! pass that scatter-adds every tag-free feature into a per-position score
//! matrix, and a sequential pass that adds the 16 tag-dependent dense rows
//! and takes the argmax — without building a single feature string.
//!
//! Because perceptron updates touch only the gold/predicted class pair,
//! features average ~3 non-zero classes; the hash-table families therefore
//! store sparse `(class, weight)` rows, and the dense tag/type tables carry
//! a presence bitset so rows of absent features are never loaded. Skipped
//! zero additions are IEEE no-ops (scores start at `+0.0` and accumulate
//! sums of finite weights, which cannot round to `-0.0`), so sparsity does
//! not change any per-class sum; only the accumulation *order* differs from
//! the string-keyed reference, exactly as in `segment()`'s two-pass scorer
//! (#139). Output equality is pinned empirically by the exact-equality
//! differential tests against `segment_with_pos_reference`.
//!
//! The argmax loop mirrors `AveragedPerceptron::predict_idx_into` (first
//! strictly-greater class wins), so tie-breaking is identical.

use rustc_hash::FxHashMap;

use crate::language::Language;
use crate::packed_model::{
    BC_FIRST_ID, BW_IDS, TC_FIRST_ID, TEMPLATES, TYPE_ONLY_IDS, UW_IDS, WC_IDS, wc_key,
};
use crate::perceptron::AveragedPerceptron;
use crate::upos::SegmentLabel;

/// One entry of a merged sparse family row: `(family slot k, class index,
/// weight)`. Non-zero weights only, sorted by `(k, class)`.
pub(crate) type SlotClassWeight = (u8, u16, f64);

/// One entry of a per-key sparse class row: `(class index, weight)`.
/// Non-zero weights only, sorted by class.
pub(crate) type ClassWeight = (u16, f64);

/// A trained Averaged Perceptron compiled for two-pass scoring in
/// `segment_with_pos`.
///
/// Char-bearing templates live in merged-vector hash tables (one probe per
/// text position covers a whole family, as in
/// [`crate::packed_model::PackedModel`]) with sparse per-class rows;
/// tag/type-only templates live in direct-indexed dense tables with
/// `n_classes`-wide rows plus a presence bitset. Class labels are parsed
/// once at build time.
#[derive(Debug)]
pub(crate) struct PackedPosModel {
    /// Number of classes (the width of every dense row and of each
    /// per-position row in the score matrix). Zero when the perceptron has
    /// no classes.
    pub(crate) n_classes: usize,
    /// Class index -> parsed label, parallel to the perceptron's sorted
    /// class list. Unparseable class names map to [`SegmentLabel::O`],
    /// matching the `predict`-then-`parse().unwrap_or(O)` fallback of the
    /// string-keyed path.
    labels: Vec<SegmentLabel>,
    /// Char code -> merged sparse `UW1..UW6` row ([`SlotClassWeight`]
    /// entries). One probe per text position covers all six unigram-word
    /// templates (scatter-added to the six neighboring decision positions).
    pub(crate) uw: FxHashMap<u32, Box<[SlotClassWeight]>>,
    /// `(a << 24) | b` (adjacent char codes) -> merged sparse `BW1..BW3`
    /// row, same entry layout as `uw`.
    pub(crate) bw: FxHashMap<u64, Box<[SlotClassWeight]>>,
    /// [`wc_key`]-keyed sparse [`ClassWeight`] rows of the `WC*` templates
    /// (Japanese/Chinese only), gathered per position. (A merged per-char
    /// layout probed once per position was tried and measured slower:
    /// frequent chars accumulate rows of dozens of (family, type) entries,
    /// and scanning them costs more than four exact-key probes.)
    pub(crate) wc: FxHashMap<u64, Box<[ClassWeight]>>,
    /// Scatter twin of the `UC*` dense tables: type id `v`, family slot `k`
    /// -> row at `((v * 6) + k) * n_classes`.
    pub(crate) uc: Vec<f64>,
    /// Scatter twin of the `BC*` dense tables: type-id pair
    /// (`t1 * T + t2`), slot `k` -> row at `((pair * 3) + k) * n_classes`.
    pub(crate) bc: Vec<f64>,
    /// Scatter twin of the `TC*` dense tables: type-id triple, slot `k` ->
    /// row at `((triple * 4) + k) * n_classes`.
    pub(crate) tc: Vec<f64>,
    /// Dense weight tables indexed by template id, one per
    /// [`crate::packed_model::Template::is_dense`] template (empty for the
    /// char-bearing templates). `dense[tid][idx * n_classes..][..n_classes]`
    /// is the per-class row of the feature with mixed-radix index `idx`
    /// ([`crate::packed_model::Template::dense_index`]); rows of absent
    /// features stay all-zero. The canonical store for the tag-dependent
    /// sequential pass; the `uc`/`bc`/`tc` scatter vectors above are
    /// derived views of ids 14..27.
    dense: Vec<Vec<f64>>,
    /// Presence bitset per dense template: bit `idx` is set iff a model
    /// feature landed on `dense[tid]` row `idx`. The sequential pass skips
    /// absent (all-zero) rows without touching the weight table, mirroring
    /// the reference path's hash-miss behavior.
    present: Vec<Vec<u64>>,
}

impl PackedPosModel {
    /// Compiles the perceptron's string-keyed per-class weights into the
    /// two-pass scoring tables for `language`. Called once per model
    /// (re)load, not on the hot path.
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

        let mut uw_rows: FxHashMap<u32, Vec<SlotClassWeight>> = FxHashMap::default();
        let mut bw_rows: FxHashMap<u64, Vec<SlotClassWeight>> = FxHashMap::default();
        let mut wc_rows: FxHashMap<u64, Vec<ClassWeight>> = FxHashMap::default();
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
        let mut present: Vec<Vec<u64>> = TEMPLATES
            .iter()
            .map(|t| {
                if t.is_dense() {
                    vec![0u64; t.dense_size(type_radix).div_ceil(64)]
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
                let sparse = || {
                    class_weights
                        .iter()
                        .enumerate()
                        .filter(|(_, w)| **w != 0.0)
                        .map(|(c, &w)| (c as u16, w))
                };
                if template.is_dense() {
                    let idx = template.dense_index_from_key(key, type_radix);
                    dense[tid][idx * n_classes..][..n_classes].copy_from_slice(class_weights);
                    present[tid][idx >> 6] |= 1u64 << (idx & 63);
                } else if UW_IDS.contains(&tid) {
                    // Single Chr slot: the char code is the low 24 bits.
                    let code = (key & 0xFF_FFFF) as u32;
                    let k = (tid - UW_IDS.start) as u8;
                    uw_rows.entry(code).or_default().extend(sparse().map(|(c, w)| (k, c, w)));
                } else if BW_IDS.contains(&tid) {
                    // Two Chr slots: (a << 24) | b in the low 48 bits, the
                    // same layout the scatter pass builds from adjacent
                    // char codes.
                    let k = (tid - BW_IDS.start) as u8;
                    bw_rows
                        .entry(key & 0xFFFF_FFFF_FFFF)
                        .or_default()
                        .extend(sparse().map(|(c, w)| (k, c, w)));
                } else {
                    let (chr, typ) = template.decode_chr_typ(key);
                    wc_rows
                        .entry(wc_key(tid - WC_IDS.start, chr, typ))
                        .or_default()
                        .extend(sparse());
                }
            }
        }

        // Freeze the merged rows: deterministic (slot, class) order, boxed
        // to drop the spare Vec capacity. Rows are only created for features
        // with at least one non-zero weight, but an all-zero weight row
        // (possible on a live-trained perceptron) can leave an empty Vec
        // behind; dropping it entirely is the same exact no-op.
        let freeze3 = |rows: FxHashMap<u32, Vec<SlotClassWeight>>| {
            rows.into_iter()
                .filter(|(_, row)| !row.is_empty())
                .map(|(key, mut row)| {
                    row.sort_unstable_by_key(|&(k, c, _)| (k, c));
                    (key, row.into_boxed_slice())
                })
                .collect::<FxHashMap<_, _>>()
        };
        let uw = freeze3(uw_rows);
        let bw = bw_rows
            .into_iter()
            .filter(|(_, row)| !row.is_empty())
            .map(|(key, mut row)| {
                row.sort_unstable_by_key(|&(k, c, _)| (k, c));
                (key, row.into_boxed_slice())
            })
            .collect::<FxHashMap<_, _>>();
        let wc = wc_rows
            .into_iter()
            .filter(|(_, row)| !row.is_empty())
            .map(|(key, mut row)| {
                row.sort_unstable_by_key(|&(c, _)| c);
                (key, row.into_boxed_slice())
            })
            .collect::<FxHashMap<_, _>>();

        // Derive the scatter twins of the type-only dense tables (ids
        // 14..27), exactly as PackedModel::build does for the AdaBoost path:
        // family slot k of value v holds dense[first_id + k]'s row v.
        let t = type_radix;
        let copy_twin = |first_id: usize, slots: usize, values: usize, dense: &[Vec<f64>]| {
            let mut twin = vec![0.0f64; values * slots * n_classes];
            for v in 0..values {
                for k in 0..slots {
                    twin[(v * slots + k) * n_classes..][..n_classes]
                        .copy_from_slice(&dense[first_id + k][v * n_classes..][..n_classes]);
                }
            }
            twin
        };
        let uc = copy_twin(TYPE_ONLY_IDS.start, 6, t, &dense);
        let bc = copy_twin(BC_FIRST_ID, 3, t * t, &dense);
        let tc = copy_twin(TC_FIRST_ID, 4, t * t * t, &dense);

        PackedPosModel {
            n_classes,
            labels,
            uw,
            bw,
            wc,
            uc,
            bc,
            tc,
            dense,
            present,
        }
    }

    /// Sequential-pass prediction at one position: copies the static score
    /// row, adds the 16 tag-dependent dense rows (skipping absent rows via
    /// the presence bitset), and returns the argmax class index (first
    /// strictly-greater class wins ties, mirroring
    /// `AveragedPerceptron::predict_idx_into`).
    ///
    /// The 16 hard-coded mixed-radix indices are the same expressions as
    /// `segment()`'s sequential pass, pinned against
    /// [`crate::packed_model::Template::dense_index`] by
    /// `test_sequential_pass_indices_match_dense_index`.
    ///
    /// # Arguments
    /// * `static_row` - The position's row of the static score matrix
    ///   (length `n_classes`).
    /// * `p` - Boundary-tag ids `(p1, p2, p3)` of positions `i-3..i-1`.
    /// * `c` - Type ids `(c1, c2, c3, c4)` of positions `i-3..i`.
    /// * `type_radix` - The language's type-code count.
    /// * `scores` - Scratch row of length `n_classes`, overwritten here.
    ///
    /// # Returns
    /// The argmax class index. Must not be called when
    /// [`n_classes`](Self::n_classes) is zero.
    pub(crate) fn predict_seq(
        &self,
        static_row: &[f64],
        p: (usize, usize, usize),
        c: (usize, usize, usize, usize),
        type_radix: usize,
        scores: &mut [f64],
    ) -> usize {
        let cn = self.n_classes;
        scores.copy_from_slice(static_row);
        let (p1, p2, p3) = p;
        let (c1, c2, c3, c4) = c;
        let t = type_radix;
        let indices: [(usize, usize); 16] = [
            (0, p1),
            (1, p2),
            (2, p3),
            (3, p1 * 3 + p2),
            (4, p2 * 3 + p3),
            (27, p1 * t + c1),
            (28, p2 * t + c2),
            (29, p3 * t + c3),
            (30, (p2 * t + c2) * t + c3),
            (31, (p2 * t + c3) * t + c4),
            (32, (p3 * t + c2) * t + c3),
            (33, (p3 * t + c3) * t + c4),
            (34, ((p2 * t + c1) * t + c2) * t + c3),
            (35, ((p2 * t + c2) * t + c3) * t + c4),
            (36, ((p3 * t + c1) * t + c2) * t + c3),
            (37, ((p3 * t + c2) * t + c3) * t + c4),
        ];
        for (tid, idx) in indices {
            if self.present[tid][idx >> 6] & (1u64 << (idx & 63)) != 0 {
                let row = &self.dense[tid][idx * cn..][..cn];
                for (s, w) in scores.iter_mut().zip(row) {
                    *s += *w;
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
        best
    }

    /// Returns the parsed label of class index `idx` (as produced by
    /// [`predict_seq`](Self::predict_seq)).
    pub(crate) fn label(&self, idx: usize) -> &SegmentLabel {
        &self.labels[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::packed_model::SENTINEL_BASE;
    use crate::upos::Upos;

    fn perceptron_from(model: &str) -> AveragedPerceptron {
        let mut p = AveragedPerceptron::new();
        p.load_model_from_reader(model.as_bytes()).unwrap();
        p
    }

    #[test]
    fn test_build_merged_sparse_rows() {
        // Classes sorted: B-NOUN (0), B-VERB (1), O (2). Family slots and
        // key layouts of the merged tables, including a sentinel char and
        // both WC slot orders (WC2 renders type first).
        let model = "3\nB-NOUN\nB-VERB\nO\n\
                     UW1:B2\tB-NOUN\t0.1\nUW4:い\tB-VERB\t0.2\nUW4:い\tO\t-0.2\n\
                     BW1:B1あ\tO\t0.3\nBW3:あい\tB-NOUN\t0.4\n\
                     WC2:Iい\tB-VERB\t0.5\nWC4:いI\tO\t0.6\nZZZ:x\tO\t9.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));

        assert_eq!(packed.n_classes, 3);
        let b1 = SENTINEL_BASE + 2;
        let b2 = SENTINEL_BASE + 1;
        // UW1 -> slot 0, UW4 -> slot 3; entries sorted by (slot, class).
        assert_eq!(packed.uw.get(&b2).map(|r| r.as_ref()), Some(&[(0u8, 0u16, 0.1)][..]));
        assert_eq!(
            packed.uw.get(&u32::from('い')).map(|r| r.as_ref()),
            Some(&[(3u8, 1u16, 0.2), (3u8, 2u16, -0.2)][..])
        );
        let bw1_key = (u64::from(b1) << 24) | u64::from('あ');
        let bw3_key = (u64::from('あ') << 24) | u64::from('い');
        assert_eq!(packed.bw.get(&bw1_key).map(|r| r.as_ref()), Some(&[(0u8, 2u16, 0.3)][..]));
        assert_eq!(packed.bw.get(&bw3_key).map(|r| r.as_ref()), Some(&[(2u8, 0u16, 0.4)][..]));
        // Japanese type id I = 6; WC family indices are 1 (WC2) and 3 (WC4).
        assert_eq!(
            packed.wc.get(&wc_key(1, u32::from('い'), 6)).map(|r| r.as_ref()),
            Some(&[(1u16, 0.5)][..])
        );
        assert_eq!(
            packed.wc.get(&wc_key(3, u32::from('い'), 6)).map(|r| r.as_ref()),
            Some(&[(2u16, 0.6)][..])
        );
        // ZZZ parses for no template; nothing else leaked in.
        assert_eq!(packed.uw.len(), 2);
        assert_eq!(packed.bw.len(), 2);
        assert_eq!(packed.wc.len(), 2);
    }

    #[test]
    fn test_build_dense_rows_and_presence() {
        // UC1 (id 14) with Japanese I = 6: dense row [6*C..6*C+C] holds the
        // full class row and the presence bit is set; untouched rows stay
        // absent.
        let model = "3\nB-NOUN\nB-VERB\nO\nUC1:I\tB-VERB\t1.5\nUC1:I\tO\t-0.5\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));

        assert_eq!(&packed.dense[14][6 * 3..6 * 3 + 3], &[0.0, 1.5, -0.5]);
        assert_ne!(packed.present[14][0] & (1 << 6), 0);
        // Only index 6 is present in UC1's table.
        assert_eq!(packed.present[14][0].count_ones(), 1);
        // The scatter twin mirrors the dense row: UC1 is family slot 0 of
        // type value 6.
        assert_eq!(&packed.uc[(6 * 6) * 3..(6 * 6) * 3 + 3], &[0.0, 1.5, -0.5]);
    }

    #[test]
    fn test_build_labels_fallback_to_o() {
        // Class names that are not valid segment labels fall back to O,
        // matching the string path's parse().unwrap_or(O).
        let model = "2\nB-NOUN\nWEIRD\nUW4:い\tB-NOUN\t1.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        assert_eq!(packed.labels, vec![SegmentLabel::B(Upos::NOUN), SegmentLabel::O]);
        assert_eq!(packed.label(0), &SegmentLabel::B(Upos::NOUN));
        assert_eq!(packed.label(1), &SegmentLabel::O);
    }

    #[test]
    fn test_predict_seq_skips_absent_rows_and_tie_breaks() {
        // No tag-dependent feature is present: predict_seq must return the
        // argmax of the static row alone, with first-class tie-breaking.
        let model = "2\nB-NOUN\nO\nUW4:い\tO\t1.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        let mut scores = vec![0.0; 2];
        // Tie: both classes at 0.0 -> first class (B-NOUN) wins.
        assert_eq!(packed.predict_seq(&[0.0, 0.0], (0, 0, 0), (0, 0, 0, 0), 8, &mut scores), 0);
        // Static row dominates when nothing else fires.
        assert_eq!(packed.predict_seq(&[0.0, 2.5], (0, 0, 0), (0, 0, 0, 0), 8, &mut scores), 1);
        assert_eq!(scores, vec![0.0, 2.5]);
    }

    #[test]
    fn test_predict_seq_adds_present_tag_rows() {
        // UP1 (id 0) reads p1; with p1 = B (tag id 1) its row must be added
        // on top of the static row.
        let model = "2\nB-NOUN\nO\nUP1:B\tB-NOUN\t3.0\n";
        let packed = PackedPosModel::build(Language::Japanese, &perceptron_from(model));
        let mut scores = vec![0.0; 2];
        let idx = packed.predict_seq(&[0.0, 1.0], (1, 0, 0), (0, 0, 0, 0), 8, &mut scores);
        assert_eq!(scores, vec![3.0, 1.0]);
        assert_eq!(idx, 0);
        // With p1 = U (tag id 0) the UP1:B row must not fire.
        let idx = packed.predict_seq(&[0.0, 1.0], (0, 0, 0), (0, 0, 0, 0), 8, &mut scores);
        assert_eq!(scores, vec![0.0, 1.0]);
        assert_eq!(idx, 1);
    }
}
