//! Held-out evaluation of segmentation (and POS tagging, joint or
//! two-stage) quality.
//!
//! Compares a [`Segmenter`]'s output against gold-standard token sequences
//! using character-offset spans: the gold text is the concatenation of the
//! gold tokens, so predicted and gold tokens can be matched exactly by
//! their `(start, end)` offsets. Pure-whitespace tokens are excluded from
//! scoring (the Korean space-preserving protocol of issue #152; a no-op
//! for languages written without spaces).

use std::collections::HashSet;

use crate::segmenter::Segmenter;
use crate::upos::Upos;

/// Held-out segmentation quality metrics over a gold corpus.
///
/// Word metrics score exact token span matches; boundary metrics score the
/// individual start-of-token decisions (excluding the sentence start).
/// Pure-whitespace tokens are excluded from both.
#[derive(Debug, Clone)]
pub struct SegmentationMetrics {
    /// Word precision in percentage (%)
    pub word_precision: f64,
    /// Word recall in percentage (%)
    pub word_recall: f64,
    /// Word F1 in percentage (%)
    pub word_f1: f64,
    /// Boundary precision in percentage (%)
    pub boundary_precision: f64,
    /// Boundary recall in percentage (%)
    pub boundary_recall: f64,
    /// Boundary F1 in percentage (%)
    pub boundary_f1: f64,
    /// Number of evaluated sentences
    pub sentences: usize,
    /// Number of gold (non-whitespace) words
    pub gold_words: usize,
    /// Number of predicted (non-whitespace) words
    pub predicted_words: usize,
}

/// Held-out segmentation + POS tagging quality metrics, for either the
/// joint or the two-stage architecture (both are evaluated the same way,
/// through [`Segmenter::segment_with_pos`]).
///
/// The segmentation part is identical to [`SegmentationMetrics`]; the
/// tagged-word metrics additionally require the predicted POS tag to match
/// the gold tag on top of the exact token span.
#[derive(Debug, Clone)]
pub struct PosMetrics {
    /// Segmentation quality of the joint output.
    pub segmentation: SegmentationMetrics,
    /// Tagged-word precision in percentage (%): span and tag both match.
    pub tagged_precision: f64,
    /// Tagged-word recall in percentage (%)
    pub tagged_recall: f64,
    /// Tagged-word F1 in percentage (%)
    pub tagged_f1: f64,
}

/// Percentage helper: `100 * a / b`, `0.0` when the denominator is zero.
fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

/// Harmonic mean of two percentages, `0.0` when both are zero.
fn f1(p: f64, r: f64) -> f64 {
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// `(start, end, is_whitespace)` character-offset spans of `tokens` laid
/// end to end.
fn spans(tokens: &[String]) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut pos = 0usize;
    for token in tokens {
        let len = token.chars().count();
        out.push((pos, pos + len, !token.is_empty() && token.chars().all(char::is_whitespace)));
        pos += len;
    }
    out
}

/// Accumulator shared by the segmentation and POS evaluations.
#[derive(Default)]
struct Counts {
    sentences: usize,
    word_tp: usize,
    word_pred: usize,
    word_gold: usize,
    boundary_tp: usize,
    boundary_pred: usize,
    boundary_gold: usize,
}

impl Counts {
    /// Scores one sentence: gold vs predicted token vectors over the same
    /// text. Returns the set of exactly matched non-whitespace gold spans
    /// so the POS evaluation can check tags on top.
    fn add_sentence(&mut self, gold: &[String], predicted: &[String]) -> HashSet<(usize, usize)> {
        self.sentences += 1;

        let gold_spans = spans(gold);
        let predicted_spans = spans(predicted);

        let gold_set: HashSet<(usize, usize)> =
            gold_spans.iter().filter(|s| !s.2).map(|s| (s.0, s.1)).collect();
        let predicted_set: HashSet<(usize, usize)> =
            predicted_spans.iter().filter(|s| !s.2).map(|s| (s.0, s.1)).collect();
        let matched: HashSet<(usize, usize)> =
            gold_set.intersection(&predicted_set).copied().collect();
        self.word_tp += matched.len();
        self.word_pred += predicted_set.len();
        self.word_gold += gold_set.len();

        let gold_bounds: HashSet<usize> =
            gold_spans.iter().filter(|s| !s.2 && s.0 != 0).map(|s| s.0).collect();
        let predicted_bounds: HashSet<usize> =
            predicted_spans.iter().filter(|s| !s.2 && s.0 != 0).map(|s| s.0).collect();
        self.boundary_tp += gold_bounds.intersection(&predicted_bounds).count();
        self.boundary_pred += predicted_bounds.len();
        self.boundary_gold += gold_bounds.len();

        matched
    }

    fn finish(&self) -> SegmentationMetrics {
        let wp = pct(self.word_tp, self.word_pred);
        let wr = pct(self.word_tp, self.word_gold);
        let bp = pct(self.boundary_tp, self.boundary_pred);
        let br = pct(self.boundary_tp, self.boundary_gold);
        SegmentationMetrics {
            word_precision: wp,
            word_recall: wr,
            word_f1: f1(wp, wr),
            boundary_precision: bp,
            boundary_recall: br,
            boundary_f1: f1(bp, br),
            sentences: self.sentences,
            gold_words: self.word_gold,
            predicted_words: self.word_pred,
        }
    }
}

/// Evaluates word segmentation quality against gold token sequences.
///
/// Each gold sentence is a token vector; the input text handed to
/// [`Segmenter::segment`] is the concatenation of its tokens (so a gold
/// token that is a literal space, as in the Korean space-preserving TSV
/// format, reconstructs the original spacing).
///
/// # Arguments
/// * `segmenter` - The segmenter (with an AdaBoost learner) to evaluate.
/// * `gold` - Gold sentences as token vectors; empty sentences are skipped.
///
/// # Returns
/// The held-out [`SegmentationMetrics`] over all non-empty sentences.
pub fn evaluate_segmentation<I, S>(segmenter: &Segmenter, gold: I) -> SegmentationMetrics
where
    I: IntoIterator<Item = Vec<S>>,
    S: Into<String>,
{
    let mut counts = Counts::default();
    for sentence in gold {
        let tokens: Vec<String> = sentence.into_iter().map(Into::into).collect();
        if tokens.is_empty() {
            continue;
        }
        let text: String = tokens.concat();
        let predicted = segmenter.segment(&text);
        counts.add_sentence(&tokens, &predicted);
    }
    counts.finish()
}

/// Evaluates segmentation + POS tagging quality against gold `(token, tag)`
/// sequences, for either the joint or the two-stage architecture.
///
/// Segmentation is scored exactly like [`evaluate_segmentation`]; the
/// tagged-word metrics additionally require the predicted [`Upos`] to
/// match the gold tag on exactly matched spans.
///
/// # Arguments
/// * `segmenter` - The segmenter to evaluate. Internally calls
///   [`Segmenter::segment_with_pos`], so a segmenter built with either an
///   Averaged Perceptron POS learner (joint) or a two-stage learner works.
/// * `gold` - Gold sentences as `(token, tag)` vectors; empty sentences are skipped.
///
/// # Returns
/// The held-out [`PosMetrics`] over all non-empty sentences.
///
/// # Errors
/// Returns [`crate::error::LitseaError::PosLearnerNotSet`] if the segmenter
/// has neither a POS learner nor a two-stage learner set.
pub fn evaluate_pos<I, S>(segmenter: &Segmenter, gold: I) -> crate::error::Result<PosMetrics>
where
    I: IntoIterator<Item = Vec<(S, Upos)>>,
    S: Into<String>,
{
    let mut counts = Counts::default();
    let (mut tagged_tp, mut tagged_pred, mut tagged_gold) = (0usize, 0usize, 0usize);

    for sentence in gold {
        let gold_tagged: Vec<(String, Upos)> =
            sentence.into_iter().map(|(w, t)| (w.into(), t)).collect();
        if gold_tagged.is_empty() {
            continue;
        }
        let tokens: Vec<String> = gold_tagged.iter().map(|(w, _)| w.clone()).collect();
        let text: String = tokens.concat();
        let predicted = segmenter.segment_with_pos(&text)?;
        let predicted_tokens: Vec<String> = predicted.iter().map(|(w, _)| w.clone()).collect();

        let matched = counts.add_sentence(&tokens, &predicted_tokens);

        // Tag lookup by span for both sides, whitespace tokens excluded.
        let gold_spans = spans(&tokens);
        let predicted_spans = spans(&predicted_tokens);
        tagged_gold += gold_spans.iter().filter(|s| !s.2).count();
        tagged_pred += predicted_spans.iter().filter(|s| !s.2).count();
        for (i, span) in predicted_spans.iter().enumerate() {
            if span.2 || !matched.contains(&(span.0, span.1)) {
                continue;
            }
            let gold_idx = gold_spans.iter().position(|g| (g.0, g.1) == (span.0, span.1));
            if let Some(g) = gold_idx {
                if gold_tagged[g].1 == predicted[i].1 {
                    tagged_tp += 1;
                }
            }
        }
    }

    let tp = pct(tagged_tp, tagged_pred);
    let tr = pct(tagged_tp, tagged_gold);
    Ok(PosMetrics {
        segmentation: counts.finish(),
        tagged_precision: tp,
        tagged_recall: tr,
        tagged_f1: f1(tp, tr),
    })
}

/// Parses one gold line in the given corpus format into a token vector.
///
/// * `space` format: tokens separated by single spaces (`"word word ..."`).
/// * `tsv` format: tokens separated by tabs; a token may be a literal
///   space `" "` (the space-preserving TSV corpus of issue #152).
///
/// Empty tokens are dropped in both formats.
///
/// # Arguments
/// * `line` - The gold corpus line.
/// * `tsv` - Whether the line is tab-separated (`true`) or space-separated.
///
/// # Returns
/// The token vector; empty for blank lines.
pub fn parse_gold_line(line: &str, tsv: bool) -> Vec<String> {
    let sep = if tsv { '\t' } else { ' ' };
    line.split(sep).filter(|t| !t.is_empty()).map(str::to_string).collect()
}

/// Parses one POS-tagged gold line (`"word/POS word/POS ..."`) into
/// `(token, tag)` pairs, splitting each token at its **last** `/` (the
/// same rule as the training pipeline); a token without a slash gets
/// [`Upos::X`], as does an unparsable tag.
///
/// # Arguments
/// * `line` - The POS-tagged gold corpus line.
///
/// # Returns
/// The `(token, tag)` vector; empty for blank lines.
pub fn parse_gold_pos_line(line: &str) -> Vec<(String, Upos)> {
    line.split(' ')
        .filter(|t| !t.is_empty())
        .map(|token| match token.rfind('/') {
            Some(idx) => (token[..idx].to_string(), token[idx + 1..].parse().unwrap_or(Upos::X)),
            None => (token.to_string(), Upos::X),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaboost::AdaBoost;
    use crate::language::Language;

    fn identity_segmenter() -> Segmenter {
        // An empty/default learner's bias evaluates to 0.0 (model = [0.0]),
        // and segment()'s decision rule is score >= 0.0, so it treats every
        // character as its own word — exactly what these metric-math tests
        // want, without needing a real trained learner.
        Segmenter::with_learner(Language::Japanese, AdaBoost::default())
    }

    #[test]
    fn test_counts_exact_match() {
        let mut counts = Counts::default();
        let gold = vec!["これ".to_string(), "は".to_string()];
        counts.add_sentence(&gold, &gold.clone());
        let m = counts.finish();
        assert_eq!(m.word_f1, 100.0);
        assert_eq!(m.boundary_f1, 100.0);
        assert_eq!(m.gold_words, 2);
        assert_eq!(m.predicted_words, 2);
    }

    #[test]
    fn test_counts_partial_match() {
        let mut counts = Counts::default();
        // gold: これ|は  predicted: こ|れは — no word matches, no boundary
        // matches (gold boundary at 2, predicted at 1).
        let gold = vec!["これ".to_string(), "は".to_string()];
        let predicted = vec!["こ".to_string(), "れは".to_string()];
        counts.add_sentence(&gold, &predicted);
        let m = counts.finish();
        assert_eq!(m.word_f1, 0.0);
        assert_eq!(m.boundary_f1, 0.0);
    }

    #[test]
    fn test_counts_whitespace_tokens_excluded() {
        let mut counts = Counts::default();
        // Korean-style spaced gold: the space token must not count as a
        // word, and boundaries adjacent to it must still be scored.
        let gold = vec!["나는".to_string(), " ".to_string(), "봄".to_string()];
        counts.add_sentence(&gold, &gold.clone());
        let m = counts.finish();
        assert_eq!(m.gold_words, 2);
        assert_eq!(m.word_f1, 100.0);
        assert_eq!(m.boundary_f1, 100.0);
    }

    #[test]
    fn test_evaluate_segmentation_smoke() {
        // With an untrained learner the output is degenerate but the
        // evaluation must still be well-formed (0..=100 metrics, counts).
        let segmenter = identity_segmenter();
        let gold = vec![vec!["これ", "は"], vec!["テスト", "です"]];
        let m = evaluate_segmentation(&segmenter, gold);
        assert_eq!(m.sentences, 2);
        assert_eq!(m.gold_words, 4);
        assert!((0.0..=100.0).contains(&m.word_f1));
    }

    #[test]
    fn test_parse_gold_line_space_and_tsv() {
        assert_eq!(parse_gold_line("これ は テスト", false), vec!["これ", "は", "テスト"]);
        // TSV keeps a literal space token; empty tokens are dropped.
        assert_eq!(parse_gold_line("나는\t \t봄\t\t.", true), vec!["나는", " ", "봄", "."]);
        assert!(parse_gold_line("", false).is_empty());
    }

    #[test]
    fn test_parse_gold_pos_line() {
        let parsed = parse_gold_pos_line("これ/PRON は/ADP //PUNCT plain");
        assert_eq!(
            parsed,
            vec![
                ("これ".to_string(), Upos::PRON),
                ("は".to_string(), Upos::ADP),
                ("/".to_string(), Upos::PUNCT),
                ("plain".to_string(), Upos::X),
            ]
        );
    }
}
