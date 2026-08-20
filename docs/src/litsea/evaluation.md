# Evaluation

Held-out quality evaluation of segmentation and POS tagging
(`litsea::evaluation`). This is the library API behind the
[`litsea evaluate`](../litsea-cli/evaluate.md) subcommand. `evaluate_pos`
evaluates a segmenter built with
[`with_two_stage_learner`](segmenter.md#with_two_stage_learner)
(see [Two-Stage Tagging](../algorithm/two-stage-tagging.md)) through
`segment_with_pos`.

## Metrics Types

```rust
pub struct SegmentationMetrics {
    pub word_precision: f64,     // %
    pub word_recall: f64,        // %
    pub word_f1: f64,            // %
    pub boundary_precision: f64, // %
    pub boundary_recall: f64,    // %
    pub boundary_f1: f64,        // %
    pub sentences: usize,
    pub gold_words: usize,
    pub predicted_words: usize,
}

pub struct PosMetrics {
    pub segmentation: SegmentationMetrics,
    pub tagged_precision: f64, // %: span and tag both match
    pub tagged_recall: f64,    // %
    pub tagged_f1: f64,        // %
}
```

Both are re-exported at the crate root. Tokens are matched by exact
character-offset spans over the concatenation of the gold tokens;
pure-whitespace tokens are excluded from scoring (the Korean
space-preserving protocol; a no-op for languages written without spaces).

## Functions

### `evaluate_segmentation`

```rust
pub fn evaluate_segmentation<I, S>(segmenter: &Segmenter, gold: I) -> SegmentationMetrics
where
    I: IntoIterator<Item = Vec<S>>,
    S: Into<String>,
```

Segments the concatenation of each gold sentence's tokens with
[`Segmenter::segment`] and scores the result. Empty sentences are skipped.

### `evaluate_pos`

```rust
pub fn evaluate_pos<I, S>(segmenter: &Segmenter, gold: I) -> litsea::Result<PosMetrics>
where
    I: IntoIterator<Item = Vec<(S, Upos)>>,
    S: Into<String>,
```

Like `evaluate_segmentation` but drives [`Segmenter::segment_with_pos`]
and additionally scores tagged words. Returns
`LitseaError::PosLearnerNotSet` if the segmenter has neither a POS learner
nor a two-stage learner set.

### Gold-line parsers

```rust
pub fn parse_gold_line(line: &str, tsv: bool) -> Vec<String>
pub fn parse_gold_pos_line(line: &str) -> Vec<(String, Upos)>
```

`parse_gold_line` splits on spaces (or tabs with `tsv = true`, where a
token may be a literal space); `parse_gold_pos_line` splits each token at
its **last** `/` (the training pipeline's rule), defaulting to `Upos::X`
for missing or unparsable tags.

## Example

```rust
use litsea::adaboost::AdaBoost;
use litsea::evaluation::{evaluate_segmentation, parse_gold_line};
use litsea::language::Language;
use litsea::segmenter::Segmenter;

let mut learner = AdaBoost::new(0.01, 100);
learner.load_model_from_path(std::path::Path::new("./models/japanese.model"))?;
let segmenter = Segmenter::with_learner(Language::Japanese, learner);

let gold = std::fs::read_to_string("./resources/eval/japanese_gsd_test.txt")?;
let sentences = gold.lines().map(|l| parse_gold_line(l, false));
let metrics = evaluate_segmentation(&segmenter, sentences);
println!("word F1: {:.2}%", metrics.word_f1);
# Ok::<(), Box<dyn std::error::Error>>(())
```
