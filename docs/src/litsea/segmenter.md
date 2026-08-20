# Segmenter

The `Segmenter` struct is the primary interface for word segmentation.

## Definition

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: two_stage: Option<PackedTwoStageModel> (compiled stage-2 model)
    // internal: packed cache (see below)
}
```

The fields are private; use the accessor methods `language()`, `learner()`, and `learner_mut()` to reach them.

Besides these, the struct also holds `packed`: a lazily-rebuilt cache of
the learner's weights compiled into the integer-indexed tables `segment()`
scores against
(see [Prediction Pipeline](../algorithm/prediction-pipeline.md#the-compiled-scoring-tables)).
It is internal implementation detail with no accessor of its own,
invalidated automatically whenever the learner is mutated.
`two_stage` holds the compiled stage-2 tagging model set by
[`with_two_stage_learner`](#with_two_stage_learner) (see below); it is
`None` unless the segmenter was built from a two-stage model. Unlike the
cache it is not derived from a retained learner — the raw stage-2 parts
are dropped after compilation and there is no mutation path for it.

## Constructors

### `Segmenter::new`

```rust
pub fn new(language: Language) -> Self
```

Creates a segmenter with a default (untrained) `AdaBoost` learner — suitable
for training or feature extraction. Until a model is loaded or training data
is added, `segment` returns one word per character. No two-stage model is
set; `segment_with_pos` returns `Err(LitseaError::PosLearnerNotSet)` — use
[`with_two_stage_learner`](#with_two_stage_learner) for segmentation + POS
tagging.

### `Segmenter::with_learner`

```rust
pub fn with_learner(language: Language, learner: AdaBoost) -> Self
```

Creates a segmenter with the given learner, typically one that has loaded a
pre-trained model.

```rust
use litsea::language::Language;
use litsea::segmenter::Segmenter;

// With a pre-trained model
let segmenter = Segmenter::with_learner(Language::Japanese, learner);

// Without a model (for training or feature extraction)
let segmenter = Segmenter::new(Language::Japanese);
```

## Methods

### `segment`

```rust
pub fn segment(&self, sentence: &str) -> Vec<String>
```

Segments a sentence into words. Returns an empty vector for empty input.

```rust
let tokens = segmenter.segment("これはテストです。");
// ["これ", "は", "テスト", "です", "。"]
```

Internally this is a thin wrapper over
[`segment_into`](#segment_into--segmentbuffer) with a fresh buffer per
call, materializing each range as an owned `String` — there is a single
scoring implementation.

### `segment_into` / `SegmentBuffer`

```rust
pub struct SegmentBuffer { /* internal scratch + output storage */ }

impl SegmentBuffer {
    pub fn new() -> Self
}

impl Segmenter {
    pub fn segment_into<'b>(
        &self,
        sentence: &str,
        buf: &'b mut SegmentBuffer,
    ) -> &'b [(usize, usize)]
}
```

The allocation-free variant of `segment` (issue #184). Each returned
`(start, end)` pair is a byte range into `sentence`
(`&sentence[start..end]` is the token), in order, tiling the sentence
exactly. The buffer owns every per-call allocation (context arrays, score
buffer, tag scratch, output ranges); reusing one buffer across a batch of
sentences reaches a steady state where segmentation allocates nothing.
Empty input yields an empty slice.

At the published throughputs this matters: `segment` allocates one
`String` per token (millions per second in batch workloads) plus per-call
scratch, which measured as roughly a quarter of the batch profile. The
buffer holds plain data (no borrows), so it can be reused across
sentences, models, and languages; for parallel processing use one buffer
per thread.

```rust
use litsea::segmenter::{SegmentBuffer, Segmenter};

let mut buf = SegmentBuffer::new();
for line in lines {
    for &(start, end) in segmenter.segment_into(line, &mut buf) {
        let token: &str = &line[start..end];
        // write/inspect token without allocating
    }
}
```

### `char_type`

```rust
pub fn char_type(&self, c: char) -> &'static str
```

Classifies a character into its language-specific type code (delegates to `Language::char_type`).

```rust
let segmenter = Segmenter::new(Language::Japanese);
assert_eq!(segmenter.char_type('あ'), "I");  // Hiragana
assert_eq!(segmenter.char_type('漢'), "H");  // Kanji
assert_eq!(segmenter.char_type('A'), "A");   // ASCII
```

### `add_corpus`

```rust
pub fn add_corpus(&mut self, corpus: &str)
```

Processes a space-separated corpus and adds instances to the internal AdaBoost learner.

```rust
let mut segmenter = Segmenter::new(Language::Japanese);
segmenter.add_corpus("テスト です");
```

### `add_corpus_with_writer`

```rust
pub fn add_corpus_with_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, i8),
```

Processes a corpus and calls the callback for each character position with its feature set and label.

```rust
segmenter.add_corpus_with_writer("テスト です", |attrs, label| {
    println!("Features: {:?}, Label: {}", attrs, label);
});
```

### `add_corpus_tsv` / `add_corpus_tsv_with_writer`

```rust
pub fn add_corpus_tsv(&mut self, corpus: &str)
pub fn add_corpus_tsv_with_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, i8),
```

Tab-separated variants of `add_corpus` / `add_corpus_with_writer`: tokens
are separated by tab characters, and a token may be a literal space `" "`.
This preserves the original spacing of the sentence in the training text so
the model can learn from space characters as boundary context (used for the
Korean model; issue #152).

```rust
let mut segmenter = Segmenter::new(Language::Korean);
segmenter.add_corpus_tsv("나는\t \t고양이");
```

### Accessors

```rust
pub fn language(&self) -> Language
pub fn learner(&self) -> &AdaBoost
pub fn learner_mut(&mut self) -> &mut AdaBoost
```

Provide access to the segmenter's language and its internal learner (for a
two-stage segmenter, the stage-1 boundary classifier).

> Feature extraction for a character position (38 features for Korean, 42 for Japanese/Chinese) is an internal detail; the former `get_attributes` method is now private.

## POS-Mode API

The segmenter also supports **word segmentation and POS tagging** with a
two-stage model (issue #147).

### `with_two_stage_learner`

```rust
pub fn with_two_stage_learner(language: Language, learner: TwoStageLearner) -> Self
```

Creates a segmenter with a two-stage model (a `litsea-two-stage v1` file
loaded into a `TwoStageLearner`): the model's stage-1 boundary classifier
becomes the segmenter's AdaBoost-path learner (so `segment` works
naturally), and `segment_with_pos` tags each segmented word through the
candidate-tag lexicon — single-candidate and dominant surfaces skip the
classifier entirely — with the stage-2 word-level tagger deciding ambiguous
surfaces (candidate-masked argmax) and unknown surfaces (full argmax over
all classes). See [Model File Format](../advanced/model-file-format.md) for
the two-stage format.

### `segment_with_pos`

```rust
pub fn segment_with_pos(&self, sentence: &str) -> Result<Vec<(String, Upos)>>
```

Segments a sentence with the stage-1 boundary classifier (exactly as
`segment`) and tags each word with its UPOS tag through the two-stage
tagging path. An empty sentence yields `Ok` with an empty vector.

**Errors** with `LitseaError::PosLearnerNotSet` if no two-stage learner is
set — build the segmenter with `with_two_stage_learner()` first.

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_two_stage.model"))?;

let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);
let tokens = segmenter.segment_with_pos("これはテストです。")?;
// [("これ", Upos::PRON), ("は", Upos::ADP), ("テスト", Upos::NOUN),
//  ("です", Upos::AUX), ("。", Upos::PUNCT)]
```

### `add_corpus_with_pos_writer`

```rust
pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, SegmentLabel)
```

Streams the character-level training features of a POS-tagged corpus
(`word/POS word/POS ...`), including the first position, to a custom
writer, without mutating the segmenter. This is what
`Extractor::extract_two_stage` builds its stage-1 boundary features on.
