# Segmenter

The `Segmenter` struct is the primary interface for word segmentation.

## Definition

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: pos_learner: Option<AveragedPerceptron>,
}
```

The fields are private; use the accessor methods `language()`, `learner()`, `learner_mut()`, `pos_learner()`, and `pos_learner_mut()` to reach them.

## Constructors

### `Segmenter::new`

```rust
pub fn new(language: Language) -> Self
```

Creates a segmenter with a default (untrained) `AdaBoost` learner — suitable
for training or feature extraction. Until a model is loaded or training data
is added, `segment` returns one word per character. The POS learner is left
unset; `segment_with_pos` returns `Err(LitseaError::PosLearnerNotSet)` — use
[`with_pos_learner`](#pos-mode-api) for joint segmentation + POS tagging.

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
pub fn pos_learner(&self) -> Option<&AveragedPerceptron>
pub fn pos_learner_mut(&mut self) -> Option<&mut AveragedPerceptron>
```

Provide access to the segmenter's language and its internal learners.

> Feature extraction for a character position (38 features for Korean, 42 for Japanese/Chinese) is an internal detail; the former `get_attributes` method is now private.

## POS-Mode API

The segmenter also supports **joint word segmentation and POS tagging** with
an Averaged Perceptron model.

### `with_pos_learner`

```rust
pub fn with_pos_learner(language: Language, pos_learner: AveragedPerceptron) -> Self
```

Creates a segmenter configured for joint segmentation + POS tagging.

### `segment_with_pos`

```rust
pub fn segment_with_pos(&self, sentence: &str) -> Result<Vec<(String, Upos)>>
```

Segments a sentence and jointly predicts each word's UPOS tag. The
prediction at the first character position determines the first word's POS.
An empty sentence yields `Ok` with an empty vector.

**Errors** with `LitseaError::PosLearnerNotSet` if no POS learner is set —
build the segmenter with `with_pos_learner()` or register training data with
`add_corpus_with_pos()` first.

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;

let mut pos_learner = AveragedPerceptron::new();
pos_learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

let segmenter = Segmenter::with_pos_learner(Language::Japanese, pos_learner);
let tokens = segmenter.segment_with_pos("これはテストです。")?;
// [("これ", Upos::PRON), ("は", Upos::ADP), ("テスト", Upos::NOUN),
//  ("です", Upos::AUX), ("。", Upos::PUNCT)]
```

### `add_corpus_with_pos`

```rust
pub fn add_corpus_with_pos(&mut self, corpus: &str)
```

Adds a POS-tagged corpus (`word/POS word/POS ...`) as Averaged Perceptron
training data, creating the POS learner on first use.

### `add_corpus_with_pos_writer`

```rust
pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, SegmentLabel)
```

Streams the POS training features for each character position (including
the first one) to a custom writer, without mutating the segmenter. This is
what `Extractor::extract_with_pos` builds on.
