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

## Constructor

### `Segmenter::new`

```rust
pub fn new(language: Language, learner: Option<AdaBoost>) -> Self
```

Creates a new segmenter.

- `language` -- The language for character type classification
- `learner` -- An optional pre-trained `AdaBoost` model. If `None`, a default (untrained) instance is created.

```rust
use litsea::language::Language;
use litsea::segmenter::Segmenter;

// With a pre-trained model
let segmenter = Segmenter::new(Language::Japanese, Some(learner));

// Without a model (for training or feature extraction)
let segmenter = Segmenter::new(Language::Japanese, None);
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
pub fn char_type(&self, ch: &str) -> &str
```

Classifies a single character into its type code using language-specific rules. The first character of the `&str` is classified; an empty string returns `"O"`.

```rust
let segmenter = Segmenter::new(Language::Japanese, None);
assert_eq!(segmenter.char_type("あ"), "I");  // Hiragana
assert_eq!(segmenter.char_type("漢"), "H");  // Kanji
assert_eq!(segmenter.char_type("A"), "A");   // ASCII
```

### `add_corpus`

```rust
pub fn add_corpus(&mut self, corpus: &str)
```

Processes a space-separated corpus and adds instances to the internal AdaBoost learner.

```rust
let mut segmenter = Segmenter::new(Language::Japanese, None);
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
