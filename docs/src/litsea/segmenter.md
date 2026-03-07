# Segmenter

The `Segmenter` struct is the primary interface for word segmentation.

## Definition

```rust
pub struct Segmenter {
    pub language: Language,
    pub learner: AdaBoost,
    // internal: char_types: CharTypePatterns
}
```

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

### `get_type`

```rust
pub fn get_type(&self, ch: &str) -> &str
```

Classifies a single character into its type code using language-specific patterns.

```rust
let segmenter = Segmenter::new(Language::Japanese, None);
assert_eq!(segmenter.get_type("あ"), "I");  // Hiragana
assert_eq!(segmenter.get_type("漢"), "H");  // Kanji
assert_eq!(segmenter.get_type("A"), "A");   // ASCII
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

### `get_attributes`

```rust
pub fn get_attributes(
    &self,
    i: usize,
    tags: &[String],
    chars: &[String],
    types: &[String],
) -> HashSet<String>
```

Extracts the feature set for a specific character position. Returns 38 features (Korean) or 42 features (Japanese/Chinese).

> This is primarily used internally by `segment()` and `process_corpus()`.
