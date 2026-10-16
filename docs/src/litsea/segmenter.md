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
Korean and English models; issue #152).

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

> Feature extraction for a character position (38 features for Korean/English, 42 for Japanese/Chinese) is an internal detail; the former `get_attributes` method is now private.

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
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

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

## User Dictionary

A `UserDictionary` (issue #263) is an optional set of words that the
segmenter emits as single tokens whenever they occur in the input. It is
a constraint on the boundary decisions, not a change to the model: the
model still scores every position, and the dictionary overrides the
decisions at the edges of an occurrence (forced boundaries) and inside it
(forced non-boundaries). Output examples in this section come from
`japanese.model`.

### `UserDictionary`

```rust
pub struct UserDictionary { /* private */ }

impl UserDictionary {
    pub fn new() -> Self
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>
    pub fn add(&mut self, word: &str) -> bool
    pub fn contains(&self, word: &str) -> bool
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn set_gate(&mut self, gate: Option<f64>) -> Result<()>
    pub fn gate(&self) -> Option<f64>
}
```

Words are stored in a trie keyed by Unicode scalar values and matched
exactly as given — no case folding, no width or Unicode normalization. A
word may contain a space; for languages whose models emit spaces as
tokens (Korean, English) such a word becomes one token containing the
space. Words of one character have no inside to keep together, so `add`
drops them and returns `false` (it also returns `false` for a duplicate).
The type is re-exported as `litsea::UserDictionary`.

### Semantics

- An occurrence of a registered word is emitted as exactly one token: a
  boundary is forced before its first character and after its last one,
  and no boundary is placed inside it.
- Forced decisions enter the model's boundary-tag history (the
  `UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*` features) exactly like model decisions, so
  the positions after a word are scored with consistent features. On
  tag-free (pointwise) models such as `korean.model` and `english.model`
  the dictionary has no effect beyond the occurrence itself.
- Overlapping occurrences are resolved leftmost-longest: scanning from
  the left, the longest registered word starting at the current position
  wins and matching resumes after it.
- Matching is context-free: a registered word is applied wherever its
  characters occur, so a two-character fragment such as `って` or `いま`
  registered as a word breaks every other word it occurs in. Register
  whole words, and consider the gate below for dictionaries that contain
  short entries.
- The dictionary carries no part-of-speech information. In
  `segment_with_pos`, a forced word unknown to the lexicon is tagged by
  the stage-2 tagger's full argmax.
- An empty dictionary, or one whose words never occur, leaves the output
  byte-identical to segmentation without a dictionary.

### Gate

`set_gate(Some(ratio))` applies an occurrence only if its **flip ratio**
is at most `ratio`. The flip ratio is the sum of the model's absolute
scores at the positions whose unconstrained decision the word would
overturn, divided by the sum over every position the word constrains, so
it lies in `0.0..=1.0` and needs no per-model tuning: `0.0` applies a word
only where the model already agrees with it, `1.0` applies it everywhere,
and `None` (the default) skips the check. Values outside `0.0..=1.0` (or
NaN) are rejected with `LitseaError::InvalidInput`.

The gate is for dictionaries that contain short or ambiguous entries:
with `中野` registered and gate `0.0`, `山中野球部` stays `山中 | 野球 | 部`
(the model's margin against `中野` there is large) while
`私の名前は中野です` still yields `中野` as one token (the model agrees).
With no gate the word is applied in both sentences.

On tag-dependent models the gate costs one extra sequential scoring pass
for sentences that contain a candidate — the unconstrained decisions it
compares against are computed without the words an earlier position may
force, a documented approximation; on pointwise models it is free.

### `with_user_dictionary` / `set_user_dictionary` / `user_dictionary`

```rust
pub fn with_user_dictionary(self, dictionary: UserDictionary) -> Self
pub fn set_user_dictionary(&mut self, dictionary: Option<UserDictionary>)
pub fn user_dictionary(&self) -> Option<&UserDictionary>
```

Attach, replace or clear the dictionary, or read it back. The compiled
scoring table is not involved, so this is cheap at any time, and without
a dictionary the segmentation hot path is a single check per sentence.

```rust
use std::path::Path;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::user_dictionary::UserDictionary;

let mut learner = AdaBoost::new(0.01, 100);
learner.load_model_from_path(Path::new("./models/japanese.model"))?;

let segmenter = Segmenter::with_learner(Language::Japanese, learner)
    .with_user_dictionary(UserDictionary::from_words(["雪だるま", "新幹線"]));
let tokens = segmenter.segment("雪だるまを作る");
// ["雪だるま", "を", "作る"]   (without the dictionary: ["雪", "だる", "ま", "を", "作る"])
```
