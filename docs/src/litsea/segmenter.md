# Segmenter

The `Segmenter` struct is the primary interface for word segmentation.

## Definition

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: two_stage: Option<PackedTwoStageModel> (compiled stage-2 model)
    // private: user_dictionary: Option<UserDictionary>
    // internal: packed cache (see below)
}
```

The fields are private; use the accessor methods `language()`, `learner()`, `learner_mut()`, and `user_dictionary()` to reach them. The user dictionary is set with `with_user_dictionary` or `set_user_dictionary` (see [User Dictionary](#user-dictionary)).

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
set; `segment_with_pos` returns `Err(LitseaError::PosLearnerNotSet)` for any
non-empty sentence — use
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

Processes a space-separated corpus and adds instances to the internal AdaBoost learner, one per character position except the first.

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

Processes a corpus and calls the callback for each character position except the first (whose label is always a word start) with its feature set and boundary label (`1` = word start, `-1` = continuation). For example, `"テスト です"` has five characters, so the callback is called four times.

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
Korean and English models; issue #152). Like `add_corpus` /
`add_corpus_with_writer`, they skip the first character position.

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
tagging path. A word that a user dictionary registered with a tag gets
that tag instead (see [Tags](#tags)). An empty sentence yields `Ok` with an
empty vector.

**Errors** with `LitseaError::PosLearnerNotSet` if no two-stage learner is
set and the sentence is not empty (an empty sentence returns `Ok` with an
empty vector before the learner is checked) — build the segmenter with
`with_two_stage_learner()` first.

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

### `add_corpus_tsv_with_pos_writer`

```rust
pub fn add_corpus_tsv_with_pos_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, SegmentLabel)
```

Tab-separated variant of `add_corpus_with_pos_writer`
(`word/POS\tword/POS\t...`, issue #198). A token may be a literal space
`" "`, so the training text keeps the original spacing; a space token has no
`/POS` suffix and is labeled `B(Upos::X)`; the two-stage extractor
collapses every stage-1 label to `B`/`O`, so the tag on a space has no
effect there. Like the
space-separated variant, it includes the first position and does not mutate
the segmenter. `Extractor::extract_two_stage` uses it for tab-separated
corpora.

## User Dictionary

A `UserDictionary` (issue #263) is an optional set of words that the
segmenter emits as single tokens whenever they occur in the input. It is
a constraint on the boundary decisions, not a change to the model: the
model still scores every position, and the dictionary overrides the
decisions at the edges of an occurrence (forced boundaries) and inside it
(forced non-boundaries). A word may also carry a UPOS tag that
`segment_with_pos` gives its tokens (issue #264, see [Tags](#tags)).
Output examples in this section come from `japanese.model`, and from
`japanese_pos.model` for the tags.

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
    pub fn add_with_pos(&mut self, word: &str, pos: Upos) -> bool
    pub fn contains(&self, word: &str) -> bool
    pub fn pos(&self, word: &str) -> Option<Upos>
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
does not store them and returns `false`, while `add_with_pos` stores them
as tag-only entries (see [One-Character Entries](#one-character-entries));
both return `false` for a word that is already registered, and `len` and
`contains` count the one-character entries too. The type is re-exported
as `litsea::UserDictionary`.

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
- A word registered without a tag is tagged by `segment_with_pos` like any
  other word: through the lexicon if it knows the surface, otherwise by
  the stage-2 tagger's full argmax. A tag registered with `add_with_pos`
  overrides both (see [Tags](#tags)).
- An empty dictionary, one whose words never occur, or one with only
  one-character entries leaves the output of `segment` byte-identical to
  segmentation without a dictionary.

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

### Tags

`add_with_pos(word, pos)` registers a word of two or more characters
exactly as `add` does — it constrains segmentation the same way — and
attaches a UPOS tag to it (issue #264); a word of one character becomes a
tag-only entry (see [One-Character Entries](#one-character-entries)).
Calling it for a word that is already registered sets the
tag, replacing any earlier one; `add` never removes a tag. `pos(word)`
returns the registered tag, or `None` for a word that is unregistered or
untagged.

In `segment_with_pos`, every token whose surface is a tagged word gets the
registered tag, ahead of both the lexicon's fixed tags and the stage-2
tagger; every other token keeps the tagger's tag. The stage-2 tagger
scores each word from its own surface and the neighboring characters,
never from the tags of other words, so a registered tag changes no other
token's tag.

- Without a gate, the tagged tokens are exactly the occurrences the
  dictionary forces.
- With a gate, an occurrence the gate rejects is still tagged if the model
  emits it exactly as registered anyway — for example because the forced
  boundary of an accepted neighboring word completes it. An occurrence
  that does not come out as a token is not tagged.
- `segment` and `segment_into` ignore tags, and `segment_with_pos` does no
  extra work for a dictionary without tags.

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;
use litsea::upos::Upos;
use litsea::user_dictionary::UserDictionary;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

let mut dictionary = UserDictionary::new();
dictionary.add_with_pos("チャットGPT", Upos::PROPN);
let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner)
    .with_user_dictionary(dictionary);
let tokens = segmenter.segment_with_pos("チャットGPTに聞く")?;
// [("チャットGPT", Upos::PROPN), ("に", Upos::ADP), ("聞く", Upos::VERB)]
// Registered without a tag, the word comes out as ("チャットGPT", Upos::NOUN);
// without the dictionary, as ("チャット", Upos::NOUN), ("GPT", Upos::NOUN).
```

### One-Character Entries

A word of one character has no inside to keep together, so it cannot
constrain segmentation, and `add` ignores it. `add_with_pos` registers it
as a tag-only entry (issue #276):

- It is never matched. It forces no boundary, `segment` and `segment_into`
  give the same output as without it, and the gate does not apply to it.
- In `segment_with_pos`, a token that is exactly that character gets the
  registered tag, while the character inside a longer token keeps the
  tagger's tag. Whether an entry applies therefore depends on the model
  emitting the character as a token of its own.
- Like every tag it is context-free, so it suits a word whose tag does not
  depend on the sentence: a one-character name, or a symbol that the
  training data lacks. An entry for a function word such as `で` or `に`
  retags every occurrence, whatever its role. Tags chosen this way from
  the UD test sets gained nothing on held-out sentences, and lost tagged
  tokens for Chinese and Korean (issue #276).

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;
use litsea::upos::Upos;
use litsea::user_dictionary::UserDictionary;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

// The setup of the example above, with this dictionary instead:
let mut dictionary = UserDictionary::new();
dictionary.add_with_pos("楓", Upos::PROPN);
let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner)
    .with_user_dictionary(dictionary);
let tokens = segmenter.segment_with_pos("楓が笑った。")?;
// [("楓", Upos::PROPN), ("が", Upos::ADP), ("笑っ", Upos::VERB),
//  ("た", Upos::AUX), ("。", Upos::PUNCT)]
// Without the entry, the name comes out as ("楓", Upos::NOUN).
```

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
