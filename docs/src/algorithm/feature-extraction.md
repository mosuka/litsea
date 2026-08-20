# Feature Extraction

Litsea uses character n-gram features to capture the local context around each potential word boundary. This chapter catalogs all feature types: the character-level boundary templates shared by the AdaBoost and two-stage stage-1 pipelines, and the word-level templates used by two-stage's stage-2 tagger.

## Feature Categories

For each character position *i* in the input, the segmenter extracts features from a sliding window of characters, their type codes, and previous boundary decisions.

### Base Features (38 features)

| Category | IDs | Description | Window |
|----------|-----|-------------|--------|
| **UW** (Unary Word) | UW1--UW6 | Individual characters at positions i-3 to i+2 | 6 |
| **BW** (Bigram Word) | BW1--BW3 | Adjacent character pairs | 3 |
| **UC** (Unary Char-type) | UC1--UC6 | Character type codes at positions i-3 to i+2 | 6 |
| **BC** (Bigram Char-type) | BC1--BC3 | Adjacent type code pairs | 3 |
| **TC** (Trigram Char-type) | TC1--TC4 | Type code triples | 4 |
| **UP** (Unary Previous-tag) | UP1--UP3 | Previous 3 boundary decisions | 3 |
| **BP** (Bigram Previous-tag) | BP1--BP2 | Boundary decision pairs | 2 |
| **UQ** (Unary tag+type) | UQ1--UQ3 | Combined boundary decision + type code | 3 |
| **BQ** (Bigram tag+type) | BQ1--BQ4 | Combined decision + type code bigrams | 4 |
| **TQ** (Trigram tag+type) | TQ1--TQ4 | Combined decision + type code trigrams | 4 |

### Language-Specific Features (4 features, Japanese and Chinese only)

| Category | IDs | Description | Count |
|----------|-----|-------------|-------|
| **WC** (Word+Char-type) | WC1--WC4 | Character + type code mixed features | 4 |

- `WC1`: character at i-1 + type code at i
- `WC2`: type code at i-1 + character at i
- `WC3`: character at i-1 + type code at i-1
- `WC4`: character at i + type code at i

> **Why no WC for Korean and English?** Korean Hangul syllables are classified into only two types (SN and SF), so WC features would add noise rather than useful signal. English measured the same outcome directly: a dev-split comparison scored 98.68% Word F1 with the 38 base templates versus 98.65% with all 42 (WC included) -- see [English](../language-support/english.md#no-wc-features).

### Total Feature Count

| Language | Base | WC | Total |
|----------|------|----|-------|
| Japanese | 38 | 4 | **42** |
| Chinese | 38 | 4 | **42** |
| Korean | 38 | 0 | **38** |
| English | 38 | 0 | **38** |

## Single Source of Truth

The whole template above is defined once as a declarative table
(`packed_model::TEMPLATES` -- prefix plus an ordered list of tag/char/type
slots, in a fixed emission order). Both feature representations derive from
it:

- the **string form** below, used for training data extraction, corpus
  processing, and model files;
- the **integer-indexed scoring tables** used by the two-pass scorers of
  `segment()` and `segment_with_pos()`, into which model files are
  compiled at load time (see
  [Prediction Pipeline](prediction-pipeline.md#the-compiled-scoring-tables)).

Adding or reordering a template therefore changes every consumer
consistently. The table order defines the string writer's emission
sequence, which model files and training data depend on.

## Feature Format

Each feature is represented as a string in the format `PREFIX:VALUE`:

```text
UW4:は        ← The character at position i is "は"
UC4:I         ← The type code at position i is "I" (Hiragana)
BW2:はテ      ← The bigram at position i-1..i is "はテ"
BC2:IK        ← The type bigram is Hiragana + Katakana
UP3:B         ← The previous boundary decision was "B" (boundary)
WC1:はK       ← Character "は" combined with type "K"
```

## Sliding Window Layout

The segmenter pads the input with sentinel characters:

```text
Index:   0    1    2    3    4    5    ...  n+2  n+3  n+4  n+5
Chars:   B3   B2   B1   c1   c2   c3  ...  cn   E1   E2   E3
Types:   O    O    O    t1   t2   t3  ...  tn   O    O    O
Tags:    U    U    U    U    ?    ?   ...  ?
```

- **B3, B2, B1** -- Begin sentinels (padding)
- **E1, E2, E3** -- End sentinels (padding)
- **O** -- "Other" type for padding positions
- **U** -- "Unknown" tag for initial positions
- **B** -- "Boundary" tag (word start)
- **O** -- "Other" tag (continuation)

Features are extracted for positions 4 through len-4 (inclusive) for the boundary (AdaBoost) pipeline; the POS pipeline also emits position 3, the first real character, because `segment_with_pos` predicts there to derive the first word's POS (#100). Positions run 4 (or 3) through len-4 (inclusive), where the full window of i-3 to i+2 is available.

## Training Data Format

The `extract` command writes features to a file in this format (real output for the corpus line `これ は テスト です 。`):

```text
-1	BC1:OI	BC2:II	BC3:II	BP1:UU	BP2:UU	BQ1:UOI	BQ2:UII	BQ3:UOI	BQ4:UII	BW1:B1こ	...
1	BC1:II	BC2:II	BC3:IK	BP1:UU	BP2:UO	BQ1:UII	BQ2:UII	BQ3:OII	BQ4:OII	BW1:これ	...
```

Each line contains:

1. A label (`1` for boundary, `-1` for non-boundary)
2. Tab-separated feature strings, written in alphabetically sorted order (not template emission order), so each line starts with the `BC1:` feature

## Word-Level Feature Templates (Two-Stage)

Everything above is the character-level template set scored at each
boundary decision. [Two-stage POS tagging](two-stage-tagging.md)'s stage-2
word tagger scores a separate, unrelated template set defined in
`litsea::word_features` (`N_WORD_TEMPLATES = 23`) -- one row of features
per already-segmented *word*, not per character position. It is a
different declarative table from `packed_model::TEMPLATES` above, compiled
into its own runtime, `packed_two_stage::PackedTwoStageModel` (see
[Prediction Pipeline](prediction-pipeline.md#the-compiled-scoring-tables)).

For a word spanning `[start, end)` of a sentence (`w` = surface,
`n = end - start`):

| Prefix | Value | Representation |
|--------|-------|-----------------|
| `WS` | The word surface itself | Hashed string |
| `WL` | `min(n, 4)` | Dense (word length) |
| `FC` / `LC` | First / last character | Hashed char |
| `ft` / `lt` | First / last character's type code | Dense (type) |
| `TS` | Type codes of the first <= 8 characters | Hashed string |
| `L1`-`L3` / `R1`-`R3` | Context characters at distance 1-3 to the left/right | Hashed char |
| `cl1`-`cl3` / `cr1`-`cr3` | Context character types at distance 1-3 | Dense (type) |
| `LB` / `RB` | Context bigrams (distance 2+1 left / 1+2 right) | Hashed pair |
| `P2` / `S2` | First / last two characters (words with `n >= 2` only) | Hashed pair |

That is 23 templates in total. Context positions beyond the sentence use
begin/end sentinel characters, analogous to the character-level pipeline's
`B1`-`B3` / `E1`-`E3` padding above.

Not every template is written on every extraction: which subset lands in
the `.stage2` feature file is controlled by `TwoStageFeatureSet`
(`full` / `balanced` / `fast`) at extraction time -- see [Extracting
Features](../training-guide/extracting-features.md#two-stage-feature-extraction)
for the CLI flag and what each variant includes.
