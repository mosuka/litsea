# Prediction Pipeline

This chapter provides a step-by-step walkthrough of how `Segmenter::segment()` processes input text.

## Example: Segmenting "これはテストです。"

### Step 1: Initialize Arrays with Padding

```text
chars: ["B3", "B2", "B1"]
types: ["O",  "O",  "O" ]
tags:  ["U",  "U",  "U", "U"]
```

The tags array gets one extra "U" because `tags[3]` represents the first real character's tag (set to "Unknown" since there is no prior boundary decision).

### Step 2: Scan Input Characters

For each character in the input, determine its type using language-specific patterns and append to the arrays:

```text
chars: ["B3","B2","B1", "こ","れ","は","テ","ス","ト","で","す","。"]
types: ["O", "O", "O",  "I", "I", "I", "K", "K", "K", "I", "I", "P"]
```

### Step 3: Append End Sentinels

```text
chars: [..., "。", "E1", "E2", "E3"]
types: [..., "P",  "O",  "O",  "O" ]
```

### Step 4: Iterate and Predict

For each position `i` from 4 through `len(chars) - 4` (inclusive):

```text
i=4 (れ): Extract features → predict → label=-1 (O) → word="これ"
i=5 (は): Extract features → predict → label=+1 (B) → push "これ", word="は"
i=6 (テ): Extract features → predict → label=+1 (B) → push "は", word="テ"
i=7 (ス): Extract features → predict → label=-1 (O) → word="テス"
i=8 (ト): Extract features → predict → label=-1 (O) → word="テスト"
i=9 (で): Extract features → predict → label=+1 (B) → push "テスト", word="で"
i=10(す): Extract features → predict → label=-1 (O) → word="です"
i=11(。): Extract features → predict → label=+1 (B) → push "です", word="。"
```

### Step 5: Push Final Word

Push the remaining word "。" to the result.

### Result

```text
["これ", "は", "テスト", "です", "。"]
```

## How Prediction Works: Two Passes

`segment()` never builds feature strings. When a model is loaded, the
segmenter compiles the learner's string-keyed weights into integer-indexed
tables once (see below), and each sentence is then scored in **two
passes**, exploiting the fact that only 16 of the 38--42 features depend
on earlier boundary decisions:

1. **Static pass** -- every tag-free feature is accumulated into a
   per-position score buffer in one sweep over the sentence:
   - Each character position makes **one merged `UW` probe** (char code ->
     `[UW1..UW6]` weights) and **one direct `UC` vector load** (type id ->
     `[UC1..UC6]`), scatter-adding the six values to the six neighboring
     decision positions they feed.
   - Each adjacent pair makes one merged `BW` probe and one direct `BC`
     vector load (three values each), and each triple one direct `TC`
     vector load (four values).
   - For Japanese/Chinese, each character makes one merged `WC` row probe
     (the row is direct-indexed by type id and scatter-added to the two
     decision positions the character feeds); the block is skipped
     entirely for models without `WC` features.
2. **Sequential pass** -- at each position *i*, the score starts from the
   bias plus the precomputed static score, adds the 16 tag-dependent
   weights (`UP*`, `BP*`, `UQ*`, `BQ*`, `TQ*` -- all direct-indexed dense
   array loads, no hashing), and decides: if `score >= 0`, the character
   starts a new word; otherwise it continues the current one. The decision
   is pushed to the tags array and feeds the next positions' lookups.

   ```text
   score = bias + static[i] + sum(dense[tag-dependent template][mixed-radix index])
   ```

   **Pointwise fast path** (issue #183): if the model has no tag-dependent
   features at all (e.g. it was trained with `litsea extract --tag-free`,
   like the bundled `korean.model`/`english.model`), the compiled model records this at
   load time and the sequential pass is skipped entirely -- the decision
   reduces to `bias + static[i] >= 0` with no tag bookkeeping. Output is
   identical either way (the skipped loads would each add `0.0`); only the
   serial dependency between positions disappears.

The bias is a cached field (`-sum(model) / 2.0`, kept in sync by every
weight-mutating path) and is read once per sentence. The packed context
carries parallel `u32` char-code and `u8` type-id arrays plus per-character
byte offsets (the boundary pipeline emits tokens as byte ranges of the
input, #184; `segment()` materializes them as `String`s, `segment_into()`
returns them directly from a reusable buffer); the sentinel entries
(`B3`...`E3`) map to code points just above the Unicode scalar range.

### The Compiled Scoring Tables

The feature template is defined once as a declarative table
(`packed_model::TEMPLATES`), from which three consumers derive: the string
writer used by training and extraction, the load-time parser that
converts each model feature string into an integer key, and the two-pass
scorer's tables. Compilation happens
eagerly in `Segmenter::with_learner` and is invalidated whenever the
learner is mutated (`learner_mut()` / `add_corpus`), then rebuilt lazily
on the next `segment()` call.

The compiled model splits by key-space size and tag dependence:

- **Merged-vector hash tables** for char n-grams: `UW1..6` collapse into
  one `char -> [f64; 6]` table and `BW1..3` into one
  `(char, char) -> [f64; 3]` table, so a whole family costs one probe.
  `WC1..4` likewise collapse into one `char -> [slot][type_id]` row table
  (one probe per character, type dimension direct-indexed).
- **Dense arrays** for tag/type-only templates: each of the 29 gets a
  direct-indexed table sized by the exact mixed-radix product (3 per tag
  slot, 7--10 per type slot; about 74 KB total for Japanese). The
  `UC`/`BC`/`TC` tables additionally get merged scatter-vector views for
  the static pass.

Model features that the segmenter's language could never generate (for
example Korean type codes in a Japanese segmenter) are omitted from all
tables; they are unreachable at scoring time, so scores are unaffected.

This table and its three consumers cover the character-level (stage-1 /
AdaBoost) feature set only. [Two-stage POS
tagging](two-stage-tagging.md)'s stage-2 word tagger is compiled from a
separate, parallel declarative table (`litsea::word_features`, 23
word-level templates) into its own runtime,
`packed_two_stage::PackedTwoStageModel` -- see [Feature
Extraction](feature-extraction.md#word-level-feature-templates-two-stage)
for its template catalog and the section below for how it fits into
`segment_with_pos`.

### Output Equivalence

Scoring accumulates in two-pass order, which differs from the historical
string-keyed accumulation order, so the `f64` sums are no longer
guaranteed bit-for-bit identical. In practice no output difference has
been observed: the exact-equality differential tests (all bundled models,
sentinel stress strings, and a real-text corpus) pass unchanged, and they
remain in the test suite as the detection net for any knife-edge score
flip a future model might expose.

## User Dictionary

With a user dictionary set (`Segmenter::with_user_dictionary` or
`set_user_dictionary`; on the CLI, `--user-words` and `--user-words-gate`),
`segment()` adds one step between the static pass and the decisions
(issue #263):

1. Occurrences of registered words of two or more characters are found in
   the sentence; overlapping occurrences are resolved leftmost-longest.
2. With a gate, an occurrence is applied only if its flip ratio (the sum of
   the absolute unconstrained scores at the positions it would flip,
   divided by the sum over all positions it constrains) is at most the
   gate.
3. Each applied occurrence forces a boundary (`B`) at both of its edges and
   no boundary (`O`) inside it. The other positions are decided by the
   model as above; on a tag-dependent model, a forced decision enters the
   tag history like a model decision.

Without a dictionary, or when no registered word occurs in the sentence,
the decisions are the model's alone. See [Segmenter: User
Dictionary](../litsea/segmenter.md#user-dictionary) for the full semantics.

## Segmentation and POS Tagging (`segment_with_pos`)

`segment_with_pos` requires a segmenter built with
[`with_two_stage_learner`](../litsea/segmenter.md#with_two_stage_learner)
(issue #147). It segments with the ordinary `segment()` boundary path
described above, then tags each resulting word through
`packed_two_stage::PackedTwoStageModel::tag_words` -- a candidate-tag
lexicon lookup, with a masked-argmax fallback for ambiguous known words
and a full-argmax fallback for unknown words. The POS layer therefore adds
no per-character scoring at all; see [Two-Stage
Tagging](two-stage-tagging.md) for the full explanation of that pipeline
and its cost model. When the user dictionary has tagged words
(issue #264), a word whose surface is a tagged dictionary word gets the
registered tag instead of the lexicon's or the tagger's; a tagged entry of
one character (issue #276) changes no boundary and retags only the words
that are exactly that character.

## Training vs. Prediction

| Aspect | Training (`process_corpus`) | Prediction (`segment`) |
|--------|---------------------------|----------------------|
| Tags source | Pre-computed from the annotated corpus | Dynamically generated by the model |
| First tag | "U" (overrides "B" at position 3) | "U" (no prior decision) |
| First position | Boundary pipeline skips it; the two-stage stage-1 extraction (`extract --pos`) emits it, always labelled `B` (#100) | Never scored: it always starts the first word, in `segment_with_pos` too |
| Labels | Known from corpus (+1 or -1) | Predicted by AdaBoost |
| Features | Written to file via callback (string form) | Packed `u64` keys, no strings |

During training, tags are derived from the ground-truth corpus segmentation, so the model learns from correct boundary decisions. During prediction, tags are generated on-the-fly, meaning each decision depends on all previous predictions -- this is a **left-to-right greedy** approach.

## Performance Characteristics

The segmentation algorithm is **linear** in the length of the input:

- Each character position is visited once: O(n)
- Feature lookup at each position: O(1) (a fixed number of templates, scored through merged-table probes and direct array loads; no per-template keys or strings are built)
- Prediction at each position: O(f) where f is the number of active features (~38-42, or 16 fewer for a tag-free model), but with families merged -- the static pass costs ~2 hash probes (`UW`, `BW`) plus a handful of direct vector loads per character, and the sequential pass is 16 direct dense-array loads, skipped entirely for a tag-free model (the bundled `korean.model` and `english.model`)
- Total: O(n * f) which is effectively O(n)
- Allocation profile: the packed context stores no slices of the input, only flat `u32`/`u8` arrays and per-character byte offsets in a reusable `SegmentBuffer`, the bias is cached, and no strings are built anywhere in the hot loop (the packed table itself is compiled once per model load, off the hot path)
