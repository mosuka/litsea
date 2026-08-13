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

The bias is a cached field (`-sum(model) / 2.0`, kept in sync by every
weight-mutating path) and is read once per sentence. The packed context
(`packed_context`) borrows word-assembly string slices directly from the
input and carries parallel `u32` char-code and `u8` type-id arrays; the
sentinel entries (`B3`...`E3`) map to code points just above the Unicode
scalar range.

### The Compiled Scoring Tables

The feature template is defined once as a declarative table
(`packed_model::TEMPLATES`), from which four consumers derive: the string
writer used by training and extraction, the load-time parser that
converts each model feature string into an integer key, the two-pass
scorer's tables, and the multiclass POS twin of parser and scorer
(`packed_pos_model::PackedPosModel`, see below). Compilation happens
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
  slot, 8--10 per type slot; about 74 KB total for Japanese). The
  `UC`/`BC`/`TC` tables additionally get merged scatter-vector views for
  the static pass.

Model features that the segmenter's language could never generate (for
example Korean type codes in a Japanese segmenter) are omitted from all
tables; they are unreachable at scoring time, so scores are unaffected.

### Output Equivalence

Scoring accumulates in two-pass order, which differs from the historical
string-keyed accumulation order, so the `f64` sums are no longer
guaranteed bit-for-bit identical. In practice no output difference has
been observed: the exact-equality differential tests (all bundled models,
sentinel stress strings, and a real-text corpus) pass unchanged, and they
remain in the test suite as the detection net for any knife-edge score
flip a future model might expose.

## Joint Segmentation and POS Tagging (`segment_with_pos`)

`segment_with_pos` runs the same two-pass pipeline with the Averaged
Perceptron instead of AdaBoost (issue #143). The perceptron's weights are
compiled once into a multiclass twin of the segmentation tables
(`packed_pos_model::PackedPosModel`), cached next to the AdaBoost tables
and invalidated by `pos_learner_mut()` / `add_corpus_with_pos()`. No
feature strings are built at inference time:

1. **Static pass** -- as in `segment()`, one merged `UW` probe per
   position, one merged `BW` probe per pair, `WC` gathers, and
   `UC`/`BC`/`TC` scatter-twin blocks accumulate into an
   `n x n_classes` score matrix. Because perceptron updates touch only
   the gold/predicted class pair, features average ~3 non-zero classes,
   so the hash-table families store sparse `(class, weight)` rows.
2. **Sequential pass** -- at each position, the 16 tag-dependent dense
   rows are added on top of the static row and the argmax class is
   taken (first strictly-greater class wins, exactly like
   `AveragedPerceptron`'s prediction). A presence bitset skips the rows
   of features absent from the model without touching the weight
   tables. The predicted class maps to a pre-parsed `SegmentLabel`
   (`B-<POS>` or `O`) -- no per-character label string parsing.
3. Unlike `segment()`, decisions start at the **first character
   position** (`lo = 3`): its label determines the first word's POS.
   Since #100 the POS training pipeline emits that position too, so
   training and inference are symmetric (the boundary/AdaBoost pipeline
   still skips it because its first-position label is degenerate).
4. A `B-<POS>` label closes the current word and starts a new one
   carrying that POS; `O` extends the current word. Words are
   materialized from byte offsets of the input, one exact-size
   allocation each.

The string-keyed path is kept test-only (`segment_with_pos_reference`)
as the oracle for exact-equality differential tests, which measure zero
output divergence across all bundled POS models, stress strings, and a
real-text corpus -- the same guarantee net as `segment()`'s.

## Training vs. Prediction

| Aspect | Training (`process_corpus`) | Prediction (`segment`) |
|--------|---------------------------|----------------------|
| Tags source | Pre-computed from the annotated corpus | Dynamically generated by the model |
| First tag | "U" (overrides "B" at position 3) | "U" (no prior decision) |
| First position | Boundary pipeline skips it; POS pipeline emits it (#100) | POS mode predicts it for the first word's POS |
| Labels | Known from corpus (+1 or -1) | Predicted by AdaBoost |
| Features | Written to file via callback (string form) | Packed `u64` keys, no strings |

During training, tags are derived from the ground-truth corpus segmentation, so the model learns from correct boundary decisions. During prediction, tags are generated on-the-fly, meaning each decision depends on all previous predictions -- this is a **left-to-right greedy** approach.

## Performance Characteristics

The segmentation algorithm is **linear** in the length of the input:

- Each character position is visited once: O(n)
- Feature extraction at each position: O(1) (fixed number of templates, each packed into a `u64` on the stack)
- Prediction at each position: O(f) where f is the number of active features (~38-42), but with families merged -- the static pass costs ~2 hash probes (`UW`, `BW`) plus a handful of direct vector loads per character, and the sequential pass is 16 direct dense-array loads
- Total: O(n * f) which is effectively O(n)
- Allocation profile: the packed context borrows word slices from the input and carries flat `u32`/`u8` arrays, the bias is cached, and no strings are built anywhere in the hot loop (the packed table itself is compiled once per model load, off the hot path)
