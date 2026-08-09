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

For each position `i` from 4 to `len(chars) - 3`:

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

## How Prediction Works at Each Position

`segment()` never builds feature strings. When a model is loaded, the
segmenter compiles the learner's string-keyed weights into **packed `u64`
feature keys** once (see below); the hot loop then works entirely on
integers. At each position *i*, the segmenter:

1. **Packs features** -- For each of the 38--42 templates, packs the
   template id (top byte) together with the relevant boundary-tag ids,
   character type ids, and character code points (8 bits per tag/type, 24
   bits per character; the padding sentinels `B3`...`E3` map to code points
   just above the Unicode scalar range) into a single `u64` key on the
   stack. No allocation, no string formatting.
2. **Computes score** -- Each packed key is looked up in an
   `FxHashMap<u64, f64>` (0.0 for unknown features) and added to a running
   score that starts at the bias. The bias itself is a cached field
   (`-sum(model) / 2.0`, kept in sync by every weight-mutating path) and is
   read once per sentence, not recomputed per position:

   ```text
   score = bias + sum(packed_weights[key(template, context)] for each template)
   ```

3. **Makes decision** -- If `score >= 0`, the character starts a new word (boundary); otherwise, it continues the current word
4. **Updates tags** -- Pushes the B or O tag id to the tags array, which affects feature extraction for subsequent positions

The packed context (`packed_context`) borrows word-assembly string slices
directly from the input and carries parallel `u32` char-code and `u8`
type-id arrays; the sentinel entries (`B3`...`E3`) are static values.

### The Packed Scoring Table

The feature template is defined once as a declarative table
(`packed_model::TEMPLATES`), from which three consumers derive: the string
writer used by training/extraction/POS paths, the packed-key writer used by
`segment()`, and the load-time parser that converts each model feature
string into its packed key. Compilation happens eagerly in
`Segmenter::with_learner` and is invalidated whenever the learner is
mutated (`learner_mut()` / `add_corpus`), then rebuilt lazily on the next
`segment()` call. Model features that the segmenter's language could never
generate (for example Korean type codes in a Japanese segmenter) are
omitted from the table; they are unreachable at scoring time, so scores are
unaffected. A lookup miss adds `0.0`, so the floating-point accumulation
sequence -- and therefore the segmentation output -- is bit-for-bit
identical to the historical string-keyed implementation.

## Joint Segmentation and POS Tagging (`segment_with_pos`)

`segment_with_pos` runs the same left-to-right pipeline with the Averaged Perceptron instead of AdaBoost:

1. Features for each position are collected into a reused `Vec<String>` (`collect_attributes`), and the perceptron's per-class score vector is also a reused buffer.
2. The perceptron predicts a `SegmentLabel` (`B-<POS>` or `O`) per position with a single `FxHashMap` lookup per feature (feature -> per-class weight vector layout).
3. The prediction at the **first character position** determines the first word's POS. Since #100 the POS training pipeline emits that position too, so training and inference are symmetric (the boundary/AdaBoost pipeline still skips it because its first-position label is degenerate).
4. A `B-<POS>` label closes the current word and starts a new one carrying that POS; `O` extends the current word.

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
- Prediction at each position: O(f) where f is the number of active features (~38-42), each a single integer-keyed `FxHashMap` probe
- Total: O(n * f) which is effectively O(n)
- Allocation profile: the packed context borrows word slices from the input and carries flat `u32`/`u8` arrays, the bias is cached, and no strings are built anywhere in the hot loop (the packed table itself is compiled once per model load, off the hot path)
