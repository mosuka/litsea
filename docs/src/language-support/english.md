# English

Litsea supports English word segmentation with a character type set tuned
for Latin-script orthography: case, whitespace, and the apostrophe as
distinct classes.

## Character Types

| Code | Name | Pattern | Examples |
|------|------|---------|----------|
| **U** | Uppercase Latin | `[A-ZＡ-Ｚ]` | A, Z, Ｔ |
| **W** | Whitespace | Space, tab, no-break space | ` `, `\t`, U+00A0 |
| **Q** | Apostrophe | `[\u{27}\u{2019}]` | `'`, `’` |
| **P** | Punctuation | ASCII punctuation (minus apostrophe) + General Punctuation dashes/quotes/ellipsis (minus U+2019) + CJK/full-width | `.`, `-`, `"`, `@`, `。` |
| **A** | Lowercase Latin | `[a-zａ-ｚ]` | a, z |
| **N** | Digits | `[0-9０-９]` | 0, 5, ５ |
| **O** | Other | Fallback | CJK ideographs, accented Latin outside ASCII |

### Uppercase as a Distinct Class

Sentence-initial capitals, proper nouns, and acronyms correlate strongly
with word boundaries in English, so uppercase Latin gets its own type ("U")
instead of collapsing into the same class as lowercase ("A"). This mirrors
how the other languages carve out a linguistically distinctive subset (e.g.
Korean's particle characters) from a broader shared class.

### Whitespace as a Distinct Class

The shared `punct_latin_digit()` helper used by every language does not
classify ASCII punctuation or the ASCII space (U+0020) — both fall through
to `"O"` for Japanese, Chinese, and Korean, since none of those languages'
corpora need horizontal whitespace to carry a boundary signal on its own.
English does: the type table adds "W" for space, tab, and no-break space
(only the plain space occurs in the training corpus; the other two share
its id so pasted input inherits the same behavior instead of falling back
to "O").

### Apostrophe as a Distinct Class

The apostrophe is the character-level signal that separates a contraction
or possessive from ordinary punctuation: `do` + `n't`, `Google` + `'s`. It
gets its own type ("Q") covering both the ASCII apostrophe (U+0027) and the
typographic right single quotation mark (U+2019), which is common in the
training corpus's source text. `Q` is deliberately excluded from `P` so the
character-level feature templates can key on it directly.

### Punctuation Is Uniform

Unlike the other three languages — where ASCII punctuation such as `@`
falls through to `"O"` and only CJK/full-width punctuation maps to `"P"` —
English classifies essentially all ASCII punctuation (minus the apostrophe)
as `"P"`, alongside the same General Punctuation range (dashes, curly
quotes, ellipsis) that covers non-ASCII editions of the training text. This
is a deliberate, English-specific difference: `char_type('@')` returns
`"O"` for Japanese/Chinese/Korean but `"P"` for English.

**Hyphen** is classified as `"P"` rather than given its own type. UD
English-EWT tokenizes hyphenated compounds as separate tokens (e.g.
`search`-`engine`), so a hyphen behaves like ordinary separator punctuation
in the gold standard; the raw character is still visible to the
character-level (`UW*`/`BW*`) templates, so hyphen-specific behavior
remains learnable without an eighth type code, which would grow the
dense feature tables by a further \\((8/7)^3 \\approx 1.49\\times\\).

### No WC Features

English does **not** use WC (word + character-type) features, the same
choice as Korean and for a related reason: the dominant boundary signal
(whitespace) already resolves most positions, so the mixed char/type
templates add little on top of it. This was verified empirically, not just
by analogy — on a held-out dev split, the tag-free segmentation model
scored **98.68%** Word F1 with the 38 base templates versus **98.65%** with
all 42 templates (`WC1`--`WC4` included), i.e. adding WC features made the
model *worse*, not better.

### Space-Preserving Training

English is written with spaces between words, and (outside contractions
and a few punctuation cases) those spaces mark most word boundaries. Like
[Korean](korean.md#space-preserving-training), the model is trained on a
**space-preserving TSV corpus**: tokens are tab-separated and each space is
kept as its own token, so the training text contains the space characters
of the original sentence and the model can use them as boundary context.
Generate the corpus with `corpus_udtreebank.sh -s` and extract features
with `litsea extract --format tsv`. At inference no special handling is
needed: `segment()` receives the spaced text as-is and emits each space as
its own token.

**Multiword tokens (contractions).** UD English-EWT represents a
contraction such as `don't` as a *range* line (e.g. ids `3-4`) covering two
word lines (`do`, `n't`) with no space between them. `corpus_udtreebank.sh
-s` treats a range line specially: it emits no token of its own, suppresses
space insertion *between* the range's member words, and applies the
range's own `SpaceAfter` annotation after the last member word. Concretely,
the sentence "I don't know." becomes the token sequence `I`, ` `, `do`,
`n't`, ` `, `know`, `.` -- matching `english.model`'s actual output (see
the example below). This invariant — concatenating a range's member word
forms reproduces the range's own surface form — holds for every multiword
token in UD English-EWT.

Because each space is its own single-character token, the character-level
labeling marks two separate boundaries around it, exactly as for Korean:
see [Korean's explanation](korean.md#space-preserving-training) of why this
is a near-trivial rule for the model and does not affect held-out Word F1
(pure-whitespace tokens are excluded from scoring).

## Pre-trained Models

### english.model

- **Training corpus**: UD English-EWT (space-preserving TSV corpus)
- **Training options**: `--format tsv --tag-free`, 20 epochs of Averaged
  Perceptron training (chosen by a dev-split epoch sweep over {10, 20, 30,
  50}; quality peaked at epoch 20 and degraded slightly beyond it),
  collapsed to AdaBoost scalar weights, not pruned (4,794 features) —
  see [Training Procedure](../pre-trained-models.md#training-procedure)
  for the full recipe
- **Word F1 (held-out)**: 98.31%
- **Boundary F1 (held-out)**: 99.18%
- **File size**: ~125 KB

The model is trained without the 16 tag-dependent feature templates
(`--tag-free`, issue #183). A dev-split comparison confirmed tag features
buy almost nothing for English (tagged 38-template best: 98.71% Word F1 at
epoch 30, vs. tag-free 38-template best: 98.68% at epoch 20 — a 0.03pt
difference), so the bundled model ships tag-free and lets `segment()` skip
its sequential scoring pass entirely. See [Tag-Free (Pointwise)
Models](../pre-trained-models.md#tag-free-pointwise-models).

Held-out metrics are computed on the original spaced text with space
tokens excluded from scoring.

### english_pos.model

- **Algorithm**: two-stage segmentation + POS tagging (a binary boundary
  classifier plus a word-level tagger with a candidate-tag lexicon)
- **Stage-2 feature set**: `full` (chosen by a dev-split sweep over
  fast/balanced/full; full gave the best tagged-word accuracy), 50 epochs
- **Word F1 (held-out, unspaced protocol)**: 70.33%
- **Tagged Word F1 (held-out, unspaced protocol)**: 65.83%
- **Word F1 (held-out, real-world/spaced)**: 77.55%
- **Tagged Word F1 (held-out, real-world/spaced)**: 69.89%
- **File size**: ~3.6 MB
- **Details**: see [Pre-trained Models](../pre-trained-models.md#english_posmodel)

> **This model's segmentation quality is substantially lower than
> `english.model`'s, and this is expected, not a bug.** Like
> `korean_pos.model`, this model is *trained* on the *unspaced* `word/POS`
> corpus (the two-stage POS pipeline has no space-preserving variant). The
> "unspaced protocol" row above scores the model the same way it was
> trained, so it is not a train/inference mismatch; the "real-world/spaced"
> row (issue #196) instead reconstructs the model's actual spaced input
> from a space-preserving POS gold (`litsea evaluate --pos --format tsv`
> against `resources/eval/english_ewt_test_pos_spaced.tsv`) and measures
> what `segment --pos` really produces on natural text, without retraining.
> Real-world quality (77.55%) is meaningfully better than the
> unspaced-protocol number suggests, but still well short of
> `english.model`'s 98.31%.
>
> The remaining gap comes from a real difference between English and
> Korean, not an implementation defect: `korean_pos.model`'s real-world
> Word F1 (94.01%) sits only 5.9pt below `korean.model`'s 99.91%, while
> `english_pos.model`'s real-world Word F1 sits 20.8pt below
> `english.model`'s 98.31% -- roughly 3.5x the cost. Korean is
> agglutinative, so its particles and verb endings leave strong
> character-level cues for word boundaries even with every space removed;
> English orthography carries almost no such sub-word signal -- a sequence
> like `"thecatsatonthemat"` gives a classifier little to work with -- so
> training on unspaced text costs English much more real-world quality
> than it costs Korean. At inference the model receives real, spaced text,
> and spaces are re-emitted as their own tokens (tagged by the stage-2
> fallback); the golden test in `litsea/tests/golden.rs` pins this actual,
> imperfect real-world behavior. What issue #196 added is a way to
> *measure* quality on real spaced input, not a way to *train* on it --
> combining the space-preserving TSV protocol with two-stage POS training
> would still need a new corpus format and `Extractor`/`Segmenter` changes
> the pipeline does not currently support, and remains a possible
> follow-up.

## Example

```sh
echo "I don't know." | litsea segment -l english ./models/english.model
# I   do n't   know .
```
