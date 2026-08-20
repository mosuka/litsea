# Pre-trained Models

Litsea ships with several pre-trained models in the `models/` directory.

## Model Catalog

The word segmentation models are evaluated on the held-out test split of
their training treebank (sentences never seen during training). **Word F1**
scores exact word matches; **Boundary F1** scores individual boundary
decisions. Note that the `train` command prints *in-sample* metrics
(measured on the training data itself), which are higher than these
held-out figures.

**Algorithm note**: `japanese.model`, `chinese.model`, `korean.model`, and
`english.model` are trained as a 2-class (boundary/non-boundary) Averaged Perceptron and
then collapsed to scalar per-feature weights (issue #165) -- the file is
still the plain AdaBoost text format the engine has always loaded, and
`Segmenter::with_learner` / `AdaBoost::load_model_from_path` work
unchanged. The collapse is a lossless transform (see
`scripts/collapse_binary_perceptron.py`'s docstring for the derivation),
not an approximation: a perceptron trained this way reaches substantially
higher held-out quality than AdaBoost's presence-stump weak learners on the
same corpus and templates, at the cost of a larger model file (more
distinct features get non-zero weight) and a training procedure that goes
through `train --perceptron` (see [Training Procedure](#training-procedure)
below) rather than plain `train`.

### japanese.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | UD Japanese-GSD |
| Epochs | 50 |
| Pruned To | top 40,000 features by \|weight\| |
| Word F1 (held-out) | 96.70% |
| Boundary F1 (held-out) | 98.59% |
| File Size | ~1.1 MB |

### korean.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | UD Korean-GSD (space-preserving TSV corpus) |
| Epochs | 30 |
| Feature Templates | tag-free (pointwise, issue #183) |
| Pruned To | not pruned (3,132 features) |
| Word F1 (held-out) | 99.91% |
| Boundary F1 (held-out) | 99.96% |
| File Size | ~86 KB |

The Korean model is trained and evaluated on text that preserves the
original inter-eojeol spaces (each space is its own token; space tokens are
excluded from the F1 computation). Spaces mark most word boundaries in
Korean, so a model that sees them during training resolves the UD
Korean-GSD standard almost deterministically -- this is also why Korean's
feature count and file size stay small (there is little ambiguity left for
the model to learn). Japanese and Chinese are written without spaces, so
their protocol is unchanged.

The Korean model is additionally trained **without the 16 tag-dependent
feature templates** (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`, which read the
boundary decisions at the previous one to three positions): with the space
signal available, those templates measured as contributing nothing (99.91%
tag-free vs. 99.90% with them, with ~22% fewer features). A model with no
tag-dependent features is *pointwise* -- every position's decision depends
only on the input text -- so `segment()` skips its sequential scoring pass
entirely (issue #183). See [Tag-Free (Pointwise)
Models](#tag-free-pointwise-models) below for the trade-off in the other
languages.

### english.model

| Property | Value |
|----------|-------|
| Language | English |
| Training Corpus | UD English-EWT (space-preserving TSV corpus) |
| Epochs | 20 |
| Feature Templates | tag-free (pointwise, issue #183), no WC features |
| Pruned To | not pruned (4,794 features) |
| Word F1 (held-out) | 98.31% |
| Boundary F1 (held-out) | 99.18% |
| File Size | ~125 KB |

Like `korean.model`, `english.model` is trained and evaluated on text that
preserves the original spaces (each space is its own token, excluded from
the F1 computation); see [English](language-support/english.md) for the
space-preserving training protocol, the multiword-token (contraction)
handling, and the epoch sweep that picked 20 epochs and the tag-free /
no-WC configuration. English's residual boundary ambiguity (contractions,
hyphenated compounds, abbreviations like "U.S.") is why its held-out Word
F1 sits below Korean's near-deterministic 99.91%, even though both share
the same space-preserving recipe.

### chinese.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | UD Chinese-GSD |
| Epochs | 100 |
| Pruned To | top 70,000 features by \|weight\| |
| Word F1 (held-out) | 90.69% |
| Boundary F1 (held-out) | 95.64% |
| File Size | ~2.0 MB |

### RWCP.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Source | Extracted from the original [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) |
| License | BSD 3-Clause (Taku Kudo) |
| File Size | ~22 KB |

### JEITA_Genpaku_ChaSen_IPAdic.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | JEITA Project Sugita Genpaku corpus |
| Tokenizer | ChaSen with IPAdic |
| File Size | ~16 KB |

## Training Procedure

`RWCP.model` and `JEITA_Genpaku_ChaSen_IPAdic.model` are legacy/compatibility
models and are trained (or sourced) as before -- see [Training
Models](training-guide/training-models.md) for the plain AdaBoost procedure.
`japanese.model`, `chinese.model`, and `korean.model` are retrained with the
binary-perceptron-collapse procedure (#165), which needs no engine changes
but does need a few extra steps beyond plain `litsea train`:

```sh
# 1. Extract plain boundary features (the same step as before). Add
#    --tag-free to drop the 16 tag-dependent templates and train a
#    pointwise model (used for korean.model; see the next section).
litsea extract -l <language> [--format tsv for Korean] [--tag-free] <corpus> <features.txt>

# 2. Remap boundary labels 1/-1 -> B/O. This is required for correctness. not
#    cosmetic: it makes the perceptron's own tie-break (lowest class index
#    wins) agree with AdaBoost's "score >= 0.0 favors boundary" convention.
#    Training directly on "1"/"-1" would silently invert what ties resolve to.
sed -i 's/^1\t/B\t/; s/^-1\t/O\t/' <features.txt>

# 3. Train a 2-class Averaged Perceptron. --perceptron is the generic
#    trainer (PerceptronTrainer treats labels as opaque strings).
litsea train --perceptron --num-epochs <N> <features.txt> <perceptron.model>

# 4. Collapse to the plain AdaBoost model format (lossless -- see the
#    script's docstring for the derivation).
scripts/collapse_binary_perceptron.py <perceptron.model> <collapsed.model>

# 5. Optional: if the larger feature count regresses `cargo bench --
#    external_corpus` throughput more than acceptable, prune to the top-N
#    features by magnitude and re-check both held-out quality and speed.
scripts/prune_adaboost_model.py <collapsed.model> <pruned.model> <n>
```

Epoch count and pruning threshold are per-language tuning knobs, not fixed
constants -- pick them from an epoch sweep and a quality-vs-throughput
sweep on held-out data, the same way the bundled models above were chosen
(see the issue for the full sweep data). As a general shape: quality keeps
improving well past a handful of epochs and eventually plateaus (or
mildly overfits, as Japanese does past ~50 epochs) rather than needing a
single "correct" epoch count; pruning quality tends to degrade gracefully
until a language-specific cliff, so sweep a few pruning levels around
where `cargo bench` throughput starts recovering rather than guessing one
number.

## Tag-Free (Pointwise) Models

16 of the boundary feature templates (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`) read
the model's own boundary decisions at the previous one to three positions.
They chain each decision to the previous ones, which forces `segment()`'s
scoring into a strictly sequential pass. A model trained **without** them
(`litsea extract --tag-free`) is *pointwise* -- every position depends
only on the input text -- and `segment()` detects this at model-load time
and skips the sequential pass entirely (issue #183).

What the tag features are worth differs sharply by language (all figures
from converged epoch sweeps on the UD GSD test splits, issue #183):

| Language | Word F1 with tags | Word F1 tag-free | Throughput change |
|----------|-------------------|------------------|-------------------|
| Korean | 99.90% | **99.91%** | faster (sequential pass skipped) |
| English | 98.71% | **98.68%*** | faster (sequential pass skipped) |
| Japanese | **96.70%** | 96.33% | ~+45-50% measured end-to-end |
| Chinese | **90.69%** | 90.18% | ~+12% measured end-to-end |

\* English's dev-split comparison (98.71% with tags vs. 98.68% tag-free, a
0.03pt difference) is close enough that either choice is defensible; the
bundled model ships tag-free for the same throughput reason as Korean. The
held-out test-split figure reported for `english.model` above (98.31%) is
measured only once, at the end of the sweep, and is not directly comparable
to this dev-split pair.

With the inter-eojeol space signal available, Korean's tag features
contribute nothing, so `korean.model` ships tag-free (and ~22% smaller).
English is close behind for the same reason (dominant whitespace signal).
For Japanese and Chinese they still buy 0.37-0.51pt of Word F1, so the
bundled models keep them -- quality stays the default. If your workload
prefers speed, retrain with the same procedure above plus `--tag-free` on
the extract step; the throughput numbers were measured on this project's
development machine with the paired methodology of
[Benchmarking](advanced/benchmarking.md), so expect the ratio, not the
absolute numbers, to carry over.

## Two-Stage POS Tagging Models

The [two-stage architecture](algorithm/two-stage-tagging.md) (issue #147)
segments with a binary boundary classifier and tags each resulting word
through a candidate-tag lexicon plus a word-level tagger, instead of
scoring every UPOS class at every character position.

Held-out rows are word / tagged-word F1 measured with `litsea evaluate
--pos` on the UD GSD test splits (see
[Evaluating Models](training-guide/evaluating-models.md)). The Korean POS
gold follows the POS pipeline's convention (no space tokens), so it is
evaluated on unspaced text. "Stage-2 feature set" is the word-level
template selection (`fast`, `balanced`, or `full`; see [Extracting
Features](training-guide/extracting-features.md)) chosen for the bundled
file per language, from the measured tradeoff in [Two-Stage
Tagging](algorithm/two-stage-tagging.md#choosing-a-stage-2-feature-set).
Throughput is from `cargo bench -- external_corpus` on the same corpora as
the [Benchmarking](advanced/benchmarking.md) page, run on this project's
development machine (not dedicated, idle hardware -- see that page's
methodology note).

**Epoch note**: an epoch sweep during two-stage bundling (10 to 150
epochs) found that stage 1's *segmentation* quality specifically continues
improving well past 10 epochs and plateaus around 50 -- the bundled
two-stage models below use 50 epochs, chosen from that sweep. When
retraining, a one-shot low-epoch run will understate the quality the
architecture can reach (see the [methodology
note](algorithm/two-stage-tagging.md#a-methodology-note-use-enough-training-epochs)).

### japanese_pos.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | UD Japanese-GSD (7,050 sentences) |
| Epochs | 50 |
| Stage-2 Feature Set | `fast` |
| Word F1 (held-out) | 96.78% |
| Tagged Word F1 (held-out) | 92.95% |
| Throughput | 4.38M chars/s |
| File Size | ~5.4 MB |

### chinese_pos.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 50 |
| Stage-2 Feature Set | `balanced` |
| Word F1 (held-out) | 90.82% |
| Tagged Word F1 (held-out) | 82.29% |
| Throughput | 3.38M chars/s |
| File Size | ~8.0 MB |

### korean_pos.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | UD Korean-GSD (4,400 sentences, unspaced word/POS protocol -- see the note below) |
| Epochs | 50 |
| Stage-2 Feature Set | `balanced` |
| Word F1 (held-out) | 83.24% |
| Tagged Word F1 (held-out) | 78.86% |
| Throughput | 4.54M chars/s |
| File Size | ~5.0 MB |

Korean's throughput profile traces to its lexicon: held-out text is 34.5%
unknown words (surfaces never seen in training), and unknown words always
take the full stage-2 classifier fallback rather than the cheap
dominance-skip or candidate-masked paths, so a larger share of Korean's
words pay the full stage-2 cost than in Japanese or Chinese.

**Korean protocol note**: `korean_pos.model` is trained on the
unspaced `word/POS` corpus, *not* the space-preserving TSV corpus
`korean.model` uses (issue #152). The two-stage extractor takes a single
corpus for both stages, and building a combined space-preserving +
POS-tagged format is a separate feature not yet implemented; the numbers
above are therefore not comparable to `korean.model`'s 99.91% (a different
corpus and protocol entirely, not a stronger or weaker two-stage result).

### english_pos.model

| Property | Value |
|----------|-------|
| Language | English |
| Training Corpus | UD English-EWT (12,544 sentences, unspaced word/POS protocol -- see the note below) |
| Epochs | 50 |
| Stage-2 Feature Set | `full` |
| Word F1 (held-out) | 70.33% |
| Tagged Word F1 (held-out) | 65.83% |
| Throughput | 2.05M chars/s |
| File Size | ~3.6 MB |

**English protocol note, read before comparing to `english.model`'s
98.31%.** `english_pos.model` is trained (and evaluated) on the same
unspaced `word/POS` protocol as `korean_pos.model` -- this is not a
train/inference mismatch, `evaluate --pos` scores it the same way it was
trained. The much larger quality gap for English than for Korean (Korean:
99.90% unspaced vs. 99.91% spaced -- almost no gap; English: 70.33%
unspaced vs. 98.31% spaced -- a large gap) reflects a genuine difference
between the two languages, not an implementation defect: Korean is
agglutinative, so particles and verb endings leave strong character-level
cues for word boundaries even with every space removed, while English
orthography carries almost none, so segmenting unspaced English text (e.g.
`"thecatsatonthemat"`) is intrinsically a much harder task for this
model's feature templates. At inference the model still receives real,
spaced input, and spaces are re-emitted as their own tokens; see
[English](language-support/english.md#english_posmodel) for the full
explanation and a pinned example of the real (imperfect) output. Combining
the space-preserving TSV protocol with two-stage POS training would need a
new corpus format the pipeline does not currently support.

#### Usage

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_pos.model
```

Output:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## Choosing a Model

- For **Japanese**, use `japanese.model` for the best accuracy, or `RWCP.model` for compatibility with the original TinySegmenter
- For **Chinese**, use `chinese.model`
- For **Korean**, use `korean.model`
- For **English**, use `english.model`
- For **POS tagging**, use the **two-stage** models
  (`japanese_pos.model`, `chinese_pos.model`,
  `korean_pos.model`, `english_pos.model`) with `segment --pos` /
  `evaluate --pos` (see [Two-Stage Tagging](algorithm/two-stage-tagging.md)
  for the architecture and measured figures; for English specifically,
  read the protocol note above before relying on `english_pos.model`'s
  segmentation quality).
- For **domain-specific** needs, consider [training your own model](training-guide/preparing-corpus.md) or [retraining](training-guide/retraining-models.md) an existing one

## Sample Data

The `resources/` directory also contains sample data used for benchmarking:

- **bocchan.txt** -- 坊っちゃん (Natsume Soseki), ~307 KB. Used by the `segment_long_japanese` benchmarks and differential tests.
- **wagahaiwa_nekodearu.txt** -- 吾輩は猫である (Natsume Soseki), ~1.1 MB, Aozora Bunko.
- **mujeong.txt** -- 무정 (Yi Kwang-su, 1917), ~786 KB, ko.wikisource.
- **rulin_waishi.txt** -- 儒林外史 (Wu Jingzi), ~985 KB, zh.wikisource.
- **pride_and_prejudice.txt** -- Pride and Prejudice (Jane Austen), ~688 KB, Project Gutenberg eBook #1342 (header, footer, and illustration captions stripped; one paragraph per line).

The `wagahaiwa_nekodearu.txt`/`mujeong.txt`/`rulin_waishi.txt` trio is
byte-identical to the corpora of the external
[tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
harness and feeds the `external_corpus` benchmark group (see
[Benchmarking](advanced/benchmarking.md)); `pride_and_prejudice.txt` feeds
the same benchmark group's English cases but has no counterpart in that
external harness yet. All are public domain.
