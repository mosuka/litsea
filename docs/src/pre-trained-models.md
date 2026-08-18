# Pre-trained Models

Litsea ships with several pre-trained models in the `models/` directory.

## Model Catalog

The word segmentation models are evaluated on the held-out test split of
their training treebank (sentences never seen during training). **Word F1**
scores exact word matches; **Boundary F1** scores individual boundary
decisions. Note that the `train` command prints *in-sample* metrics
(measured on the training data itself), which are higher than these
held-out figures.

**Algorithm note**: `japanese.model`, `chinese.model`, and `korean.model`
are trained as a 2-class (boundary/non-boundary) Averaged Perceptron and
then collapsed to scalar per-feature weights (issue #165) -- the file is
still the plain AdaBoost text format the engine has always loaded, and
`Segmenter::with_learner` / `AdaBoost::load_model_from_path` work
unchanged. The collapse is a lossless transform (see
`scripts/collapse_binary_perceptron.py`'s docstring for the derivation),
not an approximation: a perceptron trained this way reaches substantially
higher held-out quality than AdaBoost's presence-stump weak learners on the
same corpus and templates, at the cost of a larger model file (more
distinct features get non-zero weight) and a training procedure that goes
through `train --pos` (see [Training Procedure](#training-procedure)
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

# 3. Train a 2-class Averaged Perceptron. --pos is being reused generically
#    here (PosTrainer treats labels as opaque strings), not for POS tagging.
litsea train --pos --num-epochs <N> <features.txt> <perceptron.model>

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
| Japanese | **96.70%** | 96.33% | ~+45-50% measured end-to-end |
| Chinese | **90.69%** | 90.18% | ~+12% measured end-to-end |

With the inter-eojeol space signal available, Korean's tag features
contribute nothing, so `korean.model` ships tag-free (and ~22% smaller).
For Japanese and Chinese they still buy 0.37-0.51pt of Word F1, so the
bundled models keep them -- quality stays the default. If your workload
prefers speed, retrain with the same procedure above plus `--tag-free` on
the extract step; the throughput numbers were measured on this project's
development machine with the paired methodology of
[Benchmarking](advanced/benchmarking.md), so expect the ratio, not the
absolute numbers, to carry over.

## POS Tagging Models

In-sample rows are the `train` command's metrics on the training data;
held-out rows are word / tagged-word F1 measured with `litsea evaluate
--pos` on the UD GSD test splits (see
[Evaluating Models](training-guide/evaluating-models.md)). The Korean POS
gold follows the POS pipeline's convention (no space tokens), so it is
evaluated on unspaced text.

### japanese_pos.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Japanese-GSD (7,050 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 98.23% |
| Macro Precision (in-sample) | 96.82% |
| Macro Recall (in-sample) | 93.30% |
| Word F1 (held-out) | 96.56% |
| Tagged Word F1 (held-out) | 92.51% |
| File Size | ~11 MB |

### chinese_pos.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 97.04% |
| Macro Precision (in-sample) | 97.17% |
| Macro Recall (in-sample) | 96.14% |
| Word F1 (held-out) | 90.52% |
| Tagged Word F1 (held-out) | 81.18% |
| File Size | ~19 MB |

### korean_pos.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Korean-GSD (4,400 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 95.14% |
| Macro Precision (in-sample) | 95.00% |
| Macro Recall (in-sample) | 86.15% |
| Word F1 (held-out) | 80.51% |
| Tagged Word F1 (held-out) | 71.03% |
| File Size | ~8.9 MB |

#### Usage

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_pos.model
```

Output:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## Two-Stage POS Tagging Models

The [two-stage architecture](algorithm/two-stage-tagging.md) (issue #147)
segments with a binary boundary classifier and tags each resulting word
through a candidate-tag lexicon plus a word-level tagger, instead of
scoring every UPOS class at every character position. It is additive: the
`*_pos.model` files above are unaffected, and `segment --pos` /
`evaluate --pos` auto-detect either kind from the file. See [Two-Stage vs.
Joint Tagging](algorithm/two-stage-tagging.md) for the architecture and the
per-language recommendation.

In-sample and held-out rows use the same protocol as the joint models
above. "Stage-2 feature set" is the word-level template selection (`fast`,
`balanced`, or `full`; see [Extracting
Features](training-guide/extracting-features.md)) chosen for the bundled
file per language, from the measured tradeoff in [Two-Stage vs. Joint
Tagging](algorithm/two-stage-tagging.md#choosing-a-stage-2-feature-set).
Throughput is from `cargo bench -- external_corpus` on the same corpora as
the [Benchmarking](advanced/benchmarking.md) page, run on the same
development machine as the other benchmark figures on this page (not
dedicated, idle hardware -- see that page's methodology note); the joint
comparison figure is the throughput measured in the same run, not the
`*_pos.model` table above, since bench-to-bench variance on shared hardware
makes a same-run comparison the only fair one.

**Epoch note**: the joint models above are documented at their original
10-epoch training. An epoch sweep during two-stage bundling (10 to 150
epochs) found that stage 1's *segmentation* quality specifically continues
improving well past 10 epochs and plateaus around 50 -- the bundled
two-stage models below use 50 epochs, chosen from that sweep, not the same
10-epoch convention as the joint models. At *matched* epoch counts, the
sweep also found joint's segmentation retains a small, consistent edge
over two-stage for Chinese (roughly 0.5-0.9pt across every epoch count
tested) -- a real, reproducible difference, most likely because the joint
model's per-character tag-dependent features carry more signal per
training example than stage 1's boundary-only features. It does not,
however, rise to a hard ceiling: 50-epoch two-stage training closes it and
the bundled `chinese_two_stage.model` below has a higher held-out Word F1
than the published (10-epoch) `chinese_pos.model`.

### japanese_two_stage.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | UD Japanese-GSD (7,050 sentences) |
| Epochs | 50 |
| Stage-2 Feature Set | `fast` |
| Word F1 (held-out) | 96.78% (joint: 96.56%) |
| Tagged Word F1 (held-out) | 92.95% (joint: 92.51%) |
| Throughput vs. joint | ~2.8x (3 runs: 2.65-3.05x) |
| File Size | ~5.4 MB |

### chinese_two_stage.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 50 |
| Stage-2 Feature Set | `balanced` |
| Word F1 (held-out) | 90.82% (joint: 90.52%) |
| Tagged Word F1 (held-out) | 82.29% (joint: 81.18%) |
| Throughput vs. joint | ~2.3x (3 runs: 2.13-2.44x) |
| File Size | ~8.0 MB |

### korean_two_stage.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | UD Korean-GSD (4,400 sentences, unspaced word/POS protocol -- see the note below) |
| Epochs | 50 |
| Stage-2 Feature Set | `balanced` |
| Word F1 (held-out) | 83.24% (joint: 80.51%) |
| Tagged Word F1 (held-out) | 78.86% (joint: 71.03%) |
| Throughput vs. joint | ~1.8x (3 runs: 1.75-1.90x) |
| File Size | ~5.0 MB |

Korean's smaller speedup traces to its lexicon: held-out text is 34.5%
unknown words (surfaces never seen in training), and unknown words always
take the full stage-2 classifier fallback rather than the cheap
dominance-skip or candidate-masked paths, so a larger share of Korean's
words pay the full stage-2 cost than in Japanese or Chinese.

**Korean protocol note**: `korean_two_stage.model` is trained on the same
unspaced `word/POS` corpus as `korean_pos.model`, *not* the
space-preserving TSV corpus `korean.model` uses (issue #152). The two-stage
extractor takes a single corpus for both stages, and building a combined
space-preserving + POS-tagged format is a separate feature not yet
implemented; the numbers above are therefore comparable to `korean_pos.model`
but not to `korean.model`'s 99.91% (a different corpus and protocol
entirely, not a stronger or weaker two-stage result).

#### Usage

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_two_stage.model
```

The output is identical in shape to the joint model's:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## Choosing a Model

- For **Japanese**, use `japanese.model` for the best accuracy, or `RWCP.model` for compatibility with the original TinySegmenter
- For **Chinese**, use `chinese.model`
- For **Korean**, use `korean.model`
- For **POS tagging**, prefer the **two-stage** models
  (`japanese_two_stage.model`, `chinese_two_stage.model`,
  `korean_two_stage.model`) -- 1.8-2.8x the throughput and a smaller file,
  at Word F1 and Tagged F1 that both beat the currently published joint
  models in every language. Keep the joint `*_pos.model` files available
  for compatibility with any code already built against `with_pos_learner`
  / the joint model files, and as the simpler reference implementation the
  two-stage numbers above were measured against. At *matched* training
  epochs the two architectures are closer, with joint retaining a small
  edge on Chinese segmentation specifically (see the epoch note above and
  [Two-Stage vs. Joint Tagging](algorithm/two-stage-tagging.md) for the
  full comparison and methodology).
- For **domain-specific** needs, consider [training your own model](training-guide/preparing-corpus.md) or [retraining](training-guide/retraining-models.md) an existing one

## Sample Data

The `resources/` directory also contains sample data used for benchmarking:

- **bocchan.txt** -- 坊っちゃん (Natsume Soseki), ~307 KB. Used by the `segment_long_japanese` benchmarks and differential tests.
- **wagahaiwa_nekodearu.txt** -- 吾輩は猫である (Natsume Soseki), ~1.1 MB, Aozora Bunko.
- **mujeong.txt** -- 무정 (Yi Kwang-su, 1917), ~786 KB, ko.wikisource.
- **rulin_waishi.txt** -- 儒林外史 (Wu Jingzi), ~985 KB, zh.wikisource.

The last three are byte-identical to the corpora of the external
[tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
harness and feed the `external_corpus` benchmark group (see
[Benchmarking](advanced/benchmarking.md)). All are public domain.
