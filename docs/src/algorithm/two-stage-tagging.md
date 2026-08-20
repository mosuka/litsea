# Two-Stage Tagging

Litsea's `segment_with_pos()` is backed by the **two-stage** model
(`--pos`, `Segmenter::with_two_stage_learner`, issue #147). The model
file's `litsea-two-stage v1` header identifies the format (see [Model File
Format](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)),
and the method returns `Vec<(String, Upos)>` word/tag pairs.

## How it works

Two-stage tagging never scores POS classes at the character level:

1. **Stage 1** segments with a binary boundary classifier -- structurally
   the same scalar-weight AdaBoost format `segment()` uses, so it runs at
   the same speed.
2. **Stage 2** tags each *word* stage 1 produces, using a candidate-tag
   lexicon built from the training corpus plus a word-level classifier
   (see [Model File
   Format](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)):
   - A surface with a single observed tag, or one that covers at least the
     `dominance` fraction of its training occurrences, is tagged with **no
     classifier call at all**.
   - An ambiguous known surface is scored only over its observed candidate
     tags (a masked argmax), not all ~18 classes.
   - An unknown surface falls back to the full-class argmax.

Since a word averages more characters than one, and most words are either
unambiguous or skip the classifier entirely, stage 2's total cost is a
small fraction of per-character multiclass scoring. This is what makes the
POS path nearly as fast as plain segmentation: the historical joint
architecture (removed in favor of two-stage) scored every one of the ~18
UPOS classes at every character position and ran 1.8-2.8x slower on the
same corpora.

## A methodology note: use enough training epochs

An epoch sweep (10 to 150 epochs) during the two-stage rollout showed that
model quality keeps improving well past the 10-epoch convention older
models were trained with, and plateaus at around 50 epochs on the UD GSD
training sets. In particular, stage 1's boundary-only features need more
epochs than richer per-character feature sets to converge on the same
corpus. **The bundled two-stage models use 50 epochs**; when retraining, a
one-shot low-epoch run will understate the quality the architecture can
reach.

## Measured quality and throughput

Held-out figures are from `litsea evaluate --pos` on the UD GSD/EWT test
splits (see [Pre-trained Models](../pre-trained-models.md)). Throughput is
`cargo bench -- external_corpus` on this project's development machine
(not dedicated, idle hardware); run-to-run spread is noticeable -- see
[Benchmarking](../advanced/benchmarking.md) for the methodology and its
limits.

| Language | Word F1 | Tagged F1 | Throughput |
|----------|---------|-----------|------------|
| Japanese | 96.78% | 92.95% | 4.38M chars/s |
| Chinese | 90.82% | 82.29% | 3.38M chars/s |
| Korean (unspaced protocol) | 83.24% | 78.86% | 4.54M chars/s |
| Korean (real-world/spaced) | 94.01% | 83.20% | -- |
| English (unspaced protocol) | 70.33% | 65.83% | 2.05M chars/s |
| English (real-world/spaced) | 77.55% | 69.89% | -- |

Two observations worth keeping in mind:

- **Candidate-tag restriction is a real quality lever.** Restricting a
  known word's candidates to what was actually observed removes most
  opportunities to pick an implausible tag, which offsets scoring a
  smaller feature set per word.
- **Throughput varies by lexicon coverage.** Korean's 34.5% held-out
  unknown-word rate means a larger share of its words pay stage 2's
  full-class fallback rather than the cheap dominance-skip or
  candidate-masked paths.

**Japanese and Chinese have no "unspaced protocol" vs. "real-world" split**
because their real text has no spaces to begin with -- the single row
above already *is* their real-world number. Korean and English are
space-delimited languages evaluated on two different protocols: the
"unspaced protocol" row measures the model on the same unspaced text it
was trained on (not a train/inference mismatch -- `evaluate --pos` scores
it consistently either way); the "real-world/spaced" row (issue #196)
reconstructs the model's actual spaced input from a space-preserving POS
gold and measures what `segment --pos` really produces, without changing
training. Both rows are the *same model* -- these two-stage POS models are
trained on unspaced text regardless of protocol shown, since the two-stage
training pipeline itself has no space-preserving corpus format yet.

**English's Word F1 is much lower than Korean's, and this is not a bug.**
The clearest apples-to-apples comparison is each language's real-world
two-stage number against its own dedicated space-preserving *segmentation*
model -- the ceiling a model trained directly on spaced text reaches for
that language: Korean's real-world POS quality (94.01%) sits only 5.9pt
below `korean.model`'s 99.91%, while English's (77.55%) sits 20.8pt below
`english.model`'s 98.31%. Both `korean_pos.model` and `english_pos.model`
are trained on unspaced text (the two-stage pipeline has no
space-preserving corpus format), so this gap is the cost of that unspaced
training protocol -- and it is roughly 3.5x larger for English than for
Korean, because Korean's agglutinative particles and verb endings leave
much stronger boundary cues even with spaces removed than English
orthography does. See
[English](../language-support/english.md#english_posmodel) and
[Pre-trained Models](../pre-trained-models.md#english_posmodel) for the
full explanation.

## Choosing a stage-2 feature set

Stage 2's word-level tagger can be extracted with three feature sets
(`--stage2-features` on `litsea extract --pos`; see [Extracting
Features](../training-guide/extracting-features.md)), trading tagging
quality for throughput. Segmentation quality is unaffected -- it is
decided entirely by stage 1. The figures below are at 50 epochs, matching
the bundled models.

| Feature set | Chinese Tagged F1 | Korean Tagged F1 | English Tagged F1 |
|-------------|-----|-----|-----|
| `fast` (default) | 81.33% | 77.42% | 64.88% |
| `balanced` | 82.29% | 78.86% | 64.82% |
| `full` | 82.96% | 78.88% | 65.70% |

For Japanese, `fast` alone reaches 92.95% tagged F1, so the bundled
`japanese_pos.model` uses it. For Chinese, `balanced` gives most of
`full`'s gain (82.29% vs. 82.96%) at meaningfully better throughput, so
`chinese_pos.model` uses `balanced` rather than `full`. For Korean,
`full` adds essentially nothing over `balanced` (78.88% vs. 78.86%), so
`korean_pos.model` also uses `balanced`. For English, `full` is the clear
winner by a full point over the other two (65.70% vs. ~64.8%), so
`english_pos.model` uses `full`. Retraining with a different
set is a matter of re-running `extract --pos --stage2-features
<set>` + `train --pos`; there is no need to change any other part of
the pipeline.
