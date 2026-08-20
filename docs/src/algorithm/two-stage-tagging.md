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

Held-out figures are from `litsea evaluate --pos` on the UD GSD test
splits (see [Pre-trained Models](../pre-trained-models.md)). Throughput is
`cargo bench -- external_corpus` on this project's development machine
(not dedicated, idle hardware); run-to-run spread is noticeable -- see
[Benchmarking](../advanced/benchmarking.md) for the methodology and its
limits.

| Language | Word F1 | Tagged F1 | Throughput |
|----------|---------|-----------|------------|
| Japanese | 96.78% | 92.95% | 4.38M chars/s |
| Chinese | 90.82% | 82.29% | 3.38M chars/s |
| Korean | 83.24% | 78.86% | 4.54M chars/s |

Two observations worth keeping in mind:

- **Candidate-tag restriction is a real quality lever.** Restricting a
  known word's candidates to what was actually observed removes most
  opportunities to pick an implausible tag, which offsets scoring a
  smaller feature set per word.
- **Throughput varies by lexicon coverage.** Korean's 34.5% held-out
  unknown-word rate means a larger share of its words pay stage 2's
  full-class fallback rather than the cheap dominance-skip or
  candidate-masked paths.

## Choosing a stage-2 feature set

Stage 2's word-level tagger can be extracted with three feature sets
(`--stage2-features` on `litsea extract --pos`; see [Extracting
Features](../training-guide/extracting-features.md)), trading tagging
quality for throughput. Segmentation quality is unaffected -- it is
decided entirely by stage 1. The figures below are at 50 epochs, matching
the bundled models.

| Feature set | Chinese Tagged F1 | Korean Tagged F1 |
|-------------|-----|-----|
| `fast` (default) | 81.33% | 77.42% |
| `balanced` | 82.29% | 78.86% |
| `full` | 82.96% | 78.88% |

For Japanese, `fast` alone reaches 92.95% tagged F1, so the bundled
`japanese_pos.model` uses it. For Chinese, `balanced` gives most of
`full`'s gain (82.29% vs. 82.96%) at meaningfully better throughput, so
`chinese_pos.model` uses `balanced` rather than `full`. For Korean,
`full` adds essentially nothing over `balanced` (78.88% vs. 78.86%), so
`korean_pos.model` also uses `balanced`. Retraining with a different
set is a matter of re-running `extract --pos --stage2-features
<set>` + `train --pos`; there is no need to change any other part of
the pipeline.
