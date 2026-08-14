# Two-Stage vs. Joint Tagging

Litsea offers two architectures for `segment_with_pos()`: the original
**joint** model (`--pos`, `Segmenter::with_pos_learner`) and the newer
**two-stage** model (`--two-stage`, `Segmenter::with_two_stage_learner`,
issue #147). Both implement the exact same `segment_with_pos()` signature
and return the same `Vec<(String, Upos)>` shape; the model file's header
selects the pipeline (see [Model File
Format](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)),
so callers do not need to know which one they loaded.

## Why two architectures exist

The joint model scores **every one of the ~18 UPOS classes at every
character position** (see [Prediction
Pipeline](prediction-pipeline.md#joint-segmentation-and-pos-tagging-segment_with_pos)):
each decision touches an `n_classes`-wide score row instead of the single
scalar `segment()` adds. That is the entire cost difference -- the feature
templates are identical -- and it is the reason the POS path runs several
times slower than plain segmentation.

Two-stage avoids this by **not tagging at the character level at all**:

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
small fraction of scoring every class at every character -- this is the
source of the throughput gain below, not a different feature set or a
smaller model in principle.

## A methodology note: training epochs matter more than architecture

An early comparison of the bundled models -- both trained with the
existing joint convention of 10 epochs -- showed a real-looking 0.90pt
Chinese segmentation gap in joint's favor, which read like an inherent
limit of the two-class boundary classifier. An epoch sweep (10 to 150
epochs, both architectures) showed this was mostly a training artifact,
not an architectural one: **both** models' quality keeps improving well
past 10 epochs, and two-stage's stage 1 needs *more* epochs than the joint
model to reach a comparable point on the same corpus. At *matched* epoch
counts, joint's Chinese segmentation kept a smaller, consistent edge
(roughly 0.5-0.9pt across every epoch count tested from 10 to 150) --
a real, reproducible difference, most likely because the joint model's
per-character tag-dependent features carry more signal per training
example than stage 1's boundary-only features. It is a modest effect, not
a hard ceiling: with enough training (50 epochs, where quality plateaus)
two-stage's Chinese segmentation *exceeds* the published (10-epoch) joint
model's.

The bundled two-stage models below therefore use 50 epochs, found from
this sweep -- not the joint models' original 10-epoch convention, which
was never revisited when two-stage bundling raised the question. The
joint `*_pos.model` figures in [Pre-trained Models](../pre-trained-models.md)
remain their originally published (10-epoch) numbers; they were not
retrained for this comparison, so a matched-epoch re-run of the joint
models would likely also improve their numbers somewhat (the sweep showed
joint's Chinese quality still rising at 100 epochs). The comparison below
is therefore "the two-stage models as bundled" vs. "the joint models as
currently published," not "the best achievable accuracy of each
architecture" -- a caveat worth keeping in mind when reading the table.

## Measured comparison

Held-out figures are from `litsea evaluate --pos` on the UD GSD test
splits (see [Pre-trained Models](../pre-trained-models.md)). Throughput is
`cargo bench -- external_corpus` on this project's development machine
(not dedicated, idle hardware); three runs on the same build gave a
noticeable spread (Japanese 2.65-3.05x, Chinese 2.13-2.44x, Korean
1.75-1.90x) -- see [Benchmarking](../advanced/benchmarking.md) for the
methodology and its limits. The ranges below are the same three runs; the
point figure is their mean.

| Language | Word F1 (joint -> two-stage) | Tagged F1 (joint -> two-stage) | Throughput (joint -> two-stage) |
|----------|-------------------------------|-----------------------------------|------------------------------------|
| Japanese | 96.56% -> 96.78% (+0.22) | 92.51% -> 92.95% (+0.44) | 1.55M -> 4.38M chars/s (~2.8x, range 2.65-3.05x) |
| Chinese | 90.52% -> 90.82% (+0.30) | 81.18% -> 82.29% (+1.11) | 1.49M -> 3.38M chars/s (~2.3x, range 2.13-2.44x) |
| Korean | 80.51% -> 83.24% (+2.73) | 71.03% -> 78.86% (+7.83) | 2.48M -> 4.54M chars/s (~1.8x, range 1.75-1.90x) |

Three things stand out:

- **Both Word F1 and Tagged F1 favor two-stage (as bundled) in every
  language**, sometimes by a wide margin (Korean's +7.83pt tagging). As
  the methodology note above explains, part of this is the extra training
  epochs the bundled two-stage models use relative to the published joint
  models, not purely the architecture -- see that note before treating
  this table as an unconditional architecture ranking.
- **Candidate-tag restriction is a real quality lever, independent of the
  epoch effect.** Even at matched epochs (the sweep in the note above),
  two-stage's tagging quality met or beat joint's: restricting a known
  word's candidates to what was actually observed removes most
  opportunities to pick an implausible tag, which offsets scoring a
  smaller feature set per word.
- **Throughput gains vary (1.8-2.8x) rather than hitting a single fixed
  number.** Korean's smaller speedup traces to its lexicon: 34.5% of
  held-out words are unknown (never seen in training) and always pay
  stage 2's full-class fallback rather than the cheap dominance-skip or
  candidate-masked paths, so a larger share of Korean's words carry the
  full stage-2 cost than in Japanese or Chinese.

## Choosing a stage-2 feature set

Stage 2's word-level tagger can be extracted with three feature sets
(`--stage2-features` on `litsea extract --two-stage`; see [Extracting
Features](../training-guide/extracting-features.md)), trading tagging
quality for throughput. Segmentation quality is unaffected -- it is
decided entirely by stage 1. The figures below are at 50 epochs, matching
the bundled models.

| Feature set | Chinese Tagged F1 | Korean Tagged F1 |
|-------------|-----|-----|
| `fast` (default) | 81.33% | 77.42% |
| `balanced` | 82.29% | 78.86% |
| `full` | 82.96% | 78.88% |

For Japanese, `fast` alone already clears the joint baseline comfortably
(92.95% vs. 92.51%), so the bundled `japanese_two_stage.model` uses it.
For Chinese, `balanced` gives most of `full`'s gain (82.29% vs. 82.96%) at
meaningfully better throughput, so `chinese_two_stage.model` uses
`balanced` rather than `full`. For Korean, `full` adds essentially nothing
over `balanced` (78.88% vs. 78.86%), so `korean_two_stage.model` also uses
`balanced`. Retraining with a different set is a matter of re-running
`extract --two-stage --stage2-features <set>` + `train --two-stage`; there
is no need to change any other part of the pipeline.

## Which one to use

**Prefer the two-stage models for new work.** As bundled, they beat the
currently published joint models on both Word F1 and Tagged F1 in every
language, at 1.8-2.8x the throughput and a smaller file.

Keep the joint (`*_pos.model`) models available for:

- **Compatibility** with existing code built against `with_pos_learner()`
  or the joint model file format; nothing about the two-stage addition
  changes joint model behavior (see the [design
  discussion](https://github.com/mosuka/litsea/issues/147) -- it is purely
  additive).
- **Matched-training-budget Chinese segmentation.** At the same epoch
  count, joint keeps a small, real edge on Chinese segmentation (see the
  methodology note above); the bundled two-stage model already closes
  this with more training, but a from-scratch two-stage retrain at fewer
  epochs would not.
- **A reference implementation.** The joint model's per-character scoring
  is architecturally simpler (no lexicon, no dominance threshold, no
  candidate masking) and is what the two-stage numbers above were
  validated against; keeping it available preserves that baseline for
  future comparisons.

Both are auto-detected by `segment --pos` / `evaluate --pos` from the
model file (see [Model File
Format](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)),
so switching between them is a matter of pointing at a different file, not
a code or flag change.
