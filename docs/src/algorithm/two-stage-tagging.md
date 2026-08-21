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
| Korean | 99.88% | 93.95% | 4.21M chars/s |
| English | 98.30% | 90.55% | 7.32M chars/s |

Two observations worth keeping in mind:

- **Candidate-tag restriction is a real quality lever.** Restricting a
  known word's candidates to what was actually observed removes most
  opportunities to pick an implausible tag, which offsets scoring a
  smaller feature set per word.
- **Throughput varies by lexicon coverage.** Korean's high held-out
  unknown-word rate means a larger share of its words pay stage 2's
  full-class fallback rather than the cheap dominance-skip or
  candidate-masked paths. English sits at the other extreme since #198:
  ~43% of its tokens are spaces, every one of which is a
  single-candidate lexicon hit that skips the classifier entirely.

### Corpus protocol matters more than anything else here (issue #198)

The stage-1 classifier is only as good as the text it trained on. Until
issue #198 the two-stage extractor read a space-separated `word/POS`
corpus and reconstructed each sentence by concatenating word forms with
**no separator** — correct for Japanese and Chinese, whose real text has
no spaces, but for Korean and English it threw away the single strongest
boundary signal their input carries.

Training on the space-preserving corpus instead (`extract --pos --format
tsv`) moved both languages onto their dedicated segmentation models'
level:

| Language | Word F1 before | Word F1 after | Tagged F1 before | Tagged F1 after |
|----------|----------------|---------------|------------------|-----------------|
| Korean | 94.01% | **99.88%** | 83.20% | **93.95%** |
| English | 77.55% | **98.30%** | 69.89% | **90.55%** |

(The "before" column is the real-world/spaced measurement from issue #196,
not the unspaced-protocol figure, so it is an apples-to-apples comparison
of what a user actually got.) Korean now sits 0.03pt from `korean.model`'s
99.91% and English 0.01pt from `english.model`'s 98.31%.

The unspaced corpus caused **two** distinct train/inference mismatches,
and both are now fixed:

1. **Stage 1** never saw the space characters that mark most word
   boundaries in these languages.
2. **Stage 2** was affected too, less obviously: its context features
   (`L1`-`L3` / `R1`-`R3` and `cl1`-`cl3` / `cr1`-`cr3`) read the
   surrounding characters, and at inference a word's neighbour is usually
   a space — but in unspaced training it was the next word's character
   instead. Since spaces are ~43% of tokens, essentially every word was
   affected.

Whitespace tokens get no stage-2 training row (they would be ~43% of rows
for one degenerate `X` class) but do get a lexicon entry, which makes them
single-candidate and therefore tagged `X` through the fixed-tag path
without invoking the classifier — deterministic, and cheaper than the
full-argmax guess the previous models fell back to.

Japanese and Chinese are unaffected by all of this: their text has no
spaces, so the space-separated corpus already matched their real input.

## Choosing a stage-2 feature set

Stage 2's word-level tagger can be extracted with three feature sets
(`--stage2-features` on `litsea extract --pos`; see [Extracting
Features](../training-guide/extracting-features.md)), trading tagging
quality for throughput. Segmentation quality is unaffected -- it is
decided entirely by stage 1. The figures below are at 50 epochs, matching
the bundled models.

| Feature set | Chinese Tagged F1 | Korean Tagged F1 | English Tagged F1 |
|-------------|-----|-----|-----|
| `fast` (default) | 81.33% | 90.62% | 88.68% |
| `balanced` | 82.29% | 92.95% | 88.66% |
| `full` | 82.96% | **93.33%** | **90.43%** |

(Korean and English were re-swept on the space-preserving corpus for issue #198; those two columns are dev-split figures from that sweep, while the
Chinese column predates it. Korean's winner moved from `balanced` to
`full` as a result.)

For Japanese, `fast` alone reaches 92.95% tagged F1, so the bundled
`japanese_pos.model` uses it. For Chinese, `balanced` gives most of
`full`'s gain (82.29% vs. 82.96%) at meaningfully better throughput, so
`chinese_pos.model` uses `balanced` rather than `full`. For Korean and
English on the space-preserving corpus, `full` is the clear winner
(Korean 93.33% vs. 92.95% for `balanced`; English 90.43% vs. ~88.7% for
either alternative), so both `korean_pos.model` and `english_pos.model`
use `full`. Retraining with a different
set is a matter of re-running `extract --pos --stage2-features
<set>` + `train --pos`; there is no need to change any other part of
the pipeline.
