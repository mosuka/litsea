# evaluate

Evaluate a trained model against a held-out gold corpus and print quality
metrics. Unlike the in-sample metrics printed by `train`, this measures
quality on text the model has never seen.

A **gold corpus** is a text file containing the correct answers: one
sentence per line, already segmented into the correct tokens by human
annotation (the same file formats used for training corpora — space- or
tab-separated tokens, or `word/POS` with `--pos`). "Gold" refers to the
gold standard the model's output is judged against. For a meaningful
**held-out** evaluation it must contain sentences that were *not* used to
train the model — the bundled files in `resources/eval/` are the UD GSD
**test** splits, while the bundled models are trained on the train splits.

## Usage

```sh
litsea evaluate [OPTIONS] <MODEL_URI> <GOLD_FILE>
```

## Arguments

| Argument | Description |
|----------|------------|
| `MODEL_URI` | Path or URL to the trained model file. Supports: local file paths, `file://`, `http://`, `https://` |
| `GOLD_FILE` | Path to the gold corpus (one sentence per line) |

## Options

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | Language of the model and gold corpus. Accepts: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |
| `--pos` | off | Evaluate segmentation + POS tagging. The gold corpus must then be in `word/POS` format. Requires a [two-stage](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1) model (`train --pos`) |
| `--format <FORMAT>` | `space` | Gold corpus format: `space` (space-separated tokens) or `tsv` (tab-separated tokens; a token may be a literal space, as in the Korean space-preserving corpus). Ignored with `--pos` |

## Metrics

Two token sequences are compared for every sentence:

- **Gold tokens** -- the reference segmentation from the gold corpus: the
  human-annotated correct answer (here, the tokenization of the UD GSD
  treebank test split). The evaluated sentence text is reconstructed by
  concatenating them.
- **Predicted tokens** -- what the model produces when the reconstructed
  sentence text is fed to `segment` (or `segment --pos`), exactly as a
  user would at inference time.

Predicted and gold tokens are matched by exact character-offset spans over
the reconstructed sentence. Pure-whitespace tokens are excluded from
scoring, so the Korean space-preserving protocol does not inflate the
numbers.

| Metric | Measures | A low value means |
|--------|----------|-------------------|
| Word Precision | Of the **predicted** words, the fraction that exactly matches a gold word (both ends correct) | many spurious words: over-segmentation or wrongly merged words |
| Word Recall | Of the **gold** words, the fraction recovered exactly | many gold words missed |
| Word F1 | Harmonic mean of word precision and recall | overall segmentation quality |
| Boundary Precision | Of the **predicted** word-start positions, the fraction that is a gold boundary | many false boundaries (over-segmentation) |
| Boundary Recall | Of the **gold** word-start positions, the fraction found | many missed boundaries (under-segmentation) |
| Boundary F1 | Harmonic mean of boundary precision and recall | overall boundary quality |
| Tagged Word Precision / Recall / F1 (`--pos`) | Like the word metrics, but the predicted POS tag must also match | correct spans carrying wrong tags |

A word counts as correct only when **both** of its boundaries are correct,
so word metrics are always at least as strict as boundary metrics — a
single misplaced boundary invalidates the two words on either side of it.
`Sentences` is the number of evaluated (non-empty) gold sentences.

## Examples

Reproduce the documented held-out figures with the bundled gold data
(`resources/eval/`, converted from the UD GSD test splits):

```sh
litsea evaluate -l japanese models/japanese.model resources/eval/japanese_gsd_test.txt
litsea evaluate -l korean --format tsv models/korean.model resources/eval/korean_gsd_test.tsv
litsea evaluate -l chinese models/chinese.model resources/eval/chinese_gsd_test.txt
litsea evaluate --pos -l japanese models/japanese_pos.model resources/eval/japanese_gsd_test_pos.txt
```

Output:

```text
Evaluation Metrics:
  Sentences: 543
  Word Precision: 96.73%
  Word Recall: 96.66%
  Word F1: 96.70%
  Boundary Precision: 98.63%
  Boundary Recall: 98.56%
  Boundary F1: 98.59%
```
