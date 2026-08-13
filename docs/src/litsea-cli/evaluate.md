# evaluate

Evaluate a trained model against a held-out gold corpus and print quality
metrics. Unlike the in-sample metrics printed by `train`, this measures
quality on text the model has never seen.

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
| `--pos` | off | Evaluate joint segmentation + POS tagging. The gold corpus must then be in `word/POS` format |
| `--format <FORMAT>` | `space` | Gold corpus format: `space` (space-separated tokens) or `tsv` (tab-separated tokens; a token may be a literal space, as in the Korean space-preserving corpus). Ignored with `--pos` |

## Metrics

Predicted and gold tokens are matched by exact character-offset spans over
the reconstructed sentence (the concatenation of the gold tokens).
Pure-whitespace tokens are excluded from scoring, so the Korean
space-preserving protocol does not inflate the numbers.

- **Word precision / recall / F1** -- exact token matches
- **Boundary precision / recall / F1** -- individual start-of-token
  decisions (excluding the sentence start)
- With `--pos`: **tagged-word precision / recall / F1** -- span **and**
  POS tag both match

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
  Word Precision: 91.50%
  Word Recall: 91.47%
  Word F1: 91.48%
  Boundary Precision: 96.32%
  Boundary Recall: 96.29%
  Boundary F1: 96.31%
```
