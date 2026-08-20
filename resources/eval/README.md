# Held-out evaluation gold data

Gold-standard segmentation and POS corpora for `litsea evaluate`, converted
from the **test** splits of the Universal Dependencies GSD/EWT treebanks
(the bundled models are trained on the corresponding **train** splits, so
these files are held-out for them).

| File | Format | Source |
|------|--------|--------|
| `japanese_gsd_test.txt` | space-separated words | [UD_Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD) `ja_gsd-ud-test.conllu` |
| `chinese_gsd_test.txt` | space-separated words | [UD_Chinese-GSD](https://github.com/UniversalDependencies/UD_Chinese-GSD) `zh_gsd-ud-test.conllu` |
| `korean_gsd_test.tsv` | space-preserving TSV | [UD_Korean-GSD](https://github.com/UniversalDependencies/UD_Korean-GSD) `ko_gsd-ud-test.conllu` |
| `english_ewt_test.tsv` | space-preserving TSV | [UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) `en_ewt-ud-test.conllu` |
| `*_gsd_test_pos.txt` / `english_ewt_test_pos.txt` | `word/POS` | same sources |
| `korean_gsd_test_pos_spaced.tsv` | space-preserving TSV of `word/POS` tokens | UD_Korean-GSD `ko_gsd-ud-test.conllu` |
| `english_ewt_test_pos_spaced.tsv` | space-preserving TSV of `word/POS` tokens | UD_English-EWT `en_ewt-ud-test.conllu` |

The `*_pos_spaced.tsv` files (issue #196) measure a two-stage POS model's
real-world quality on natural, spaced input via `litsea evaluate --pos
--format tsv`, as opposed to the unspaced protocol the `*_gsd_test_pos.txt`
/ `english_ewt_test_pos.txt` files above measure (matching how the
two-stage models are actually trained). Space tokens in the `_spaced.tsv`
files carry no `/POS` suffix -- CoNLL-U has no UPOS annotation for
whitespace, and they are excluded from scoring regardless (see
[Pre-trained Models](../../docs/src/pre-trained-models.md#korean_posmodel)).
Japanese and Chinese don't need an equivalent: their real text has no
spaces, so their existing `*_gsd_test_pos.txt` already represents
real-world usage.

## License

Unlike the rest of this repository (MIT / Apache-2.0), the files in this
directory are derived from the UD GSD/EWT treebanks and are licensed under
**CC BY-SA 4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>), the
license of the source treebanks. See each treebank's repository for the
full attribution history.

## Regenerating

With a UD GSD/EWT checkout (e.g. via `scripts/download_udtreebank.sh`):

```sh
bash scripts/corpus_udtreebank.sh    <ud>/ja_gsd-ud-test.conllu resources/eval/japanese_gsd_test.txt
bash scripts/corpus_udtreebank.sh    <ud>/zh_gsd-ud-test.conllu resources/eval/chinese_gsd_test.txt
bash scripts/corpus_udtreebank.sh -s <ud>/ko_gsd-ud-test.conllu resources/eval/korean_gsd_test.tsv
bash scripts/corpus_udtreebank.sh -s <ud>/en_ewt-ud-test.conllu resources/eval/english_ewt_test.tsv
bash scripts/corpus_udtreebank.sh -p <ud>/ja_gsd-ud-test.conllu resources/eval/japanese_gsd_test_pos.txt
# (likewise -p for zh / ko / en, the latter writing english_ewt_test_pos.txt)
bash scripts/corpus_udtreebank.sh -p -s <ud>/ko_gsd-ud-test.conllu resources/eval/korean_gsd_test_pos_spaced.tsv
bash scripts/corpus_udtreebank.sh -p -s <ud>/en_ewt-ud-test.conllu resources/eval/english_ewt_test_pos_spaced.tsv
```
