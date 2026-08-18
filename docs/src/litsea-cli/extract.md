# extract

Extract features from a corpus file for model training.

## Usage

```sh
litsea extract [OPTIONS] <CORPUS_FILE> <FEATURES_FILE>
```

## Arguments

| Argument | Description |
|----------|------------|
| `CORPUS_FILE` | Path to the input corpus file (words separated by spaces, one sentence per line) |
| `FEATURES_FILE` | Path to the output features file |

## Options

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | Language for character type classification. Accepts: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |
| `--pos` | off | Enable POS (Part-of-Speech) feature extraction mode. Requires a POS corpus as input |
| `--format <FORMAT>` | `space` | Corpus format: `space` (space-separated words) or `tsv` (tab-separated tokens; a token may be a literal space, preserving the original spacing). `tsv` cannot be combined with `--pos` or `--two-stage` |
| `--two-stage` | off | Extract [two-stage](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1) training features instead of joint POS features. Requires a POS corpus as input; cannot be combined with `--pos` |
| `--stage2-features <SET>` | `fast` | Stage-2 word-feature set for `--two-stage`: `full` (best quality), `balanced`, or `fast` (best throughput) |
| `--tag-free` | off | Exclude the 16 tag-dependent feature templates (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`) so the trained model is pointwise and `segment()` skips its sequential scoring pass (issue #183; used for the bundled `korean.model` -- see [Tag-Free (Pointwise) Models](../pre-trained-models.md#tag-free-pointwise-models) for the per-language quality/speed trade-off). Composable with `--format tsv`; cannot be combined with `--pos` or `--two-stage` |

## Corpus Format

The input corpus must have words separated by spaces, one sentence per line:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

### TSV Corpus Format (`--format tsv`)

With `--format tsv`, tokens are separated by tab characters and a token may
be a literal space `" "`. This preserves the original spacing of the
sentence in the training text, which is essential for languages like Korean
where spaces mark most word boundaries (see
[Korean](../language-support/korean.md)). Generate such a corpus from a UD
Treebank with `corpus_udtreebank.sh -s`:

```sh
litsea extract -l korean --format tsv ./ko_corpus.tsv ./ko_features.txt
```

## Output Format

The features file contains one line per character position. For the corpus line `これ は テスト です 。`, the first two lines are:

```text
-1	BC1:OI	BC2:II	BC3:II	BP1:UU	BP2:UU	BQ1:UOI	BQ2:UII	BQ3:UOI	BQ4:UII	BW1:B1こ	BW2:これ	...
1	BC1:II	BC2:II	BC3:IK	BP1:UU	BP2:UO	BQ1:UII	BQ2:UII	BQ3:OII	BQ4:OII	BW1:これ	BW2:れは	...
```

- `1` = word boundary
- `-1` = non-boundary
- Features are written tab-separated in alphabetically sorted order, so each line starts with the `BC1:` feature rather than following the template definition order

## Examples

```sh
# Japanese
litsea extract -l japanese ./corpus.txt ./features.txt

# Chinese
litsea extract -l zh ./corpus_zh.txt ./features_zh.txt

# Korean
litsea extract -l ko ./corpus_ko.txt ./features_ko.txt
```

Output to stderr on success:

```text
Feature extraction completed successfully.
```

## POS Feature Extraction

When the `--pos` flag is specified, `extract` expects a **POS corpus** instead of a plain word-separated corpus. Each line contains words annotated with UPOS tags in the format `word/POS`:

### POS Corpus Format

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

### POS Feature Output Format

In POS mode, the label column uses segment labels (`B-NOUN`, `B-VERB`, ..., `B-X`, `O`) instead of binary `1`/`-1`. Features are again written tab-separated in alphabetically sorted order:

```text
B-PRON	BC1:OO	BC2:OI	BC3:II	BP1:UU	BP2:UU	BQ1:UOO	BQ2:UOI	BQ3:UOO	BQ4:UOI	BW1:B2B1	...
O	BC1:OI	BC2:II	BC3:II	BP1:UU	BP2:UU	BQ1:UOI	BQ2:UII	BQ3:UOI	BQ4:UII	BW1:B1こ	...
```

### POS Extraction Example

```sh
litsea extract --pos -l japanese ./pos_corpus.txt ./pos_features.txt
```

## Two-Stage Feature Extraction

With `--two-stage`, `extract` reads the same POS corpus format as `--pos`
(`word/POS word/POS ...`) but writes **three** files derived from
`FEATURES_FILE` as a prefix, for the [two-stage segmentation + POS tagging
architecture](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1):

| File | Content |
|------|---------|
| `{FEATURES_FILE}.stage1` | Boundary features, one row per character position, label `B` or `O` (the same feature templates as `--pos`) |
| `{FEATURES_FILE}.stage2` | Word-level features, one row per word, label a UPOS tag; which templates are written is controlled by `--stage2-features` |
| `{FEATURES_FILE}.lexicon` | The candidate-tag lexicon: `surface\tTAG:count[,TAG:count...]`, most-frequent-first |

Pass the same prefix to `litsea train --two-stage`:

```sh
litsea extract --two-stage -l japanese ./pos_corpus.txt ./two_stage_features
# writes ./two_stage_features.stage1, .stage2, .lexicon
```
