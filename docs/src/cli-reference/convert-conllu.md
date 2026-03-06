# convert-conllu

Convert [CoNLL-U](https://universaldependencies.org/format.html) (Universal Dependencies) format files to Litsea POS corpus format.

## Usage

```sh
litsea convert-conllu <INPUT_FILE> <OUTPUT_FILE>
```

## Arguments

| Argument | Description |
|----------|------------|
| `INPUT_FILE` | Path to the input CoNLL-U file |
| `OUTPUT_FILE` | Path to the output Litsea POS corpus file |

## Input Format (CoNLL-U)

CoNLL-U is a tab-separated format used by [Universal Dependencies](https://universaldependencies.org/) treebanks. Each line represents a token with 10 fields:

```text
# sent_id = example-1
# text = 太郎は花子が読んでいる本を次郎に渡した。
1	太郎	太郎	PROPN	_	_	13	nsubj	_	SpaceAfter=No
2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
3	花子	花子	PROPN	_	_	5	nsubj	_	SpaceAfter=No
4	が	が	ADP	_	_	3	case	_	SpaceAfter=No
5	読ん	読む	VERB	_	_	8	acl	_	SpaceAfter=No
...
```

The relevant fields for conversion are:

- Column 1: `ID` -- token index (integer, range, or decimal)
- Column 2: `FORM` -- word form (surface text)
- Column 4: `UPOS` -- Universal POS tag

## Output Format (Litsea POS Corpus)

Each sentence becomes one line with tokens in `word/POS` format:

```text
太郎/PROPN は/ADP 花子/PROPN が/ADP 読ん/VERB で/SCONJ いる/AUX 本/NOUN を/ADP 次郎/PROPN に/ADP 渡し/VERB た/AUX 。/PUNCT
```

## Handling Special Cases

- **Comment lines** (`# ...`): Skipped
- **Multiword tokens** (range IDs like `1-2`): Skipped (individual tokens are used instead)
- **Empty nodes** (decimal IDs like `1.1`): Skipped
- **Missing UPOS** (`_` in column 4): Token is assigned the `X` tag

## Examples

Convert a UD Japanese-GSD treebank file:

```sh
litsea convert-conllu ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus_pos.txt
```

Full POS training workflow starting from a UD treebank:

```sh
# 1. Convert CoNLL-U to Litsea POS corpus
litsea convert-conllu ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus_pos.txt

# 2. Extract POS features
litsea extract --pos -l japanese ./corpus_pos.txt ./features_pos.txt

# 3. Train a POS model
litsea train --pos -e 10 ./features_pos.txt ./resources/japanese_pos.model
```

## Data Source

Universal Dependencies treebanks are available at [universaldependencies.org](https://universaldependencies.org/). For Japanese, the recommended treebank is [UD_Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD) (7,050 sentences).
