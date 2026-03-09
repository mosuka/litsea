# Preparing a Corpus

A good training corpus is essential for model accuracy. This guide explains how to prepare one using [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks.

## Data Source: UD Treebanks

Litsea uses UD Treebanks as the data source for both word segmentation and POS tagging. UD Treebanks provide high-quality, manually annotated data in CoNLL-U format for many languages.

### Available Treebanks

| Language | Treebank | Repository |
|----------|----------|------------|
| Japanese | UD Japanese-GSD | `UD_Japanese-GSD` |
| Chinese | UD Chinese-GSD | `UD_Chinese-GSD` |
| Korean | UD Korean-GSD | `UD_Korean-GSD` |

### Step 1: Download a UD Treebank

Use `scripts/download_udtreebank.sh` to download a UD Treebank. It prints the path to the training CoNLL-U file to stdout:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
```

Supported languages: `ja` (Japanese, default), `ko` (Korean), `zh` (Chinese). Use `-o` to specify the output directory (default: current directory).

## Corpus for Word Segmentation

For word segmentation (AdaBoost), the corpus must be a plain text file with:

- **One sentence per line**
- **Words separated by spaces**

```text
太郎 は 走っ た 。
Litsea は コンパクト な 単語 分割 ソフトウェア です 。
```

### Convert CoNLL-U to Word Segmentation Corpus

Use `scripts/corpus_udtreebank.sh` to convert a CoNLL-U file to corpus format:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
```

This converts the CoNLL-U data into space-separated words (one sentence per line).

## Corpus for POS Tagging

For POS tagging (Averaged Perceptron), each word must be annotated with its POS tag.

### POS Corpus Format

Each line represents one sentence, with words annotated as `word/POS` pairs separated by spaces:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
Litsea/PROPN は/ADP 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

The POS tags follow the [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tagset with 17 categories: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X.

### Convert CoNLL-U to POS Corpus

Use `scripts/corpus_udtreebank.sh` with the `-p` flag to produce a POS corpus:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
```

Multi-word tokens and empty nodes are automatically handled during conversion.

## Automated Corpus Preparation

Litsea includes helper scripts in the `scripts/` directory that automate the UD Treebank download and conversion:

- **`scripts/download_udtreebank.sh`** -- Downloads a UD Treebank and prints the path to the training CoNLL-U file
- **`scripts/corpus_udtreebank.sh`** -- Converts a CoNLL-U file to Litsea corpus format

```sh
# Download UD Treebank and get CoNLL-U file path
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)

# Generate word segmentation corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt

# Generate POS corpus
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
```

Supported languages for `download_udtreebank.sh`: `ja` (Japanese, default), `ko` (Korean), `zh` (Chinese).

## Corpus Quality Tips

- **Diversity** -- Include text from various domains (news, literature, web, etc.)
- **Size** -- Larger corpora generally produce better models, but diminishing returns apply
- **Consistency** -- Ensure consistent tokenization throughout the corpus
- **Deduplication** -- Remove duplicate sentences to avoid bias
- **Cleaning** -- Remove HTML tags, special formatting, and non-text content
