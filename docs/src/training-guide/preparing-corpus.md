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

```sh
git clone https://github.com/UniversalDependencies/UD_Japanese-GSD
```

## Corpus for Word Segmentation

For word segmentation (AdaBoost), the corpus must be a plain text file with:

- **One sentence per line**
- **Words separated by spaces**

```text
太郎 は 走っ た 。
Litsea は コンパクト な 単語 分割 ソフトウェア です 。
```

### Convert CoNLL-U to Word Segmentation Corpus

Use `litsea convert-conllu` (without `--pos`) to extract space-separated words from a CoNLL-U file:

```sh
litsea convert-conllu UD_Japanese-GSD/ja_gsd-ud-train.conllu corpus.txt
```

This strips POS tags and outputs only space-separated words, one sentence per line.

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

Use `litsea convert-conllu --pos` to convert a CoNLL-U file to the `word/POS` format:

```sh
litsea convert-conllu --pos UD_Japanese-GSD/ja_gsd-ud-train.conllu pos_corpus.txt
```

Multi-word tokens and empty nodes are automatically handled during conversion.

## Automated Corpus Preparation

Litsea includes a helper script in the `scripts/` directory that automates the UD Treebank download and conversion:

```sh
bash scripts/corpus.sh -l ja -c corpus.txt -p pos_corpus.txt
```

This script:

1. Clones the appropriate UD Treebank repository for the specified language
2. Converts the training data to word segmentation corpus format (`corpus.txt`)
3. Converts the training data to POS corpus format (`pos_corpus.txt`)

Supported languages: `ja` (Japanese), `ko` (Korean), `zh` (Chinese).

## Corpus Quality Tips

- **Diversity** -- Include text from various domains (news, literature, web, etc.)
- **Size** -- Larger corpora generally produce better models, but diminishing returns apply
- **Consistency** -- Ensure consistent tokenization throughout the corpus
- **Deduplication** -- Remove duplicate sentences to avoid bias
- **Cleaning** -- Remove HTML tags, special formatting, and non-text content
