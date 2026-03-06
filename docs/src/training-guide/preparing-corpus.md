# Preparing a Corpus

A good training corpus is essential for model accuracy. This guide explains how to prepare one.

## Corpus Format

The corpus must be a plain text file with:
- **One sentence per line**
- **Words separated by spaces**

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## Automated Corpus Preparation

Litsea includes helper scripts in the `scripts/` directory for building corpora from Wikipedia.

### Step 1: Download Wikipedia Texts

```sh
bash scripts/wikitexts.sh ja   # Japanese
bash scripts/wikitexts.sh ko   # Korean
bash scripts/wikitexts.sh zh   # Chinese
```

This script:
1. Downloads article titles from the Wikipedia API
2. Filters by language-specific criteria
3. Fetches article text
4. Splits into sentences using `litsea split-sentences`

### Step 2: Tokenize with Lindera

```sh
bash scripts/corpus.sh ja ./wikitexts_ja.txt ./corpus_ja.txt
bash scripts/corpus.sh ko ./wikitexts_ko.txt ./corpus_ko.txt
bash scripts/corpus.sh zh ./wikitexts_zh.txt ./corpus_zh.txt
```

This script uses [Lindera](https://github.com/lindera/lindera) with language-specific dictionaries:

| Language | Dictionary | Notes |
|----------|-----------|-------|
| Japanese | UniDic | With compound word filters |
| Korean | ko-dic | Korean dictionary |
| Chinese | CC-CEDICT | Chinese-English dictionary |

The output is in **wakati** format (space-separated tokens), ready for feature extraction.

## Preparing a POS Corpus

For POS (Part-of-Speech) tagging, Litsea uses a different corpus format where each word is annotated with its POS tag.

### POS Corpus Format

Each line represents one sentence, with words annotated as `word/POS` pairs separated by spaces:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
Litsea/PROPN は/ADP 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

The POS tags follow the [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tagset with 17 categories: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X.

### Using UD Treebanks as Data Source

[Universal Dependencies (UD)](https://universaldependencies.org/) provides high-quality treebank data in CoNLL-U format for many languages. Litsea includes a converter to transform CoNLL-U files into the POS corpus format.

#### Step 1: Download a UD Treebank

```sh
git clone https://github.com/UniversalDependencies/UD_Japanese-GSD
```

Available UD Treebanks for supported languages:

| Language | Treebank | Repository |
|----------|----------|------------|
| Japanese | UD Japanese-GSD | `UD_Japanese-GSD` |
| Chinese | UD Chinese-GSD | `UD_Chinese-GSD` |
| Korean | UD Korean-GSD | `UD_Korean-GSD` |

#### Step 2: Convert CoNLL-U to POS Corpus

```sh
litsea convert-conllu UD_Japanese-GSD/ja_gsd-ud-train.conllu corpus.txt
```

This converts the CoNLL-U format into the `word/POS` format that Litsea expects. Multi-word tokens and empty nodes are automatically handled during conversion.

## Corpus Quality Tips

- **Diversity** -- Include text from various domains (news, literature, web, etc.)
- **Size** -- Larger corpora generally produce better models, but diminishing returns apply
- **Consistency** -- Ensure consistent tokenization throughout the corpus
- **Deduplication** -- Remove duplicate sentences to avoid bias
- **Cleaning** -- Remove HTML tags, special formatting, and non-text content
