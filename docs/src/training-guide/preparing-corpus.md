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

## Corpus Quality Tips

- **Diversity** -- Include text from various domains (news, literature, web, etc.)
- **Size** -- Larger corpora generally produce better models, but diminishing returns apply
- **Consistency** -- Ensure consistent tokenization throughout the corpus
- **Deduplication** -- Remove duplicate sentences to avoid bias
- **Cleaning** -- Remove HTML tags, special formatting, and non-text content
