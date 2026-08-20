# Training Guide

This guide walks you through training custom word segmentation and POS tagging models with Litsea.

Both workflows use [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks as the data source.

## Word Segmentation (AdaBoost)

1. [Prepare a corpus](training-guide/preparing-corpus.md) from a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt`
2. [Extract features](training-guide/extracting-features.md) from the corpus
3. [Train a model](training-guide/training-models.md) using AdaBoost

## POS Tagging (Two-Stage)

1. [Prepare a POS corpus](training-guide/preparing-corpus.md) from a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt`
2. [Extract two-stage features](training-guide/extracting-features.md): `litsea extract --pos -l japanese pos_corpus.txt features`
3. [Train a two-stage POS model](training-guide/training-models.md): `litsea train --pos --num-epochs 50 features model.model`

## Per-Language Differences

The pipeline (prepare → extract → train) and the scripts are shared by all
four languages. Only two things are language-specific:

1. **The `-l` flag on `extract`** selects the language's character-type
   classification (Japanese 8 types, Chinese 9, Korean 10, English 7;
   Korean and English use no WC features — see the
   [language support overview](language-support/overview.md)). Models are
   therefore language-specific.
2. **Korean and English use the space-preserving TSV corpus format.**
   Both are written with spaces between words, and those spaces are the
   strongest boundary signal, so their corpora keep them as tokens
   (`corpus_udtreebank.sh -s` + `litsea extract --format tsv`). Japanese
   and Chinese are written without spaces, so they use the plain
   space-separated format.

```sh
# Japanese / Chinese: space-separated corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
litsea extract -l japanese corpus.txt features.txt

# Korean / English: space-preserving TSV corpus
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l korean --format tsv corpus.tsv features.txt
```

The `train` step's command shape is the same for all four languages, but
the actual hyperparameters differ. `-t 0.0001 -i 20000` (see [Training
Models](training-guide/training-models.md)) is a good starting point when
training a plain AdaBoost model from scratch with `litsea train`, but it is
not what the bundled `japanese`/`chinese`/`korean`/`english` models use --
those go through a different procedure with per-language epoch counts and
pruning. See [Training Procedure](pre-trained-models.md#training-procedure)
for the actual recipe.

## Additional Topics

- [Evaluating Models](training-guide/evaluating-models.md) -- assess model quality
- [Retraining Models](training-guide/retraining-models.md) -- fine-tune existing models
