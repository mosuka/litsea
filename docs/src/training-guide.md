# Training Guide

This guide walks you through training custom word segmentation and POS tagging models with Litsea.

Both workflows use [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks as the data source.

## Word Segmentation (AdaBoost)

1. [Prepare a corpus](training-guide/preparing-corpus.md) from a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt`
2. [Extract features](training-guide/extracting-features.md) from the corpus
3. [Train a model](training-guide/training-models.md) using AdaBoost

## POS Tagging (Averaged Perceptron)

1. [Prepare a POS corpus](training-guide/preparing-corpus.md) from a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt`
2. [Extract POS features](training-guide/extracting-features.md): `litsea extract --pos -l japanese pos_corpus.txt features.txt`
3. [Train a POS model](training-guide/training-models.md): `litsea train --pos --num-epochs 10 features.txt model.model`

## Per-Language Differences

The pipeline (prepare → extract → train), the scripts, and the training
hyperparameters (`-t 0.0001 -i 20000` for the bundled models) are shared by
all three languages. Only two things are language-specific:

1. **The `-l` flag on `extract`** selects the language's character-type
   classification (Japanese 8 types, Chinese 9, Korean 10; Korean uses no
   WC features — see the
   [language support overview](language-support/overview.md)). Models are
   therefore language-specific.
2. **Korean uses the space-preserving TSV corpus format.** Korean is
   written with spaces between eojeol, and those spaces are the strongest
   boundary signal, so the Korean corpus keeps them as tokens
   (`corpus_udtreebank.sh -s` + `litsea extract --format tsv`). Japanese
   and Chinese are written without spaces, so they use the plain
   space-separated format.

```sh
# Japanese / Chinese: space-separated corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
litsea extract -l japanese corpus.txt features.txt

# Korean: space-preserving TSV corpus
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l korean --format tsv corpus.tsv features.txt
```

The `train` step is identical for all three languages.

## Additional Topics

- [Evaluating Models](training-guide/evaluating-models.md) -- assess model quality
- [Retraining Models](training-guide/retraining-models.md) -- fine-tune existing models
