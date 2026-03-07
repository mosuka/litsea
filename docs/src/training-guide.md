# Training Guide

This guide walks you through training custom word segmentation and POS tagging models with Litsea.

Both workflows use [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks as the data source.

## Word Segmentation (AdaBoost)

1. Download a UD Treebank and [prepare a corpus](training-guide/preparing-corpus.md): `litsea convert-conllu input.conllu corpus.txt`
2. [Extract features](training-guide/extracting-features.md) from the corpus
3. [Train a model](training-guide/training-models.md) using AdaBoost

## POS Tagging (Averaged Perceptron)

1. Download a UD Treebank and [prepare a POS corpus](training-guide/preparing-corpus.md): `litsea convert-conllu --pos input.conllu corpus.txt`
2. [Extract POS features](training-guide/extracting-features.md): `litsea extract --pos -l japanese corpus.txt features.txt`
3. [Train a POS model](training-guide/training-models.md): `litsea train --pos --num-epochs 10 features.txt model.txt`

## Additional Topics

- [Evaluating Models](training-guide/evaluating-models.md) -- assess model quality
- [Retraining Models](training-guide/retraining-models.md) -- fine-tune existing models
