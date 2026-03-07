# Summary

[はじめに](README.md)

# 導入

- [導入](getting-started.md)
  - [インストール](getting-started/installation.md)
  - [クイックスタート](getting-started/quick-start.md)
- [アーキテクチャ概要](architecture/overview.md)
  - [ワークスペース構成](architecture/workspace-structure.md)
  - [モジュール設計](architecture/module-design.md)

# コアコンセプト

- [AdaBoost 二値分類](algorithm/adaboost.md)
- [Averaged Perceptron](algorithm/averaged-perceptron.md)
- [特徴量抽出](algorithm/feature-extraction.md)
- [文字種分類](algorithm/character-type-classification.md)
- [予測パイプライン](algorithm/prediction-pipeline.md)

# 言語サポート

- [言語サポート](language-support/overview.md)
  - [日本語](language-support/japanese.md)
  - [中国語](language-support/chinese.md)
  - [韓国語](language-support/korean.md)
  - [新しい言語の追加](language-support/adding-a-new-language.md)

# litsea (ライブラリ)

- [ライブラリ概要](library-api.md)
  - [Segmenter](library-api/segmenter.md)
  - [Extractor](library-api/extractor.md)
  - [Trainer](library-api/trainer.md)
  - [AdaBoost](library-api/adaboost.md)
  - [Averaged Perceptron](library-api/averaged-perceptron.md)
  - [UPOS](library-api/upos.md)
  - [CoNLL-U コンバーター](library-api/conllu.md)
  - [Language](library-api/language.md)

# litsea-cli (CLI)

- [CLI 概要](cli-reference.md)
  - [extract](cli-reference/extract.md)
  - [train](cli-reference/train.md)
  - [segment](cli-reference/segment.md)
  - [convert-conllu](cli-reference/convert-conllu.md)
  - [split-sentences](cli-reference/split-sentences.md)

# トレーニングガイド

- [トレーニングガイド](training-guide.md)
  - [コーパスの準備](training-guide/preparing-corpus.md)
  - [特徴量の抽出](training-guide/extracting-features.md)
  - [モデルの訓練](training-guide/training-models.md)
  - [モデルの評価](training-guide/evaluating-models.md)
  - [モデルの再訓練](training-guide/retraining-models.md)

# 上級トピック

- [モデルファイル形式](advanced/model-file-format.md)
- [リモートモデルの読み込み](advanced/remote-model-loading.md)
- [ベンチマーク](advanced/benchmarking.md)

# リファレンス

- [事前学習済みモデル](pre-trained-models.md)
- [ライセンス](license.md)
