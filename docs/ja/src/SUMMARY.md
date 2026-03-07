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

- [ライブラリ概要](litsea.md)
  - [Segmenter](litsea/segmenter.md)
  - [Extractor](litsea/extractor.md)
  - [Trainer](litsea/trainer.md)
  - [AdaBoost](litsea/adaboost.md)
  - [Averaged Perceptron](litsea/averaged-perceptron.md)
  - [UPOS](litsea/upos.md)
  - [CoNLL-U コンバーター](litsea/conllu.md)
  - [Language](litsea/language.md)

# litsea-cli (CLI)

- [CLI 概要](litsea-cli.md)
  - [extract](litsea-cli/extract.md)
  - [train](litsea-cli/train.md)
  - [segment](litsea-cli/segment.md)
  - [convert-conllu](litsea-cli/convert-conllu.md)
  - [split-sentences](litsea-cli/split-sentences.md)

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
