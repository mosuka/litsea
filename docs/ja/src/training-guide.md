# トレーニングガイド

このガイドでは、Litsea で独自の単語分割モデルと品詞推定モデルを学習する手順を説明します。

両方のワークフローとも、データソースとして [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks を使用します。

## 単語分割（AdaBoost）

1. UD Treebank をダウンロードして[コーパスを準備](training-guide/preparing-corpus.md): `litsea convert-conllu input.conllu corpus.txt`
2. コーパスから[特徴量を抽出](training-guide/extracting-features.md)する
3. AdaBoost で[モデルを訓練](training-guide/training-models.md)する

## 品詞推定（Averaged Perceptron）

1. UD Treebank をダウンロードして[品詞付きコーパスを準備](training-guide/preparing-corpus.md): `litsea convert-conllu --pos input.conllu corpus.txt`
2. [品詞付き特徴量を抽出](training-guide/extracting-features.md): `litsea extract --pos -l japanese corpus.txt features.txt`
3. [POS モデルを訓練](training-guide/training-models.md): `litsea train --pos --num-epochs 10 features.txt model.txt`

## その他のトピック

- [モデルの評価](training-guide/evaluating-models.md) -- モデル品質の評価
- [モデルの再訓練](training-guide/retraining-models.md) -- 既存モデルの微調整
