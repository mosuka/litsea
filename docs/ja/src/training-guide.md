# トレーニングガイド

このガイドでは、Litsea で独自の単語分割モデルと品詞推定モデルを学習する手順を説明します。

Litsea は 2 つの学習ワークフローをサポートしています:

## 単語分割（AdaBoost）

1. スペース区切りの[コーパスを準備](training-guide/preparing-corpus.md)する
2. コーパスから[特徴量を抽出](training-guide/extracting-features.md)する
3. AdaBoost で[モデルを訓練](training-guide/training-models.md)する

## 品詞推定（Averaged Perceptron）

1. [Universal Dependencies](https://universaldependencies.org/) Treebank をダウンロード
2. CoNLL-U 形式を変換: `litsea convert-conllu input.conllu corpus.txt`
3. [品詞付き特徴量を抽出](training-guide/extracting-features.md): `litsea extract --pos -l japanese corpus.txt features.txt`
4. [POS モデルを訓練](training-guide/training-models.md): `litsea train --pos --num-epochs 10 features.txt model.txt`

## その他のトピック

- [モデルの評価](training-guide/evaluating-models.md) -- モデル品質の評価
- [モデルの再訓練](training-guide/retraining-models.md) -- 既存モデルの微調整
