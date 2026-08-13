# トレーニングガイド

このガイドでは、Litsea で独自の単語分割モデルと品詞推定モデルを学習する手順を説明します。

両方のワークフローとも、データソースとして [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks を使用します。

## 単語分割（AdaBoost）

1. UD Treebank をダウンロードして[コーパスを準備](training-guide/preparing-corpus.md): `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt`
2. コーパスから[特徴量を抽出](training-guide/extracting-features.md)する
3. AdaBoost で[モデルを訓練](training-guide/training-models.md)する

## 品詞推定（Averaged Perceptron）

1. UD Treebank をダウンロードして[品詞付きコーパスを準備](training-guide/preparing-corpus.md): `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt`
2. [品詞付き特徴量を抽出](training-guide/extracting-features.md): `litsea extract --pos -l japanese pos_corpus.txt features.txt`
3. [POS モデルを訓練](training-guide/training-models.md): `litsea train --pos --num-epochs 10 features.txt model.model`

## 言語ごとの違い

パイプライン（準備 → 抽出 → 学習）、スクリプト、学習ハイパーパラメータ
（同梱モデルは `-t 0.0001 -i 20000`）は 3 言語で共通です。言語固有なのは
次の 2 点だけです:

1. **`extract` の `-l` フラグ** — 言語別の文字種分類を選択します（日本語 8 種、
   中国語 9 種、韓国語 10 種。韓国語は WC 特徴を使いません —
   [言語サポート概要](language-support/overview.md)を参照）。このためモデルは
   言語専用になります
2. **韓国語は空白保持 TSV コーパス形式を使用** — 韓国語は語節（어절）間に空白を
   持つ表記で、空白が最も強い境界シグナルです。そのため韓国語のコーパスは空白を
   トークンとして保持します（`corpus_udtreebank.sh -s` +
   `litsea extract --format tsv`）。日本語・中国語は空白を使わない表記のため、
   従来の空白区切り形式を使います

```sh
# Japanese / Chinese: space-separated corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
litsea extract -l japanese corpus.txt features.txt

# Korean: space-preserving TSV corpus
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l korean --format tsv corpus.tsv features.txt
```

`train` ステップは 3 言語とも同一です。

## その他のトピック

- [モデルの評価](training-guide/evaluating-models.md) -- モデル品質の評価
- [モデルの再訓練](training-guide/retraining-models.md) -- 既存モデルの微調整
