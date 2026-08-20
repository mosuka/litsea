# トレーニングガイド

このガイドでは、Litsea で独自の単語分割モデルと品詞推定モデルを学習する手順を説明します。

両方のワークフローとも、データソースとして [Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks を使用します。

## 単語分割（AdaBoost）

1. UD Treebank をダウンロードして[コーパスを準備](training-guide/preparing-corpus.md): `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt`
2. コーパスから[特徴量を抽出](training-guide/extracting-features.md)する
3. AdaBoost で[モデルを訓練](training-guide/training-models.md)する

## 品詞推定（二段構成）

1. UD Treebank をダウンロードして[品詞付きコーパスを準備](training-guide/preparing-corpus.md): `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp) && bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt`
2. [二段構成の特徴量を抽出](training-guide/extracting-features.md): `litsea extract --pos -l japanese pos_corpus.txt features`
3. [二段構成の POS モデルを訓練](training-guide/training-models.md): `litsea train --pos --num-epochs 50 features model.model`

## 言語ごとの違い

パイプライン（準備 → 抽出 → 学習）とスクリプトは 4 言語で共通です。
言語固有なのは次の 2 点だけです:

1. **`extract` の `-l` フラグ** — 言語別の文字種分類を選択します（日本語 8 種、
   中国語 9 種、韓国語 10 種、英語 7 種。韓国語と英語は WC 特徴を使いません —
   [言語サポート概要](language-support/overview.md)を参照）。このためモデルは
   言語専用になります
2. **韓国語と英語は空白保持 TSV コーパス形式を使用** — どちらも単語間に空白を
   持つ表記で、空白が最も強い境界シグナルです。そのためそれぞれのコーパスは
   空白をトークンとして保持します（`corpus_udtreebank.sh -s` +
   `litsea extract --format tsv`）。日本語・中国語は空白を使わない表記のため、
   従来の空白区切り形式を使います

```sh
# Japanese / Chinese: space-separated corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
litsea extract -l japanese corpus.txt features.txt

# Korean / English: space-preserving TSV corpus
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l korean --format tsv corpus.tsv features.txt
```

`train` ステップのコマンドの形は 4 言語とも同一ですが、実際に使う
ハイパーパラメータは異なります。`litsea train` でゼロから通常の AdaBoost
モデルを学習する場合、`-t 0.0001 -i 20000`（[モデルの学習](training-guide/training-models.md)を参照）は
良い出発点ですが、これは同梱の `japanese`/`chinese`/`korean`/`english` モデルが
使っている値ではありません。それらのモデルは言語ごとに異なるエポック数と
剪定を伴う別の手順で学習されています。実際の手順は
[学習手順](pre-trained-models.md#学習手順)を参照してください。

## その他のトピック

- [モデルの評価](training-guide/evaluating-models.md) -- モデル品質の評価
- [モデルの再訓練](training-guide/retraining-models.md) -- 既存モデルの微調整
