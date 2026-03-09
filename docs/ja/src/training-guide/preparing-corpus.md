# コーパスの準備

良質な学習コーパスは、モデルの精度にとって不可欠です。このガイドでは、[Universal Dependencies (UD)](https://universaldependencies.org/) Treebanks を使用したコーパスの準備方法を説明します。

## データソース: UD Treebanks

Litsea は単語分割と品詞推定の両方のデータソースとして UD Treebanks を使用します。UD Treebanks は、多くの言語に対して CoNLL-U 形式の高品質な手動アノテーション済みデータを提供しています。

### 利用可能な Treebanks

| 言語 | ツリーバンク | リポジトリ |
|----------|----------|------------|
| 日本語 | UD Japanese-GSD | `UD_Japanese-GSD` |
| 中国語 | UD Chinese-GSD | `UD_Chinese-GSD` |
| 韓国語 | UD Korean-GSD | `UD_Korean-GSD` |

### ステップ 1: UD Treebank のダウンロード

`scripts/download_udtreebank.sh` を使用して UD Treebank をダウンロードします。スクリプトは学習用 CoNLL-U ファイルのパスを標準出力に出力します:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
```

オプション:

- `-l`: 言語コード（`ja`, `ko`, `zh`。デフォルト: `ja`）
- `-o`: 出力ディレクトリ（デフォルト: カレントディレクトリ）

手動でクローンする場合:

```sh
git clone https://github.com/UniversalDependencies/UD_Japanese-GSD
```

## 単語分割用コーパス

単語分割（AdaBoost）用のコーパスは以下の条件を満たすプレーンテキストファイルである必要があります:

- **1行1文**
- **単語をスペースで区切る**

```text
太郎 は 走っ た 。
Litsea は コンパクト な 単語 分割 ソフトウェア です 。
```

### CoNLL-U から単語分割用コーパスに変換

`scripts/corpus_udtreebank.sh` を使用して、CoNLL-U ファイルからスペース区切りの単語を抽出します:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt
```

`scripts/corpus_udtreebank.sh` は CoNLL-U ファイルから品詞タグを除去し、スペース区切りの単語のみを1行1文で出力します。

## 品詞推定用コーパス

品詞推定（Averaged Perceptron）を行う場合、各単語に品詞タグを付与した形式のコーパスを使用します。

### 品詞付きコーパスの形式

1行1文で、各単語を `単語/品詞` の形式でスペース区切りに記述します:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
Litsea/PROPN は/ADP 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

品詞タグは [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) タグセットに準拠し、17カテゴリで構成されます: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X。

### CoNLL-U から品詞付きコーパスに変換

`scripts/corpus_udtreebank.sh` に `-p` オプションを指定して、CoNLL-U ファイルを `単語/品詞` 形式に変換します:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
```

`-p` オプションにより品詞付き形式で出力されます。複合語トークンや空ノードは変換時に自動的に処理されます。

## コーパスの自動作成

Litsea には、UD Treebank のダウンロードと変換を自動化するヘルパースクリプトが `scripts/` ディレクトリに用意されています:

```sh
# UD Treebank をダウンロードして CoNLL-U ファイルのパスを取得
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)

# 単語分割用コーパスを生成
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt

# 品詞付きコーパスを生成
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
```

`scripts/download_udtreebank.sh` は以下の処理を行います:

1. 指定された言語に対応する UD Treebank リポジトリをクローン
2. 学習用 CoNLL-U ファイルのパスを標準出力に出力

`scripts/corpus_udtreebank.sh` は以下の処理を行います:

1. CoNLL-U ファイルを単語分割用コーパス形式に変換（デフォルト）
2. `-p` オプション指定時は品詞付きコーパス形式に変換

サポート言語: `ja`（日本語）、`ko`（韓国語）、`zh`（中国語）。

## コーパスの品質に関するヒント

- **多様性** -- さまざまな分野のテキストを含める（ニュース、文学、ウェブなど）
- **データ量** -- 大きなコーパスほど一般的に良いモデルを生成するが、収穫逓減がある
- **一貫性** -- コーパス全体で一貫したトークン化を確保する
- **重複排除** -- 偏りを避けるために重複文を除去する
- **クリーニング** -- HTML タグ、特殊なフォーマット、非テキストコンテンツを除去する
