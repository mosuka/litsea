# コーパスの準備

良質な学習コーパスは、モデルの精度にとって不可欠です。このガイドでは、コーパスの準備方法を説明します。

## コーパスの形式

コーパスは以下の条件を満たすプレーンテキストファイルである必要があります:
- **1行1文**
- **単語をスペースで区切る**

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## コーパスの自動作成

Litsea には、Wikipedia からコーパスを構築するためのヘルパースクリプトが `scripts/` ディレクトリに用意されています。

### ステップ 1: Wikipedia テキストのダウンロード

```sh
bash scripts/wikitexts.sh ja   # Japanese
bash scripts/wikitexts.sh ko   # Korean
bash scripts/wikitexts.sh zh   # Chinese
```

このスクリプトは以下の処理を行います:
1. Wikipedia API から記事タイトルをダウンロード
2. 言語固有の基準でフィルタリング
3. 記事テキストを取得
4. `litsea split-sentences` を使用して文に分割

### ステップ 2: Lindera によるトークン化

```sh
bash scripts/corpus.sh ja ./wikitexts_ja.txt ./corpus_ja.txt
bash scripts/corpus.sh ko ./wikitexts_ko.txt ./corpus_ko.txt
bash scripts/corpus.sh zh ./wikitexts_zh.txt ./corpus_zh.txt
```

このスクリプトは [Lindera](https://github.com/lindera/lindera) を言語固有の辞書とともに使用します:

| 言語 | 辞書 | 備考 |
|----------|-----------|-------|
| 日本語 | UniDic | 複合語フィルタ付き |
| 韓国語 | ko-dic | 韓国語辞書 |
| 中国語 | CC-CEDICT | 中英辞書 |

出力は **wakati** 形式（スペース区切りのトークン）で、特徴量抽出にそのまま使用できます。

## 品詞付きコーパスの準備

品詞推定（POS Tagging）を行う場合、Litsea では各単語に品詞タグを付与した形式のコーパスを使用します。

### 品詞付きコーパスの形式

1行1文で、各単語を `単語/品詞` の形式でスペース区切りに記述します:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
Litsea/PROPN は/ADP 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

品詞タグは [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) タグセットに準拠し、17カテゴリで構成されます: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X。

### UD Treebanks をデータソースとして使用

[Universal Dependencies (UD)](https://universaldependencies.org/) は、多くの言語に対して CoNLL-U 形式の高品質なツリーバンクデータを提供しています。Litsea には CoNLL-U ファイルを品詞付きコーパス形式に変換するコンバーターが含まれています。

#### ステップ 1: UD Treebank のダウンロード

```sh
git clone https://github.com/UniversalDependencies/UD_Japanese-GSD
```

サポート言語で利用可能な UD Treebanks:

| 言語 | ツリーバンク | リポジトリ |
|----------|----------|------------|
| 日本語 | UD Japanese-GSD | `UD_Japanese-GSD` |
| 中国語 | UD Chinese-GSD | `UD_Chinese-GSD` |
| 韓国語 | UD Korean-GSD | `UD_Korean-GSD` |

#### ステップ 2: CoNLL-U から品詞付きコーパスに変換

```sh
litsea convert-conllu UD_Japanese-GSD/ja_gsd-ud-train.conllu corpus.txt
```

このコマンドは CoNLL-U 形式を Litsea が期待する `単語/品詞` 形式に変換します。複合語トークンや空ノードは変換時に自動的に処理されます。

## コーパスの品質に関するヒント

- **多様性** -- さまざまな分野のテキストを含める（ニュース、文学、ウェブなど）
- **データ量** -- 大きなコーパスほど一般的に良いモデルを生成するが、収穫逓減がある
- **一貫性** -- コーパス全体で一貫したトークン化を確保する
- **重複排除** -- 偏りを避けるために重複文を除去する
- **クリーニング** -- HTML タグ、特殊なフォーマット、非テキストコンテンツを除去する
