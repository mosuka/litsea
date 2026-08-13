# extract

モデル学習用にコーパスファイルから特徴量を抽出します。

## 使い方

```sh
litsea extract [OPTIONS] <CORPUS_FILE> <FEATURES_FILE>
```

## 引数

| Argument | Description |
|----------|------------|
| `CORPUS_FILE` | 入力コーパスファイルのパス（単語をスペースで区切り、1行に1文） |
| `FEATURES_FILE` | 出力特徴量ファイルのパス |

## オプション

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | 文字タイプ分類に使用する言語。指定可能な値: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |
| `--pos` | off | 品詞（POS）特徴量抽出モードを有効にします。入力には品詞付きコーパスが必要です |
| `--format <FORMAT>` | `space` | コーパスの形式: `space`（スペース区切りの単語）または `tsv`（タブ区切りのトークン。トークンは空白文字そのものでもよく、元の空白を保持できます）。`tsv` は `--pos` と併用できません |

## コーパスの形式

入力コーパスは、単語をスペースで区切り、1行に1文とする形式です。

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

### TSV コーパス形式（`--format tsv`）

`--format tsv` を指定すると、トークンはタブ文字で区切られ、トークンとして空白文字そのもの（`" "`）を含められます。これにより学習テキスト内に元の文の空白が保持されます。空白がほとんどの語境界を示す韓国語のような言語では、これが不可欠です（[韓国語](../language-support/korean.md)を参照）。このようなコーパスは UD Treebank から `corpus_udtreebank.sh -s` で生成できます:

```sh
litsea extract -l korean --format tsv ./ko_corpus.tsv ./ko_features.txt
```

## 出力形式

特徴量ファイルには、文字位置ごとに1行が含まれます。コーパス行 `これ は テスト です 。` に対する最初の2行は次のとおりです。

```text
-1	BC1:OI	BC2:II	BC3:II	BP1:UU	BP2:UU	BQ1:UOI	BQ2:UII	BQ3:UOI	BQ4:UII	BW1:B1こ	BW2:これ	...
1	BC1:II	BC2:II	BC3:IK	BP1:UU	BP2:UO	BQ1:UII	BQ2:UII	BQ3:OII	BQ4:OII	BW1:これ	BW2:れは	...
```

- `1` = 語境界
- `-1` = 非境界
- 特徴量はアルファベット順にソートされてタブ区切りで書き出されます。そのため各行はテンプレート定義順ではなく `BC1:` 特徴量から始まります

## 使用例

```sh
# Japanese
litsea extract -l japanese ./corpus.txt ./features.txt

# Chinese
litsea extract -l zh ./corpus_zh.txt ./features_zh.txt

# Korean
litsea extract -l ko ./corpus_ko.txt ./features_ko.txt
```

成功時のstderr出力:

```text
Feature extraction completed successfully.
```

## 品詞付き特徴量抽出（`--pos`）

`--pos` フラグを指定すると、`extract` は通常の単語区切りコーパスの代わりに **品詞付きコーパス** を入力として受け取ります。各行には、`単語/品詞` の形式で UPOS タグが付与された単語が含まれます。

### 品詞付きコーパスの形式

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

### 品詞付き特徴量の出力形式

POSモードでは、ラベル列は二値の `1`/`-1` の代わりにセグメントラベル（`B-NOUN`, `B-VERB`, ..., `B-X`, `O`）を使用します。特徴量はここでもアルファベット順にソートされてタブ区切りで書き出されます。

```text
B-PRON	BC1:OO	BC2:OI	BC3:II	BP1:UU	BP2:UU	BQ1:UOO	BQ2:UOI	BQ3:UOO	BQ4:UOI	BW1:B2B1	...
O	BC1:OI	BC2:II	BC3:II	BP1:UU	BP2:UU	BQ1:UOI	BQ2:UII	BQ3:UOI	BQ4:UII	BW1:B1こ	...
```

### 品詞付き特徴量抽出の例

```sh
litsea extract --pos -l japanese ./pos_corpus.txt ./pos_features.txt
```
