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
| `--format <FORMAT>` | `space` | コーパスの形式: `space`（スペース区切りの単語）または `tsv`（タブ区切りのトークン。トークンは空白文字そのものでもよく、元の空白を保持できます）。`tsv` は `--pos` と併用できません |
| `--pos` | off | [二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)の学習用特徴量を抽出します。入力には品詞付きコーパスが必要です |
| `--stage2-features <SET>` | `fast` | `--pos` 用の stage-2 単語特徴セット: `full`（品質最優先）、`balanced`、`fast`（速度最優先） |
| `--tag-free` | オフ | 16 個のタグ依存特徴量テンプレート（`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`）を除外し、学習されるモデルを pointwise にして `segment()` の逐次スコアリングパスをスキップ可能にする（issue #183。同梱の `korean.model` で使用 -- 言語別の品質・速度トレードオフは[タグなし（pointwise）モデル](../pre-trained-models.md#タグなしpointwiseモデル)を参照）。`--format tsv` と併用可。`--pos` とは併用不可 |

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

## 二段構成の特徴量抽出

`--pos` フラグを指定すると、`extract` は通常の単語区切りコーパスの代わりに **品詞付きコーパス** を入力として受け取ります。各行には、`単語/品詞` の形式で UPOS タグが付与された単語が含まれます。

### 品詞付きコーパスの形式

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

`extract --pos` は、
[二段構成アーキテクチャ](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)
向けに `FEATURES_FILE` をプレフィックスとした**3つ**のファイルを書き出します。

| ファイル | 内容 |
|------|---------|
| `{FEATURES_FILE}.stage1` | 境界特徴量。文字位置ごとに1行、ラベルは `B` または `O`（通常の抽出と同じ文字レベルの特徴量テンプレート。先頭を含む全位置で出力） |
| `{FEATURES_FILE}.stage2` | 単語単位の特徴量。単語ごとに1行、ラベルは UPOS タグ。書き出すテンプレートは `--stage2-features` で制御 |
| `{FEATURES_FILE}.lexicon` | 候補タグ語彙表: `surface\tTAG:count[,TAG:count...]`（出現頻度の降順） |

同じプレフィックスを `litsea train --pos` に渡します:

```sh
litsea extract --pos -l japanese ./pos_corpus.txt ./pos_features
# ./pos_features.stage1, .stage2, .lexicon を書き出す
```
