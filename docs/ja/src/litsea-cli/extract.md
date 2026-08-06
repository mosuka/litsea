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

## コーパスの形式

入力コーパスは、単語をスペースで区切り、1行に1文とする形式です。

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## 出力形式

特徴量ファイルには、文字位置ごとに1行が含まれます。

```text
1	UW1:B2	UW2:B1	UW3:L	UW4:i	UW5:t	UC1:O	UC2:O	UC3:A	UC4:A ...
-1	UW1:B1	UW2:L	UW3:i	UW4:t	UW5:s	UC1:O	UC2:A	UC3:A	UC4:A ...
```

- `1` = 語境界
- `-1` = 非境界
- 特徴量はタブ区切り

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

POSモードでは、ラベル列は二値の `1`/`-1` の代わりにセグメントラベル（`B-NOUN`, `B-VERB`, ..., `B-X`, `O`）を使用します。

```text
B-NOUN	UW1:B2 UW2:B1 UW3:こ UW4:れ UW5:は UC1:O UC2:O UC3:I UC4:I ...
O	UW1:B1 UW2:こ UW3:れ UW4:は UW5:テ UC1:O UC2:I UC3:I UC4:I ...
```

### 品詞付き特徴量抽出の例

```sh
litsea extract --pos -l japanese ./pos_corpus.txt ./pos_features.txt
```
