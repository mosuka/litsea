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

## コーパスの形式

入力コーパスは、単語をスペースで区切り、1行に1文とする形式です。

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## 出力形式

特徴量ファイルには、文字位置ごとに1行が含まれます。

```text
1	UW1:B2 UW2:B1 UW3:L UW4:i UW5:t UC1:O UC2:O UC3:A UC4:A ...
-1	UW1:B1 UW2:L UW3:i UW4:t UW5:s UC1:O UC2:A UC3:A UC4:A ...
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

`--pos` フラグを指定すると、品詞付きコーパスから Averaged Perceptron 用の特徴量を抽出します。

### 品詞付きコーパスの形式

品詞付きコーパスは、`単語/品詞` をスペースで区切り、1行に1文とする形式です。品詞タグは UPOS タグセット（ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X）を使用します。

```text
これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT
私/PRON の/PART 猫/NOUN は/PART 可愛い/ADJ 。/PUNCT
```

> **ヒント**: [`convert-conllu`](convert-conllu.md) コマンドで Universal Dependencies の CoNLL-U ファイルからこの形式に変換できます。

### 使い方

```sh
litsea extract --pos -l japanese ./corpus_pos.txt ./features_pos.txt
```

### 出力形式

品詞付き特徴量ファイルでは、ラベルが `SegmentLabel`（`B-NOUN`, `B-VERB`, ..., `B-X`, `O`）の18クラスとなります。

```text
B-NOUN	UW1:B2 UW2:B1 UW3:テ UW4:ス UC1:O UC2:O UC3:K UC4:K ...
O	UW1:B1 UW2:テ UW3:ス UW4:ト UC1:O UC2:K UC3:K UC4:K ...
B-AUX	UW1:ト UW2:で UW3:す UW4:。 UC1:K UC2:I UC3:I UC4:P ...
```
