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
