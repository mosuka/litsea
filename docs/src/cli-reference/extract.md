# extract

Extract features from a corpus file for model training.

## Usage

```sh
litsea extract [OPTIONS] <CORPUS_FILE> <FEATURES_FILE>
```

## Arguments

| Argument | Description |
|----------|------------|
| `CORPUS_FILE` | Path to the input corpus file (words separated by spaces, one sentence per line) |
| `FEATURES_FILE` | Path to the output features file |

## Options

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | Language for character type classification. Accepts: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |

## Corpus Format

The input corpus must have words separated by spaces, one sentence per line:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## Output Format

The features file contains one line per character position:

```text
1	UW1:B2 UW2:B1 UW3:L UW4:i UW5:t UC1:O UC2:O UC3:A UC4:A ...
-1	UW1:B1 UW2:L UW3:i UW4:t UW5:s UC1:O UC2:A UC3:A UC4:A ...
```

- `1` = word boundary
- `-1` = non-boundary
- Features are tab-separated

## Examples

```sh
# Japanese
litsea extract -l japanese ./corpus.txt ./features.txt

# Chinese
litsea extract -l zh ./corpus_zh.txt ./features_zh.txt

# Korean
litsea extract -l ko ./corpus_ko.txt ./features_ko.txt
```

Output to stderr on success:

```text
Feature extraction completed successfully.
```
