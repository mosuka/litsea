# segment

Segment text into words using a trained model.

## Usage

```sh
echo "text" | litsea segment [OPTIONS] <MODEL_URI>
```

## Arguments

| Argument | Description |
|----------|------------|
| `MODEL_URI` | Path or URL to the trained model file. Supports: local file paths, `file://`, `http://`, `https://` |

## Options

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | Language for character type classification. Accepts: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |
| `--pos` | off | Enable POS-tagged segmentation output. Requires a POS model trained with `train --pos` |

## Input / Output

- **Input**: Reads from stdin, one sentence per line. Empty lines are skipped.
- **Output**: Writes to stdout, space-separated tokens, one line per input line.

## Examples

**Japanese:**

```sh
echo "LitseaはTinySegmenterを参考に開発された。" \
  | litsea segment -l japanese ./resources/japanese.model
```

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
```

**Chinese:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./resources/chinese.model
```

**Korean:**

```sh
echo "한국어 단어 분할 테스트입니다." \
  | litsea segment -l korean ./resources/korean.model
```

**Processing a file:**

```sh
cat input.txt | litsea segment -l japanese ./resources/japanese.model > output.txt
```

**Loading a model from a URL:**

```sh
echo "テスト文です。" \
  | litsea segment -l japanese https://example.com/models/japanese.model
```

## POS-Tagged Segmentation

When the `--pos` flag is specified, `segment` outputs each token annotated with its UPOS tag in `word/POS` format. This requires a POS model trained with `train --pos`.

**Example:**

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./resources/japanese_pos.model
```

```text
今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

**Processing a file with POS tags:**

```sh
cat input.txt | litsea segment --pos -l japanese ./resources/japanese_pos.model > output.txt
```

## Notes

- The `--language` flag must match the language the model was trained for
- Model loading is asynchronous and supports HTTP/HTTPS with TLS (rustls)
- The model URI is not restricted to file paths -- any valid URL is accepted
- When using `--pos`, the model must be a POS model trained with `train --pos`
