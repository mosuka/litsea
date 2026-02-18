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

## Notes

- The `--language` flag must match the language the model was trained for
- Model loading is asynchronous and supports HTTP/HTTPS with TLS (rustls)
- The model URI is not restricted to file paths -- any valid URL is accepted
