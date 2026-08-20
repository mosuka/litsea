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
| `--pos` | off | Enable POS-tagged segmentation output. Requires a [two-stage](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1) model (`train --pos`) |
| `--threads <N>` | `1` | Number of worker threads for batch segmentation (issue #185). The default keeps the single-threaded behavior; with `N > 1`, input lines are segmented in parallel and written in input order, so the output is byte-identical either way (works with and without `--pos`). Wall-clock time for large inputs drops with core count; single-line latency is unchanged |

## Input / Output

- **Input**: Reads from stdin, one sentence per line. Empty lines are skipped.
- **Output**: Writes to stdout, space-separated tokens, one line per input line.
- **Pipelines**: A downstream consumer closing the pipe early (e.g.
  `litsea segment model | head -1`) terminates the command successfully
  (exit code 0), so `segment` composes cleanly in shell pipelines.

## Examples

**Japanese:**

```sh
echo "LitseaはTinySegmenterを参考に開発された。" \
  | litsea segment -l japanese ./models/RWCP.model
```

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
```

**Chinese:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./models/chinese.model
```

**Korean:**

```sh
echo "한국어 단어 분할 테스트입니다." \
  | litsea segment -l korean ./models/korean.model
```

**Processing a file:**

```sh
cat input.txt | litsea segment -l japanese ./models/japanese.model > output.txt
```

**Loading a model from a URL:**

```sh
echo "テスト文です。" \
  | litsea segment -l japanese https://example.com/models/japanese.model
```

## POS-Tagged Segmentation (`--pos`)

When the `--pos` flag is specified, segmentation and POS tagging are
performed together with a
[two-stage](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)
model produced by `train --pos`. Pointing `--pos` at any other model
kind fails with a precise error (a standalone Averaged Perceptron file —
the removed joint POS model format — is rejected with a hint to retrain
with `train --pos`).

### Usage

```sh
echo "text" | litsea segment --pos [OPTIONS] <MODEL_URI>
```

### Output Format

Each token is output in `word/POS` format. POS tags conform to the UPOS tag set.

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./models/japanese_pos.model
```

```text
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

### Processing a File

```sh
cat input.txt | litsea segment --pos -l japanese ./models/japanese_pos.model > output.txt
```

## Parallel Batch Segmentation (`--threads`)

Sentences are independent, so batch throughput scales with cores without
any engine change: lines are read in chunks, each chunk is split across
the workers (each holding its own reusable segmentation buffer), and the
outputs are written strictly in input order.

```sh
litsea segment --threads 8 -l japanese ./models/japanese.model < corpus.txt > segmented.txt
```

`--threads 1` (the default) uses exactly the previous sequential loop.
Note `cargo bench -- external_corpus` remains a single-threaded *engine*
measurement — CLI-level thread scaling and engine throughput are different
numbers and should not be compared directly (see
[Benchmarking](../advanced/benchmarking.md)).

## Notes

- The `--language` flag must match the language the model was trained for
- The CLI loads models through the async URI API and supports HTTP/HTTPS with TLS (rustls); the library also offers synchronous local loading (`load_model_from_path`)
- The model URI is not restricted to file paths -- any valid URL is accepted
- When using `--pos`, the model must be a two-stage model trained with `train --pos`
