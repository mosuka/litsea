# PHP

`litsea-php` exposes Litsea to PHP 8.1+ through [ext-php-rs](https://github.com/davidcole1340/ext-php-rs). It is distributed on Packagist as `litsea/litsea`.

## Installation

A PHP extension is a shared object built against a specific PHP ABI, so unlike PyPI and npm there is no prebuilt package: you build it and enable it.

```sh
cargo build --release -p litsea-php
php -d extension=/path/to/target/release/liblitsea_php.so your-script.php
```

Add it to `php.ini` (`extension=/path/to/liblitsea_php.so`) to load it everywhere. The build needs a Rust toolchain and libclang.

## Getting a model

The extension contains no models. Download one from the [`models/`](https://github.com/mosuka/litsea/tree/main/models) directory and pass its path — see [Pre-trained Models](../pre-trained-models.md). The model identifies its own kind, so `hasPos()` reports what was loaded and no flag is needed.

## Segmentation

```php
use Litsea\Segmenter;

$seg = Segmenter::open('japanese', 'models/japanese.model');

$seg->segment('これはテストです。');
// ['これ', 'は', 'テスト', 'です', '。']
```

The language name and its ISO 639-1 code are interchangeable (`'ja'`, `'japanese'`).

For space-delimited languages the whitespace is returned as its own token, so the tokens always reconstruct the input:

```php
Segmenter::open('ko', 'models/korean.model')->segment('안녕하세요 반갑습니다');
// ['안녕하세요', ' ', '반갑습니다']
```

## POS tagging

```php
$seg = Segmenter::open('japanese', 'models/japanese_pos.model');

foreach ($seg->segmentWithPos('これはテストです。') as $token) {
    printf("%s\t%s\t[%d:%d]\n", $token->surface, $token->pos, $token->start, $token->end);
}
// これ    PRON    [0:6]
// は      ADP     [6:9]
// テスト  NOUN    [9:18]
// です    AUX     [18:24]
// 。      PUNCT   [24:27]
```

`start` and `end` are byte offsets, and PHP strings are byte strings, so `substr($text, $token->start, $token->end - $token->start)` returns the surface directly — no encoding-aware slicing needed, unlike JavaScript.

## API

| Call | Returns |
|------|---------|
| `Segmenter::open($language, $path)` | A segmenter |
| `Segmenter::fromBytes($language, $contents)` | A segmenter |
| `Segmenter::fromUri($language, $uri)` | A segmenter (blocking download) |
| `segment($text)` | `string[]` |
| `segmentBatch($texts)` | `string[][]` |
| `segmentTokens($text)` | `Token[]` with byte offsets |
| `segmentWithPos($text)` | `Token[]` with tags and offsets |
| `segmentWithPosBatch($texts)` | `Token[][]` |
| `(new Extractor($language))->extract(...)` | `void` |
| `(new Extractor($language))->extractTwoStage(...)` | `void` |
| `(new Trainer($threshold, $iterations, $features))->train($model, $cancel?)` | `BinaryMetrics` |
| `(new PerceptronTrainer($epochs, $features))->train($model, $cancel?)` | `MulticlassMetrics` |
| `(new TwoStageTrainer($epochs, $prefix, $dominance?))->train($model, $cancel?)` | `TwoStageMetrics` |

ext-php-rs renames methods and properties to camelCase, so the PHP surface reads as `segmentWithPos()`, `hasPos()`, and `$metrics->numInstances`.

## Cancellation is pre-call only

This is the one place where PHP differs from the other bindings, and it is a property of the host rather than a gap here.

The Python binding releases the GIL and the Node.js binding runs training on a worker thread, so both can stop a run that is already going. A PHP request is single-threaded, and `pcntl` signal handlers cannot interrupt a blocking native call, so **no PHP code runs while `train()` executes**. A `CancelToken` therefore only takes effect if it was cancelled before the call:

```php
$cancel = new Litsea\CancelToken();
$cancel->cancel();

$metrics = (new Litsea\Trainer(0.01, 100000, 'features.txt'))->train('japanese.model', $cancel);
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics.

Because everything blocks, run training from the CLI SAPI rather than a web request.

## Errors

Every exception derives from `Litsea\LitseaException`, so one `catch` handles them all — the same hierarchy the Python binding exposes.

| Exception | Thrown when |
|-----------|-------------|
| `Litsea\InvalidArgumentException` | Unknown language name, unknown feature set, reused trainer |
| `Litsea\ModelException` | Download failed, or the file is a legacy joint POS model |
| `Litsea\IoException` | A file could not be read or written |
| `Litsea\ParseException` | The model or training data is malformed |
| `Litsea\UnsupportedException` | The scheme or operation is unavailable in this build |
| `Litsea\PosUnavailableException` | POS tagging requested from a segmentation-only model |

## Development

```sh
make test-litsea-php    # cargo test + build the extension + PHPUnit
make lint-litsea-php    # clippy
make build-litsea-php   # release build
```

The parity tests build the `litsea` CLI and compare the binding's output against it.
