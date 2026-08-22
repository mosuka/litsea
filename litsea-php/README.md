# litsea-php

PHP binding for [Litsea](https://github.com/mosuka/litsea), a compact word segmentation and POS (Part-of-Speech) tagging library for Japanese, Chinese, Korean, and English.

[日本語のREADME](README_ja.md)

## Installation

A PHP extension is a shared object built against a specific PHP ABI, so there is no prebuilt package to install — you build it and enable it in `php.ini`.

```sh
git clone https://github.com/mosuka/litsea.git
cd litsea
cargo build --release -p litsea-php
```

Then load the library. Either add it to `php.ini`:

```ini
extension=/path/to/litsea/target/release/liblitsea_php.so
```

or pass it per invocation:

```sh
php -d extension=/path/to/liblitsea_php.so your-script.php
```

Requires PHP 8.1 or later and a Rust toolchain, plus libclang for the build (`libclang-dev` on Debian/Ubuntu).

## Models are not bundled

Download a pre-trained model from the [Litsea repository](https://github.com/mosuka/litsea/tree/main/models) and point the segmenter at it:

| Model | Purpose | Size |
|-------|---------|------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | Segmentation | 84 KB – 2.0 MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | Segmentation + POS | 3.0 – 8.0 MB |

You never have to say which kind you have: the model file identifies itself, and `hasPos()` reports what the loaded model can do.

## Usage

### Segmentation

```php
use Litsea\Segmenter;

$seg = Segmenter::open('japanese', 'models/japanese.model');

$seg->segment('これはテストです。');
// ['これ', 'は', 'テスト', 'です', '。']

$seg->segmentBatch(['これはテストです。', '東京都から神奈川県へ引っ越した']);
```

The language name and its ISO 639-1 code are interchangeable (`'ja'`, `'japanese'`).

For space-delimited languages the whitespace comes back as its own token, so the tokens always reconstruct the input:

```php
Segmenter::open('ko', 'models/korean.model')->segment('안녕하세요 반갑습니다');
// ['안녕하세요', ' ', '반갑습니다']
```

### POS tagging

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

`start` and `end` are byte offsets. PHP strings are byte strings, so they work directly with `substr()`:

```php
substr($text, $token->start, $token->end - $token->start);   // === $token->surface
```

Calling `segmentWithPos()` on a segmentation-only model throws `Litsea\PosUnavailableException`.

### Other model sources

```php
Segmenter::fromBytes('korean', file_get_contents('korean.model'));
Segmenter::fromUri('chinese', 'https://example.com/chinese.model');
```

### Training

```php
use Litsea\Extractor;
use Litsea\Trainer;

(new Extractor('japanese'))->extract('corpus.txt', 'features.txt');

$metrics = (new Trainer(0.01, 10000, 'features.txt'))->train('japanese.model');
printf("accuracy: %.2f%%\n", $metrics->accuracy);
```

Two-stage (segmentation + POS) training:

```php
use Litsea\Extractor;
use Litsea\TwoStageTrainer;

(new Extractor('japanese'))->extractTwoStage('corpus_pos.txt', 'features', 'fast');

$metrics = (new TwoStageTrainer(10, 'features'))->train('japanese_pos.model');
printf("%.2f%% / %.2f%%\n", $metrics->stage1Accuracy, $metrics->stage2Accuracy);
```

A `TwoStageTrainer` can only be used once — training collapses stage 1 into an AdaBoost model, which consumes it. `isAvailable()` reports whether it can still run, and a second `train()` throws.

**Train from the CLI SAPI.** Training blocks the process for as long as it runs, which is not something a web request should do.

## Cancellation is pre-call only in PHP

The Python binding releases the GIL and the Node.js binding moves training to a worker thread, so both can cancel a run that is already going. PHP can do neither: a request is single-threaded, and `pcntl` signal handlers cannot interrupt a blocking native call.

A `Litsea\CancelToken` therefore only takes effect when it is cancelled **before** `train()` is called:

```php
use Litsea\CancelToken;
use Litsea\Trainer;

$cancel = new CancelToken();
$cancel->cancel();

// Stops at the first check point and still writes the partial model.
$metrics = (new Trainer(0.01, 100000, 'features.txt'))->train('japanese.model', $cancel);
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics.

## Errors

Every exception derives from `Litsea\LitseaException`, so one `catch` handles them all.

| Exception | Thrown when |
|-----------|-------------|
| `Litsea\InvalidArgumentException` | Unknown language name, unknown feature set, reused trainer |
| `Litsea\ModelException` | Download failed, or the file is a legacy joint POS model |
| `Litsea\IoException` | A file could not be read or written |
| `Litsea\ParseException` | The model or training data is malformed |
| `Litsea\UnsupportedException` | The scheme or operation is unavailable in this build |
| `Litsea\PosUnavailableException` | POS tagging requested from a segmentation-only model |

## Naming

ext-php-rs renames methods and properties to camelCase, so the PHP API reads as `segmentWithPos()`, `hasPos()`, and `$metrics->numInstances`.

## Development

```sh
make test-litsea-php    # cargo test + build the extension + PHPUnit
make build-litsea-php   # release build
```

## License

MIT. See [LICENSE](../LICENSE).
