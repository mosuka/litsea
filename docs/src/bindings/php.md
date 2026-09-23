# PHP

`litsea-php` exposes Litsea to PHP 8.1+ through [ext-php-rs](https://github.com/davidcole1340/ext-php-rs). It is published on Packagist as `mosuka/litsea` and installed with [PIE](https://github.com/php/pie); the extension registers as `litsea`.

## Installation

A PHP extension is a shared object built against a specific PHP ABI, so unlike PyPI and npm there is no prebuilt package: the extension is compiled for your PHP, by PIE or by hand.

```sh
pie install mosuka/litsea
```

PIE 1.4 or later downloads a prebuilt `litsea.so` from the GitHub release for PHP 8.1 to 8.5 (NTS or ZTS) on Linux x86_64 / arm64 (glibc 2.35 or newer) and macOS on Apple silicon, installs it into the extension directory and enables it; no Rust toolchain is involved (`--with-php-config=...` selects another PHP). Other combinations (musl such as Alpine, older glibc, Intel Macs, PIE before 1.4) fall back to a source build, which needs a Rust toolchain, libclang and PIE's build tools (`autoconf`, `libtool`, `make`); Windows is not supported. Composer itself ignores `php-ext` packages, so `composer require mosuka/litsea` is not an installation route.

Without PIE, build and load the library yourself:

```sh
cargo build --release -p litsea-php
php -d extension=/path/to/target/release/liblitsea_php.so your-script.php
```

Add it to `php.ini` (`extension=/path/to/liblitsea_php.so`) to load it everywhere. Either way `php -m` lists the extension as `litsea`.

## Stubs for static analysis

`litsea-php/stubs/litsea.stubs.php` declares every class and function for IDEs, PHPStan (`stubFiles`) and Psalm (`stubs`). It is generated from the compiled extension and checked by the test suite, so it cannot drift from the Rust source. Point the analyser at a copy of the file; never include it at runtime, where the extension already declares the classes.

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
make test-litsea-php      # cargo test + build the extension + PHPUnit
make test-litsea-php-pie  # the phpize / configure / make path that PIE runs
make stubs-litsea-php     # regenerate litsea-php/stubs/litsea.stubs.php
make lint-litsea-php      # clippy
make build-litsea-php     # release build
```

The parity tests build the `litsea` CLI and compare the binding's output against it.

`composer.json` lives at the repository root because Packagist only reads a root manifest; its `php-ext.build-path` points PIE back at `litsea-php/`, where `config.m4` and `Makefile.frag` turn PIE's `phpize` / `configure` / `make` sequence into `cargo build --release -p litsea-php`.
