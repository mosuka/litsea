# PHP

`litsea-php` exposes Litsea to PHP 8.1+ through [ext-php-rs](https://github.com/davidcole1340/ext-php-rs). It is published on Packagist as `mosuka/litsea` and installed with [PIE](https://github.com/php/pie); the extension registers as `litsea`.

## Installation

A PHP extension is a shared object built against a specific PHP ABI, so a prebuilt binary fits only one PHP version, thread-safety mode and platform: PIE downloads a matching one from the GitHub release when there is one, and otherwise compiles the extension for your PHP. You can also build it by hand.

```sh
pie install mosuka/litsea
```

PIE 1.4 or later downloads a prebuilt `litsea.so` from the GitHub release for PHP 8.1 to 8.5 (NTS or ZTS) on Linux x86_64 / arm64 (glibc 2.35 or newer) and macOS on Apple silicon, installs it into the extension directory and enables it; no Rust toolchain is involved (`--with-php-config=...` selects another PHP). Other combinations (musl such as Alpine, older glibc, Intel Macs, PIE before 1.4) fall back to a source build, which needs a Rust toolchain, libclang and PIE's build tools (`autoconf`, `libtool`, `make`); Windows is not supported. Composer itself ignores `php-ext` packages, so `composer require mosuka/litsea` is not an installation route.

Without PIE, build and load the library yourself:

```sh
cargo build --release -p litsea-php
php -d extension=/path/to/target/release/liblitsea_php.so your-script.php
```

Add it to `php.ini` (`extension=/path/to/liblitsea_php.so`) to load it everywhere. On macOS, cargo writes `liblitsea_php.dylib` instead; load that path. Either way `php -m` lists the extension as `litsea`.

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

## User dictionary

Every constructor takes three optional trailing arguments ([#263](https://github.com/mosuka/litsea/issues/263), [#264](https://github.com/mosuka/litsea/issues/264)): `$user_words`, an array of words to emit as single tokens wherever they occur; `$user_words_gate`, a flip ratio in `0.0`–`1.0` that applies a word only where the model's margin against it is small enough; and `$user_word_tags`, an array mapping further words to the UPOS tag that `segmentWithPos()` gives them. All three default to `null` and work as named arguments too.

```php
$seg = Segmenter::open('japanese', 'models/japanese.model', ['雪だるま', '新幹線']);

$seg->segment('雪だるまを作る');
// ['雪だるま', 'を', '作る']   (without the dictionary: ['雪', 'だる', 'ま', 'を', '作る'])

Segmenter::open('japanese', 'models/japanese.model', user_words: ['雪だるま'], user_words_gate: 0.5);
```

With a POS model, `$user_word_tags` fixes the tag of a word the tagger gets wrong:

```php
$tagged = Segmenter::open('japanese', 'models/japanese_pos.model', user_word_tags: ['チャットGPT' => 'PROPN']);

array_map(fn ($t) => "{$t->surface}/{$t->pos}", $tagged->segmentWithPos('チャットGPTに聞く'));
// ['チャットGPT/PROPN', 'に/ADP', '聞く/VERB']   (in $user_words instead: 'チャットGPT/NOUN')
```

A registered word gets a boundary at each edge and none inside, overlapping occurrences are resolved leftmost-longest, and an untagged word of one character is ignored. With a POS model a word in `$user_words` is tagged like any other word (`雪だるま/NOUN` with `japanese_pos.model`), while a token that is a word in `$user_word_tags` gets its tag, ahead of the model's lexicon and its tagger; `segment()` ignores the tags. A word of one character in `$user_word_tags` changes no boundary and retags only the tokens that are exactly that character (issue #276). Matching is context-free, so a short fragment registered as a word breaks every other word it occurs in; a gate of `0.0` applies a word only where the model already agrees with it. A gate outside `0.0`–`1.0`, or a tag that is not one of the 17 UPOS tags (upper case), throws `Litsea\InvalidArgumentException`. See [Segmenter](../litsea/segmenter.md#user-dictionary) for the full semantics.

## API

| Call | Returns |
|------|---------|
| `Segmenter::open($language, $path, $user_words?, $user_words_gate?, $user_word_tags?)` | A segmenter |
| `Segmenter::fromBytes($language, $data, $user_words?, $user_words_gate?, $user_word_tags?)` | A segmenter |
| `Segmenter::fromUri($language, $uri, $user_words?, $user_words_gate?, $user_word_tags?)` | A segmenter (blocking download) |
| `language()` | `string`, the canonical language name |
| `hasPos()` | `bool` |
| `segment($text)` | `string[]` |
| `segmentBatch($texts)` | `string[][]` |
| `segmentTokens($text)` | `Token[]` with byte offsets |
| `segmentWithPos($text)` | `Token[]` with tags and offsets |
| `segmentWithPosBatch($texts)` | `Token[][]` |
| `(new Extractor($language))->extract($corpus_path, $features_path, $tsv = false, $tag_free = false)` | `void` (`$tsv` reads the TSV corpus format; `$tag_free` omits the tag-dependent feature templates) |
| `(new Extractor($language))->extractTwoStage($corpus_path, $output_prefix, $feature_set = null, $tsv = false)` | `void` (writes `{$output_prefix}.stage1`, `.stage2` and `.lexicon`; `$feature_set` is `'full'`, `'balanced'` or `'fast'`, and `null` means `'fast'`) |
| `(new Trainer($threshold, $num_iterations, $features_path))->train($model_path, $cancel?)` | `BinaryMetrics` |
| `(new Trainer(...))->loadModel($model_uri)` | `void` (continues training from an existing model: a path, `file://` or `http(s)://`) |
| `(new PerceptronTrainer($num_epochs, $features_path))->train($model_path, $cancel?)` | `MulticlassMetrics` |
| `(new TwoStageTrainer($num_epochs, $features_prefix, $dominance = 0.99))->train($model_path, $cancel?)` | `TwoStageMetrics` |
| `(new TwoStageTrainer(...))->isAvailable()` | `bool` (`false` once `train()` has run) |
| `(new CancelToken())->cancel()` / `reset()` / `isCancelled()` | `void` / `void` / `bool` |
| `Litsea\version()` | `string`, the `litsea` version |
| `Litsea\supported_languages()` | `string[]` |

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

Every exception derives from `Litsea\LitseaException`, so one `catch` handles them all — the same hierarchy the Python binding exposes. A failure that fits none of the classes below is thrown as `Litsea\LitseaException` itself.

| Exception | Thrown when |
|-----------|-------------|
| `Litsea\InvalidArgumentException` | Unknown language name, unknown feature set, reused trainer, `$user_words_gate` outside `0.0`–`1.0`, a `$user_word_tags` value that is not a UPOS tag, `$dominance` outside (0.5, 1.0], an unknown URI scheme |
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
