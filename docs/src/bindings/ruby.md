# Ruby

`litsea-ruby` exposes Litsea to Ruby 3.1+ through [magnus](https://github.com/matsadler/magnus) and rb-sys. It is published to RubyGems as `litsea`.

## Installation

```sh
gem install litsea
```

The gem is source-only and compiles the extension on install, so a Rust toolchain is required.

## Getting a model

The gem contains no models. Download one from the [`models/`](https://github.com/mosuka/litsea/tree/main/models) directory and pass its path — see [Pre-trained Models](../pre-trained-models.md). The model identifies its own kind, so `has_pos?` reports what was loaded and no flag is needed.

## Segmentation

```ruby
require "litsea"

seg = Litsea::Segmenter.open(:japanese, "models/japanese.model")

seg.segment("これはテストです。")
# => ["これ", "は", "テスト", "です", "。"]
```

The language accepts a Symbol or a String, and the ISO 639-1 code works too (`:ja`, `"japanese"`).

For space-delimited languages the whitespace is returned as its own token, so the tokens always reconstruct the input:

```ruby
Litsea::Segmenter.open(:korean, "models/korean.model").segment("안녕하세요 반갑습니다")
# => ["안녕하세요", " ", "반갑습니다"]
```

## POS tagging

```ruby
seg = Litsea::Segmenter.open(:japanese, "models/japanese_pos.model")

seg.segment_with_pos("これはテストです。").each do |token|
  puts "#{token.surface}\t#{token.pos}\t[#{token.start}..#{token.end}]"
end
# これ    PRON    [0..6]
# は      ADP     [6..9]
# テスト  NOUN    [9..18]
# です    AUX     [18..24]
# 。      PUNCT   [24..27]
```

`start` and `end` are **byte** offsets. Ruby's `String#[]` counts characters, so slice with `byteslice`:

```ruby
text.byteslice(token.start, token.end - token.start)   # == token.surface
```

## API

| Call | Returns |
|------|---------|
| `Litsea::Segmenter.open(language, path)` | A segmenter |
| `Litsea::Segmenter.from_bytes(language, data)` | A segmenter (accepts a binary String) |
| `Litsea::Segmenter.from_uri(language, uri)` | A segmenter |
| `#segment(text)` | `Array<String>` |
| `#segment_batch(texts)` | `Array<Array<String>>` |
| `#segment_tokens(text)` | `Array<Litsea::Token>` with byte offsets |
| `#segment_with_pos(text)` | `Array<Litsea::Token>` with tags and offsets |
| `#segment_with_pos_batch(texts)` | `Array<Array<Litsea::Token>>` |
| `Litsea::Extractor.new(language)#extract(...)` | `nil` |
| `Litsea::Extractor.new(language)#extract_two_stage(...)` | `nil` |
| `Litsea::Trainer.new(threshold, iterations, features)#train(model, cancel:)` | `BinaryMetrics` |
| `Litsea::PerceptronTrainer.new(epochs, features)#train(model, cancel:)` | `MulticlassMetrics` |
| `Litsea::TwoStageTrainer.new(epochs, prefix, dominance:)#train(model, cancel:)` | `TwoStageMetrics` |

## Releasing the GVL

Long-running work — loading a model, extracting features, training — runs with the Global VM Lock released, so other Ruby threads keep going. That is what makes cancellation useful:

```ruby
cancel = Litsea::CancelToken.new
Thread.new { sleep 60; cancel.cancel }

metrics = Litsea::Trainer.new(0.01, 100_000, "features.txt").train("japanese.model", cancel: cancel)
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics. The binding never installs a signal handler.

Neither magnus nor rb-sys wraps `rb_thread_call_without_gvl` — magnus lists it among the C functions it does not bind, and it is declared in a header outside rb-sys's generated bindings — so the binding declares it itself in `src/gvl.rs`, behind an `extern "C"` trampoline that catches panics so none can unwind across the C frame. Two tests hold that claim honest: one asserts another Ruby thread keeps ticking during training, and one asserts a cancel from another thread lands inside the training window. Removing the GVL release turns both red.

Segmentation of a single sentence keeps the GVL: it is short enough that releasing it would cost more than the work.

A `TwoStageTrainer` can only be used once — training collapses stage 1 into an AdaBoost model, which consumes it. `available?` reports the state, and a second `train` raises.

## Errors

Every error derives from `Litsea::Error`, so one `rescue` handles them all — the same hierarchy the Python and PHP bindings expose.

| Error | Raised when |
|-------|-------------|
| `Litsea::InvalidArgumentError` | Unknown language name, unknown feature set, reused trainer |
| `Litsea::ModelError` | Download failed, or the file is a legacy joint POS model |
| `Litsea::IoError` | A file could not be read or written |
| `Litsea::ParseError` | The model or training data is malformed |
| `Litsea::UnsupportedError` | The scheme or operation is unavailable in this build |
| `Litsea::PosUnavailableError` | POS tagging requested from a segmentation-only model |

## Development

```sh
make test-litsea-ruby    # cargo test + rake compile + rake test
make lint-litsea-ruby    # clippy + rubocop
make build-litsea-ruby   # release build
```

`bundle` must be usable with the active Ruby; a version manager's shim can exist while the selected interpreter has no bundler, so the Makefile checks and says so. The parity tests build the `litsea` CLI and compare the binding's output against it.
