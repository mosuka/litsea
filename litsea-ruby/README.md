# litsea-ruby

Ruby binding for [Litsea](https://github.com/mosuka/litsea), a compact word segmentation and POS (Part-of-Speech) tagging library for Japanese, Chinese, Korean, and English.

[日本語のREADME](README_ja.md)

## Installation

```sh
gem install litsea
```

The gem compiles the native extension on install, so a Rust toolchain is required. Ruby 3.1 or later.

## Models are not bundled

The gem ships code only. Download a pre-trained model from the [Litsea repository](https://github.com/mosuka/litsea/tree/main/models) and point the segmenter at it:

| Model | Purpose | Size |
|-------|---------|------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | Segmentation | 84 KB – 2.0 MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | Segmentation + POS | 3.0 – 8.0 MB |

You never have to say which kind you have: the model file identifies itself, and `has_pos?` reports what the loaded model can do.

## Usage

### Segmentation

```ruby
require "litsea"

seg = Litsea::Segmenter.open(:japanese, "models/japanese.model")

seg.segment("これはテストです。")
# => ["これ", "は", "テスト", "です", "。"]

seg.segment_batch(["これはテストです。", "東京都から神奈川県へ引っ越した"])
```

The language accepts a Symbol or a String, and the ISO 639-1 code works too: `:ja`, `"japanese"`.

For space-delimited languages the whitespace comes back as its own token, so the tokens always reconstruct the input:

```ruby
Litsea::Segmenter.open(:korean, "models/korean.model").segment("안녕하세요 반갑습니다")
# => ["안녕하세요", " ", "반갑습니다"]
```

### POS tagging

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

`start` and `end` are **byte** offsets, so slice with `byteslice` — Ruby's `String#[]` counts characters:

```ruby
text.byteslice(token.start, token.end - token.start)   # == token.surface
```

Calling `segment_with_pos` on a segmentation-only model raises `Litsea::PosUnavailableError`.

### Other model sources

```ruby
Litsea::Segmenter.from_bytes(:korean, File.binread("korean.model"))
Litsea::Segmenter.from_uri(:chinese, "https://example.com/chinese.model")
```

### Training

```ruby
Litsea::Extractor.new(:japanese).extract("corpus.txt", "features.txt")

metrics = Litsea::Trainer.new(0.01, 10_000, "features.txt").train("japanese.model")
puts format("accuracy: %.2f%%", metrics.accuracy)
```

Two-stage (segmentation + POS) training:

```ruby
Litsea::Extractor.new(:japanese).extract_two_stage("corpus_pos.txt", "features", feature_set: "fast")

metrics = Litsea::TwoStageTrainer.new(10, "features").train("japanese_pos.model")
puts metrics.stage1.accuracy, metrics.stage2.accuracy
```

A `TwoStageTrainer` can only be used once — training collapses stage 1 into an AdaBoost model, which consumes it. `available?` reports whether it can still run, and a second `train` raises.

### Cancelling a training run

Training releases the GVL, so other Ruby threads keep running — which means one of them can stop a run that is already going:

```ruby
cancel = Litsea::CancelToken.new
Thread.new { sleep 60; cancel.cancel }

metrics = Litsea::Trainer.new(0.01, 100_000, "features.txt").train("japanese.model", cancel: cancel)
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics. The binding never installs a signal handler.

## Errors

Every error derives from `Litsea::Error`, so one `rescue` handles them all.

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
make build-litsea-ruby   # release build
```

The parity tests build the `litsea` CLI and compare the binding's output against it. Note that `bundle` must be available for the active Ruby; with rbenv, `rbenv local 3.4.9` (or any installed 3.1+) is enough.

## License

MIT. See [LICENSE](../LICENSE).
