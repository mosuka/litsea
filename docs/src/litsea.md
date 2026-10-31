# Library API Overview

The `litsea` crate provides a Rust API for word segmentation, model training, and feature extraction.

## Installation

```toml
[dependencies]
litsea = "0.15.0"
```

Loading models from local files with `load_model_from_path` or `load_model_from_reader` is synchronous and needs no async runtime. The `load_model` methods (for example `TwoStageLearner::load_model`) and `model_io::read_model_bytes` are async for every URI form, local paths included, so calling them needs an async runtime such as `tokio`.

Loading models over HTTP/HTTPS also needs the `remote_model` feature, which is off by default; without it, `http://` and `https://` URIs fail with `LitseaError::Unsupported` (see [Remote Model Loading](advanced/remote-model-loading.md)):

```toml
[dependencies]
litsea = { version = "0.15.0", features = ["remote_model"] }
```

## Module Map

```mermaid
graph LR
    A["litsea::segmenter"] --- B["Segmenter"]
    C["litsea::adaboost"] --- D["AdaBoost"]
    E["litsea::language"] --- F["Language"]
    G["litsea::extractor"] --- H["Extractor"]
    I["litsea::trainer"] --- J["Trainer, PerceptronTrainer, TwoStageTrainer, TwoStageMetrics"]
    K["litsea::error"] --- L["LitseaError, Result"]
    M["litsea::perceptron"] --- N["AveragedPerceptron"]
    O["litsea::upos"] --- P["Upos, SegmentLabel"]
    Q["litsea::metrics"] --- R["BinaryMetrics, MulticlassMetrics"]
    S["litsea::evaluation"] --- T["PosMetrics, SegmentationMetrics"]
    U["litsea::two_stage"] --- V["ModelKind, TwoStageFeatureSet, TwoStageLearner"]
    W["litsea::user_dictionary"] --- X["UserDictionary"]
    Y["litsea::model_io"] --- Z["read_model_bytes"]
```

| Module | Primary Types | Purpose |
|--------|--------------|---------|
| `litsea::segmenter` | `Segmenter`, `SegmentBuffer` | Word segmentation (owned or allocation-free output), two-stage segmentation with POS tagging |
| `litsea::user_dictionary` | `UserDictionary` | User dictionary: words emitted as single tokens, optionally with a UPOS tag for `segment_with_pos` (see [User Dictionary](litsea/segmenter.md#user-dictionary)) |
| `litsea::adaboost` | `AdaBoost` | Binary classification, model I/O |
| `litsea::perceptron` | `AveragedPerceptron` | Multiclass classification (two-stage training), model I/O |
| `litsea::upos` | `Upos`, `SegmentLabel` | UPOS POS tags, segment labels |
| `litsea::language` | `Language` | Language definitions, character classification |
| `litsea::extractor` | `Extractor` | Feature extraction from corpus |
| `litsea::trainer` | `Trainer`, `PerceptronTrainer`, `TwoStageTrainer`, `TwoStageMetrics` | Training orchestration |
| `litsea::error` | `LitseaError`, `Result` | Error type and result alias |
| `litsea::metrics` | `BinaryMetrics`, `MulticlassMetrics` | Evaluation metrics (in-sample) |
| `litsea::evaluation` | `PosMetrics`, `SegmentationMetrics` | Held-out evaluation against a gold corpus |
| `litsea::two_stage` | `ModelKind`, `TwoStageFeatureSet`, `TwoStageLearner` | Two-stage model container and model-kind detection |
| `litsea::model_io` | `read_model_bytes` | Resolves a model URI (path, `file://`, or `http(s)://` with the `remote_model` feature) to raw bytes (async) |

All primary types are also re-exported at the crate root, so `use litsea::Segmenter;` works as a shorthand for `use litsea::segmenter::Segmenter;`. The crate root also provides `litsea::version()`, which returns the crate version (for example `"0.15.0"`).

The learners resolve their own URIs, so `model_io::read_model_bytes` is rarely needed directly. It is public for callers that must inspect a model before choosing a learner — `litsea-binding-core` reads the bytes once, detects the kind with `ModelKind::detect`, and feeds the same bytes to `load_model_from_reader`, which avoids downloading a remote model twice.

## Quick Example

```rust
use std::path::Path;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;

fn main() -> litsea::Result<()> {
    let mut learner = AdaBoost::new(0.01, 100);
    learner.load_model_from_path(Path::new("./models/japanese.model"))?;

    let segmenter = Segmenter::with_learner(Language::Japanese, learner);
    let tokens = segmenter.segment("これはテストです。");

    assert_eq!(tokens, vec!["これ", "は", "テスト", "です", "。"]);
    Ok(())
}
```

## Quick Example (POS Tagging)

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;

fn main() -> litsea::Result<()> {
    let mut learner = TwoStageLearner::new();
    learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

    let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);
    let tokens = segmenter.segment_with_pos("これはテストです。")?;

    for (word, pos) in &tokens {
        print!("{}/{} ", word, pos);
    }
    println!();

    Ok(())
}
```

Full API documentation is available on [docs.rs/litsea](https://docs.rs/litsea).
