# Trainer

The `Trainer` struct orchestrates the full model training pipeline.

## Definition

```rust
pub struct Trainer {
    learner: AdaBoost,
}
```

## Constructor

### `Trainer::new`

```rust
pub fn new(
    threshold: f64,
    num_iterations: usize,
    features_path: &Path,
) -> litsea::Result<Self>
```

Creates a trainer and initializes it from a features file. This calls `AdaBoost::initialize_features()` and `AdaBoost::initialize_instances()`.

```rust
use std::path::Path;
use litsea::trainer::Trainer;

let mut trainer = Trainer::new(
    0.0001,                          // threshold
    20000,                           // max iterations
    Path::new("./features.txt"),     // features file
)?;
```

## Methods

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> litsea::Result<()>
```

Loads an existing model for retraining. Supports file paths, `file://`, and (with the `remote_model` feature) `http://` and `https://` URIs.

When called after `Trainer::new`, the loaded weights are merged into the freshly initialized training data by feature name, so incremental training starts from the existing model without corrupting the feature index.

```rust
trainer.load_model("./models/japanese.model").await?;
```

### `train`

```rust
pub fn train(
    &mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<BinaryMetrics>
```

Trains the model and saves it to the specified path. Returns evaluation metrics.

The `running` flag enables graceful interruption -- set it to `false` to stop training early.

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

let running = AtomicBool::new(true);
let metrics = trainer.train(&running, Path::new("./model.model"))?;

println!("Accuracy: {:.2}%", metrics.accuracy);
```

## Full Training Example

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::Trainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let mut trainer = Trainer::new(
        0.0001,
        20000,
        Path::new("./features.txt"),
    )?;

    // Optionally resume from an existing model
    // trainer.load_model("./models/japanese.model").await?;

    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./model.model"))?;

    println!("Accuracy:  {:.2}%", metrics.accuracy);
    println!("Precision: {:.2}%", metrics.precision);
    println!("Recall:    {:.2}%", metrics.recall);

    Ok(())
}
```

## PerceptronTrainer

`PerceptronTrainer` is the generic Averaged Perceptron counterpart of
`Trainer`: it trains a multiclass **Averaged Perceptron** over opaque
string labels from a features file (`litsea train --perceptron`). Its main
use is training the 2-class (`B`/`O`) boundary perceptron that the collapse
recipe (see [Pre-trained
Models](../pre-trained-models.md#training-procedure)) turns into the
bundled AdaBoost-format segmentation models.

### `PerceptronTrainer::new`

```rust
pub fn new(num_epochs: usize, features_path: &Path) -> litsea::Result<Self>
```

Reads the features file (each line is `label\tfeature1\tfeature2\t...`,
where labels are opaque strings, e.g. the boundary labels `B`/`O`) and
registers the training instances.

### `PerceptronTrainer::load_model`

```rust
pub async fn load_model(&mut self, model_uri: &str) -> litsea::Result<()>
```

Loads an existing perceptron model for incremental training. Classes
already registered from the training data are merged with the model's
classes.

### `PerceptronTrainer::train`

```rust
pub fn train(
    &mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<MulticlassMetrics>
```

Trains for the configured number of epochs, saves the model, and returns
multiclass metrics (accuracy, macro precision, macro recall). The `running`
flag enables graceful interruption, like `Trainer::train`.

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::PerceptronTrainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let mut trainer = PerceptronTrainer::new(10, Path::new("./features.txt"))?;
    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./perceptron.model"))?;
    println!("Accuracy: {:.2}%", metrics.accuracy);
    Ok(())
}
```

## TwoStageTrainer

`TwoStageTrainer` trains the [two-stage
model](../algorithm/two-stage-tagging.md) (issue #147): a binary boundary
classifier (stage 1) plus a word-level multiclass tagger (stage 2), both
**Averaged Perceptron**s, assembled with a candidate-tag lexicon into a
single `litsea-two-stage v1` file. After training, stage 1 is collapsed to
scalar per-feature weights in the existing AdaBoost format (a lossless
transformation -- see this module's source docs for the derivation), so the
runtime scores it exactly as it scores a plain `segment()` model. Both
`TwoStageTrainer` and `TwoStageMetrics` are re-exported from the crate root
as `litsea::TwoStageTrainer` / `litsea::TwoStageMetrics`.

### `TwoStageTrainer::new`

```rust
pub fn new(
    num_epochs: usize,
    dominance: f64,
    features_prefix: &Path,
) -> litsea::Result<Self>
```

Reads the three files written by
[`Extractor::extract_two_stage`](extractor.md) from `features_prefix`
(`{prefix}.stage1`, `{prefix}.stage2`, `{prefix}.lexicon`) and registers the
training instances for both stages.

`dominance` is the classifier-skip threshold of the assembled model: a
known word whose most frequent tag covers at least this fraction of its
training occurrences is tagged without invoking the stage-2 classifier. It
must be in `(0.5, 1.0]` and is validated eagerly in `new()`, so an
out-of-range value fails immediately rather than after training runs.

```rust
use std::path::Path;
use litsea::trainer::TwoStageTrainer;

let trainer = TwoStageTrainer::new(
    50,                            // num_epochs (both stages)
    0.99,                          // dominance
    Path::new("./features"),       // features prefix
)?;
```

### `TwoStageTrainer::train`

```rust
pub fn train(
    mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<TwoStageMetrics>
```

Unlike `Trainer::train` and `PerceptronTrainer::train`, this method takes `self`
by value (it consumes the trainer). It trains both stages as Averaged
Perceptrons for `num_epochs` epochs each, collapses stage 1 to AdaBoost
weights, assembles the two stages with the lexicon into a
`litsea-two-stage v1` model, saves it to `model_path`, and returns the
in-sample metrics of both stages. The `running` flag enables graceful
interruption, like the other trainers.

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::TwoStageTrainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let trainer = TwoStageTrainer::new(50, 0.99, Path::new("./features"))?;
    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./model.model"))?;

    println!("Stage 1: {:.2}%, Stage 2: {:.2}%", metrics.stage1.accuracy, metrics.stage2.accuracy);

    Ok(())
}
```

### `TwoStageMetrics`

```rust
pub struct TwoStageMetrics {
    pub stage1: MulticlassMetrics,
    pub stage2: MulticlassMetrics,
}
```

The in-sample metrics of a `TwoStageTrainer::train` run. `stage1` measures
the boundary classifier over its two classes (`B`/`O`); `stage2` measures
the word-level tagger over the UPOS tag classes. Both fields are
`MulticlassMetrics` -- the same type
[`PerceptronTrainer::train`](#perceptrontrainer) returns above, exposing
accuracy plus macro-averaged precision and recall.
