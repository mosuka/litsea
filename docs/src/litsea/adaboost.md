# AdaBoost

The `AdaBoost` struct implements binary classification for word boundary detection.

## Definition

```rust
pub struct AdaBoost {
    // private: threshold: f64, num_iterations: usize
    // (read via threshold() / num_iterations())
    // internal fields: model weights, features, instances, etc.
}
```

## Constructor

### `AdaBoost::new`

```rust
pub fn new(threshold: f64, num_iterations: usize) -> Self
```

Creates a new AdaBoost instance with the specified hyperparameters.

```rust
use litsea::adaboost::AdaBoost;

let mut learner = AdaBoost::new(0.01, 100);
```

### `AdaBoost::default`

`AdaBoost` also implements `Default`, which is equivalent to `AdaBoost::new(0.01, 100)` -- the default hyperparameters used across the library and CLI.

## Accessors

### `threshold`

```rust
pub fn threshold(&self) -> f64
```

Returns the early-stopping threshold this learner was created with.

### `num_iterations`

```rust
pub fn num_iterations(&self) -> usize
```

Returns the maximum number of boosting iterations.

## Model Loading

### `load_model_from_path`

```rust
pub fn load_model_from_path(&mut self, path: &Path) -> litsea::Result<()>
```

Loads model weights from a local file, synchronously. Malformed files (empty, missing bias line, duplicate bias lines or features, non-finite weights) are rejected with `LitseaError::InvalidData`. This is the preferred method for local files -- no async runtime is needed.

```rust
use std::path::Path;

learner.load_model_from_path(Path::new("./models/japanese.model"))?;
```

### `load_model_from_reader`

```rust
pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> litsea::Result<()>
```

Loads model weights from any `BufRead` source, such as an in-memory buffer or an already-open file.

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> litsea::Result<()>
```

Loads model weights from a URI. Supports:

- Local file path: `./models/japanese.model`
- File URI: `file:///path/to/model`
- HTTP: `http://example.com/model` (requires the `remote_model` feature)
- HTTPS: `https://example.com/model` (requires the `remote_model` feature)

```rust
learner.load_model("https://example.com/model").await?;
```

### `save_model`

```rust
pub fn save_model(&self, filename: &Path) -> litsea::Result<()>
```

Saves model weights to a file. Returns an error if the model is empty.

### `save_model_to_writer`

```rust
pub fn save_model_to_writer<W: Write>(&self, writer: &mut W) -> litsea::Result<()>
```

Writes the model to an arbitrary writer in the same text format as
`save_model`; this is the format-producing core `save_model` delegates to.
It is public so the model can be embedded as a section of a larger file
without going through a file path -- the [two-stage model
format](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)
uses it to embed the stage-1 `AdaBoost` model directly. The writer is not
flushed. Returns an error if the model is empty.

## Training Methods

### `initialize_features`

```rust
pub fn initialize_features(&mut self, filename: &Path) -> litsea::Result<()>
```

Reads a features file and builds the feature index. Must be called before `initialize_instances`.

### `initialize_instances`

```rust
pub fn initialize_instances(&mut self, filename: &Path) -> litsea::Result<()>
```

Reads the same features file and initializes labeled instances with their weights.

### `train`

```rust
pub fn train(&mut self, running: &AtomicBool)
```

Runs the AdaBoost training loop. Set `running` to `false` to stop early.

### `add_instance`

```rust
pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8)
```

Adds a single training instance with its feature set and label.

## Prediction

### `predict`

```rust
pub fn predict(&self, attributes: &HashSet<String>) -> i8
```

Predicts the label for a given feature set. Returns `+1` (boundary) or `-1` (non-boundary).

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:は".to_string());
attrs.insert("UC4:I".to_string());
// ... more features

let label = learner.predict(&attrs);
// label == 1 (boundary) or -1 (non-boundary)
```

### `bias`

```rust
pub fn bias(&self) -> f64
```

Returns the bias term: `-sum(all model weights) / 2.0`. The value is cached and kept in sync by every weight-mutating path, so this is O(1).

## Evaluation

### `metrics`

```rust
pub fn metrics(&self) -> BinaryMetrics
```

Calculates evaluation metrics on the training data.

## BinaryMetrics

Defined in `litsea::metrics` (also re-exported as `litsea::BinaryMetrics`):

```rust
pub struct BinaryMetrics {
    pub accuracy: f64,          // Accuracy in percentage
    pub precision: f64,         // Precision in percentage
    pub recall: f64,            // Recall in percentage
    pub num_instances: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
}
```
