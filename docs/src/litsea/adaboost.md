# AdaBoost

The `AdaBoost` struct implements binary classification for word boundary detection.

## Definition

```rust
pub struct AdaBoost {
    pub threshold: f64,
    pub num_iterations: usize,
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

## Model Loading

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

Loads model weights from a URI. Supports:

- Local file path: `./resources/japanese.model`
- File URI: `file:///path/to/model`
- HTTP: `http://example.com/model`
- HTTPS: `https://example.com/model`

```rust
learner.load_model("./resources/japanese.model").await?;
learner.load_model("https://example.com/model").await?;
```

### `save_model`

```rust
pub fn save_model(&self, filename: &Path) -> io::Result<()>
```

Saves model weights to a file. Returns an error if the model is empty.

## Training Methods

### `initialize_features`

```rust
pub fn initialize_features(&mut self, filename: &Path) -> io::Result<()>
```

Reads a features file and builds the feature index. Must be called before `initialize_instances`.

### `initialize_instances`

```rust
pub fn initialize_instances(&mut self, filename: &Path) -> io::Result<()>
```

Reads the same features file and initializes labeled instances with their weights.

### `train`

```rust
pub fn train(&mut self, running: Arc<AtomicBool>)
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
pub fn predict(&self, attributes: HashSet<String>) -> i8
```

Predicts the label for a given feature set. Returns `+1` (boundary) or `-1` (non-boundary).

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:は".to_string());
attrs.insert("UC4:I".to_string());
// ... more features

let label = learner.predict(attrs);
// label == 1 (boundary) or -1 (non-boundary)
```

### `get_bias`

```rust
pub fn get_bias(&self) -> f64
```

Returns the bias term: `-sum(all model weights) / 2.0`.

## Evaluation

### `get_metrics`

```rust
pub fn get_metrics(&self) -> Metrics
```

Calculates evaluation metrics on the training data.

## Metrics

```rust
pub struct Metrics {
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
