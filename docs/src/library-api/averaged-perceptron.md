# Averaged Perceptron

The `AveragedPerceptron` struct implements multiclass classification for joint word segmentation and POS tagging.

## Definition

```rust
pub struct AveragedPerceptron {
    // internal fields: weight vectors, class names, cumulative weights, etc.
}
```

## Constructor

### `AveragedPerceptron::new`

```rust
pub fn new() -> Self
```

Creates a new empty Averaged Perceptron instance.

```rust
use litsea::perceptron::AveragedPerceptron;

let mut learner = AveragedPerceptron::new();
```

## Model Loading

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

Loads model weights from a URI. Supports the same URI schemes as `AdaBoost::load_model`:

- Local file path: `./resources/japanese_pos.model`
- File URI: `file:///path/to/model`
- HTTP/HTTPS: `https://example.com/model`

```rust
learner.load_model("./resources/japanese_pos.model").await?;
```

### `save_model`

```rust
pub fn save_model(&self, filename: &Path) -> io::Result<()>
```

Saves model weights to a file. The format includes class names followed by feature-class-weight triples.

## Training Methods

### `initialize_features`

```rust
pub fn initialize_features(&mut self, filename: &Path) -> io::Result<()>
```

Reads a POS features file and builds the feature and class indices. Must be called before `initialize_instances`.

### `initialize_instances`

```rust
pub fn initialize_instances(&mut self, filename: &Path) -> io::Result<()>
```

Reads the same features file and initializes labeled instances.

### `train`

```rust
pub fn train(&mut self, num_epochs: usize, running: Arc<AtomicBool>)
```

Runs the Averaged Perceptron training loop for the given number of epochs. Set `running` to `false` to stop early.

## Prediction

### `predict`

```rust
pub fn predict(&self, attributes: HashSet<String>) -> String
```

Predicts the class label for a given feature set. Returns one of 18 segment labels (`B-NOUN`, `B-VERB`, ..., `O`).

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:は".to_string());
attrs.insert("UC4:I".to_string());
// ... more features

let label = learner.predict(attrs);
// label == "B-ADP", "O", etc.
```

## Evaluation

### `get_metrics`

```rust
pub fn get_metrics(&self) -> PosMetrics
```

Calculates evaluation metrics on the training data.

## PosMetrics

```rust
pub struct PosMetrics {
    pub accuracy: f64,          // Overall accuracy in percentage
    pub macro_precision: f64,   // Macro-averaged precision in percentage
    pub macro_recall: f64,      // Macro-averaged recall in percentage
    pub num_instances: usize,
}
```
