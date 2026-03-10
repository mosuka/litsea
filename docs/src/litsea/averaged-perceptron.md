# Averaged Perceptron

The `AveragedPerceptron` struct implements multiclass classification for joint word segmentation and POS tagging.

## Definition

```rust
pub struct AveragedPerceptron {
    // internal fields: weights, accumulated, timestamps, step, classes, instances
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

## Adding Instances

### `add_instance`

```rust
pub fn add_instance(&mut self, features: HashSet<String>, label: String)
```

Adds a training instance with a feature set and a label. Unknown classes are automatically registered.

```rust
use std::collections::HashSet;
use litsea::perceptron::AveragedPerceptron;

let mut learner = AveragedPerceptron::new();
let mut feats = HashSet::new();
feats.insert("UW4:猫".to_string());
feats.insert("UC4:H".to_string());
learner.add_instance(feats, "B-NOUN".to_string());
```

## Training

### `train`

```rust
pub fn train(&mut self, num_epochs: usize, running: Arc<AtomicBool>)
```

Runs the Averaged Perceptron training loop for the given number of epochs. Set `running` to `false` to stop early. Weights are automatically averaged at the end of training.

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

let running = Arc::new(AtomicBool::new(true));
learner.train(10, running);
```

## Prediction

### `predict`

```rust
pub fn predict(&self, features: &HashSet<String>) -> String
```

Predicts the class label for a given feature set. Computes a score for each class and returns the class name with the highest score. Returns an empty string if no classes are registered.

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:は".to_string());
attrs.insert("UC4:I".to_string());
// ... more features

let label = learner.predict(&attrs);
// label == "B-ADP", "O", etc.
```

## Model I/O

### `save_model`

```rust
pub fn save_model(&self, path: &Path) -> io::Result<()>
```

Saves model weights to a file. Returns an error if the model is empty.

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

Loads model weights from a URI. Supports the following URI schemes:

- Local file path: `./models/japanese_pos.model`
- File URI: `file:///path/to/model`
- HTTP: `http://example.com/model`
- HTTPS: `https://example.com/model`

```rust
learner.load_model("./models/japanese_pos.model").await?;
```

## Evaluation

### `get_metrics`

```rust
pub fn get_metrics(&self) -> Metrics
```

Calculates evaluation metrics on the training data.

## Metrics

```rust
pub struct Metrics {
    pub accuracy: f64,                            // Overall accuracy in percentage
    pub macro_precision: f64,                     // Macro-averaged precision in percentage
    pub macro_recall: f64,                        // Macro-averaged recall in percentage
    pub num_instances: usize,                     // Number of instances
    pub correct_per_class: HashMap<String, usize>,   // Correct count per class
    pub predicted_per_class: HashMap<String, usize>,  // Predicted count per class
    pub gold_per_class: HashMap<String, usize>,       // Gold label count per class
}
```
