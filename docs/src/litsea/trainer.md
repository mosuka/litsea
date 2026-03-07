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
) -> io::Result<Self>
```

Creates a trainer and initializes it from a features file. This calls `AdaBoost::initialize_features()` and `AdaBoost::initialize_instances()`.

```rust
use std::path::Path;
use litsea::trainer::Trainer;

let mut trainer = Trainer::new(
    0.005,                           // threshold
    1000,                            // max iterations
    Path::new("./features.txt"),     // features file
)?;
```

## Methods

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

Loads an existing model for retraining. Supports file paths, `file://`, `http://`, and `https://` URIs.

```rust
trainer.load_model("./resources/japanese.model").await?;
```

### `train`

```rust
pub fn train(
    &mut self,
    running: Arc<AtomicBool>,
    model_path: &Path,
) -> Result<Metrics, Box<dyn std::error::Error>>
```

Trains the model and saves it to the specified path. Returns evaluation metrics.

The `running` flag enables graceful interruption -- set it to `false` to stop training early.

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::path::Path;

let running = Arc::new(AtomicBool::new(true));
let metrics = trainer.train(running, Path::new("./model.model"))?;

println!("Accuracy: {:.2}%", metrics.accuracy);
```

## Full Training Example

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::Trainer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = Trainer::new(
        0.005,
        1000,
        Path::new("./features.txt"),
    )?;

    // Optionally resume from an existing model
    // trainer.load_model("./resources/japanese.model").await?;

    let running = Arc::new(AtomicBool::new(true));
    let metrics = trainer.train(running, Path::new("./model.model"))?;

    println!("Accuracy:  {:.2}%", metrics.accuracy);
    println!("Precision: {:.2}%", metrics.precision);
    println!("Recall:    {:.2}%", metrics.recall);

    Ok(())
}
```
