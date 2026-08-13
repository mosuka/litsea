# train

Train a word segmentation model using AdaBoost.

## Usage

```sh
litsea train [OPTIONS] <FEATURES_FILE> <MODEL_FILE>
```

## Arguments

| Argument | Description |
|----------|------------|
| `FEATURES_FILE` | Path to the input features file (output from `extract`) |
| `MODEL_FILE` | Path to the output model file |

## Options

| Option | Default | Description |
|--------|---------|------------|
| `-t`, `--threshold <THRESHOLD>` | `0.01` | Weak classifier accuracy threshold for early stopping. Lower values allow more iterations |
| `-i`, `--num-iterations <NUM_ITERATIONS>` | `100` | Maximum number of boosting iterations |
| `-m`, `--load-model-uri <LOAD_MODEL_URI>` | None | URI of an existing model to resume training from (file path or HTTP/HTTPS URL) |
| `--pos` | off | Enable POS (Part-of-Speech) training mode using Averaged Perceptron |
| `--num-epochs <NUM_EPOCHS>` | `10` | Number of training epochs (POS mode only) |

## Output

Training metrics are printed to stderr:

Metrics are computed on the training data; with enough iterations the model can fit the training corpus almost perfectly, so evaluate on held-out text for a realistic quality estimate.

```text
Result Metrics:
  Accuracy: 100.00% ( 1075868 / 1075869 )
  Precision: 100.00% ( 161283 / 161284 )
  Recall: 100.00% ( 161283 / 161283 )
  Confusion Matrix:
    True Positives: 161283
    False Positives: 1
    False Negatives: 0
    True Negatives: 914585
```

## Ctrl+C Handling

Training supports graceful interruption:

- **First Ctrl+C**: Stops training and saves the model at its current state
- **Second Ctrl+C**: Exits immediately without saving

This allows you to stop long-running training sessions without losing progress.

## Examples

Basic training:

```sh
litsea train -t 0.0001 -i 20000 ./features.txt ./models/japanese.model
```

Training with higher precision (lower threshold, more iterations):

```sh
litsea train -t 0.001 -i 5000 ./features.txt ./model.model
```

Retraining from an existing model:

```sh
litsea train -t 0.0001 -i 20000 -m ./models/japanese.model \
    ./new_features.txt ./models/japanese_v2.model
```

## Hyperparameter Tuning

| Parameter | Effect of Decreasing | Effect of Increasing |
|-----------|---------------------|---------------------|
| `threshold` | More iterations, potentially higher accuracy, longer training time | Fewer iterations, faster training, may underfit |
| `num_iterations` | Fewer boosting rounds, smaller model, may underfit | More rounds, larger model, potentially higher accuracy |

## POS Model Training

When the `--pos` flag is specified, `train` uses the **Averaged Perceptron** algorithm instead of AdaBoost. This trains a multiclass classifier for joint word segmentation and POS tagging.

### Usage

```sh
litsea train --pos [OPTIONS] <FEATURES_FILE> <MODEL_FILE>
```

### POS Training Options

| Option | Default | Description |
|--------|---------|------------|
| `--pos` | off | Enable POS training mode |
| `--num-epochs <NUM_EPOCHS>` | `10` | Number of training epochs |

### Examples

```sh
# Train a POS model from POS features
litsea train --pos --num-epochs 10 ./pos_features.txt ./models/japanese_pos.model
```

### Output

POS training metrics are printed to stderr (macro-averaged precision and recall):

```text
Result Metrics (POS):
  Accuracy: 98.23% ( 277213 )
  Macro Precision: 96.82%
  Macro Recall: 93.30%
```

### Ctrl+C Handling

Same as AdaBoost training, POS training supports graceful interruption. The first Ctrl+C stops training and saves the model at its current state.

### POS Hyperparameters

| Parameter | Effect of Decreasing | Effect of Increasing |
|-----------|---------------------|---------------------|
| `num_epochs` | Faster training, may underfit | Better accuracy, longer training, may overfit |
