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
| `-e`, `--num-epochs <NUM_EPOCHS>` | `10` | Number of training epochs (POS mode only) |

## Output

Training metrics are printed to stderr:

```text
Result Metrics:
  Accuracy: 94.15% ( 564133 / 599198 )
  Precision: 95.57% ( 330454 / 345758 )
  Recall: 94.36% ( 330454 / 350215 )
  Confusion Matrix:
    True Positives: 330454
    False Positives: 15304
    False Negatives: 19761
    True Negatives: 233679
```

## Ctrl+C Handling

Training supports graceful interruption:

- **First Ctrl+C**: Stops training and saves the model at its current state
- **Second Ctrl+C**: Exits immediately without saving

This allows you to stop long-running training sessions without losing progress.

## Examples

Basic training:

```sh
litsea train -t 0.005 -i 1000 ./features.txt ./models/japanese.model
```

Training with higher precision (lower threshold, more iterations):

```sh
litsea train -t 0.001 -i 5000 ./features.txt ./model.model
```

Retraining from an existing model:

```sh
litsea train -t 0.005 -i 1000 -m ./models/japanese.model \
    ./new_features.txt ./models/japanese_v2.model
```

## Hyperparameter Tuning

| Parameter | Effect of Decreasing | Effect of Increasing |
|-----------|---------------------|---------------------|
| `threshold` | More iterations, potentially higher accuracy, longer training time | Fewer iterations, faster training, may underfit |
| `num_iterations` | Fewer boosting rounds, smaller model, may underfit | More rounds, larger model, potentially higher accuracy |

## POS Model Training

When the `--pos` flag is specified, `train` uses the **Averaged Perceptron** algorithm instead of AdaBoost. This trains a multiclass classifier for joint word segmentation and POS tagging.

### POS Training Options

| Option | Default | Description |
|--------|---------|------------|
| `--pos` | off | Enable POS training mode |
| `-e`, `--num-epochs <NUM_EPOCHS>` | `10` | Number of training epochs |

### POS Training Output

```text
Result Metrics:
  Accuracy: 98.34%
  Macro Precision: 97.87%
  Macro Recall: 91.67%
```

### POS Training Example

```sh
# Train a POS model from POS features
litsea train --pos -e 10 ./pos_features.txt ./models/japanese_pos.model
```

### POS Hyperparameters

| Parameter | Effect of Decreasing | Effect of Increasing |
|-----------|---------------------|---------------------|
| `num_epochs` | Faster training, may underfit | Better accuracy, longer training, may overfit |
