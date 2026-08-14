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
| `--num-epochs <NUM_EPOCHS>` | `10` | Number of training epochs (POS and `--two-stage` modes) |
| `--two-stage` | off | Train a [two-stage](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1) model instead. Reads `{FEATURES_FILE}.stage1`/`.stage2`/`.lexicon` (from `extract --two-stage`). Cannot be combined with `--pos` or `-m`/`--load-model-uri` (incremental training is not supported) |
| `--dominance <DOMINANCE>` | `0.99` | Classifier-skip threshold for `--two-stage`, in `(0.5, 1.0]`: a known word whose most frequent tag covers at least this fraction of its training occurrences is tagged without invoking the stage-2 classifier |

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

## Two-Stage Model Training

With `--two-stage`, `train` builds a [two-stage
model](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1):
a binary boundary classifier (stage 1) plus a word-level POS tagger
(stage 2), assembled with the candidate-tag lexicon into a single
`litsea-two-stage v1` file. Both stages train as Averaged Perceptrons for
`--num-epochs` epochs; stage 1 is then collapsed to scalar weights in the
existing AdaBoost format (a lossless transformation — see the module docs
of `litsea::trainer` for the derivation) so the runtime scores it exactly
as it scores a plain `segment()` model.

### Usage

```sh
litsea train --two-stage [OPTIONS] <FEATURES_PREFIX> <MODEL_FILE>
```

`FEATURES_PREFIX` is the same prefix passed to `extract --two-stage`.

### Example

```sh
litsea extract --two-stage -l japanese ./pos_corpus.txt ./two_stage_features
litsea train --two-stage --num-epochs 50 ./two_stage_features ./models/japanese_two_stage.model
```

### Output

```text
Result Metrics (Two-Stage):
  Stage 1 (boundary) Accuracy: 99.36% ( 277213 )
  Stage 1 Macro Precision: 99.30%
  Stage 1 Macro Recall: 99.35%
  Stage 2 (tagging) Accuracy: 98.53% ( 168333 )
  Stage 2 Macro Precision: 98.39%
  Stage 2 Macro Recall: 97.59%
```

As with the other modes, these are in-sample metrics; evaluate on held-out
text with `litsea evaluate --pos` for a realistic quality estimate. `segment
--pos` and `evaluate --pos` auto-detect a two-stage model from its file
header, so no extra flag is needed to use it.
