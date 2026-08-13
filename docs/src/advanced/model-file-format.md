# Model File Format

Litsea models are stored as simple plain-text files.

## Format Specification

```text
<feature_name>\t<weight>
<feature_name>\t<weight>
...
<bias>
```

- Each line (except the last) contains a **feature name** and its **weight**, separated by a tab character
- **Zero-weight features** are omitted to keep the file compact
- The **last line** contains the bias term as a single number

## Example

```text
BC1:IK	0.3456
BC2:KI	-0.1234
UW4:は	0.5678
UC4:I	0.2345
...
-0.0891
```

## Bias Reconstruction

When loading a model, the bias is reconstructed using:

```text
bias_bucket_weight = -bias_value * 2 - sum(feature_weights_before_the_bias_line)
```

Files written by `save_model` always place the bias line last, so this equals
the sum of all feature weights. Legacy models (e.g. `RWCP.model`) place the
bias line mid-file; their trailing weight lines are accepted and the bias
bucket is computed from the weights preceding the bias line, matching the
historical loader.

## Validation

The loader rejects malformed files with an explicit error instead of loading
them silently:

- an **empty file**
- a file **without a bias line** (the typical symptom of a truncated
  download or interrupted copy)
- **more than one** bias line
- **duplicate** feature lines
- **non-finite** weights or bias values (`NaN`, `inf`, `-inf`), which would
  otherwise poison every score comparison

The Averaged Perceptron model loader likewise validates its class-count
header and rejects non-finite weights.

During prediction:

```text
bias = -sum(all_model_weights) / 2.0    (cached; read once per sentence)
score = bias + sum(model[feature] for feature in input_attributes)
```

The on-disk format is string-keyed and unchanged, but the segmenter does
not score against the strings directly: at load time each feature line is
parsed and compiled into a packed `u64` integer key for the hot loop (see
[Prediction Pipeline](../algorithm/prediction-pipeline.md#the-compiled-scoring-tables)).
Features that the segmenter's language could never generate (for example
type codes of another language) are ignored by that compilation -- exactly
as they could never match an input attribute before -- while the bias is
always computed over every weight in the file.

## File Size

Model files are very compact:

| Model | Size | Features |
|-------|------|----------|
| japanese.model | ~20 KB | UD Japanese-GSD |
| korean.model | ~20 KB | UD Korean-GSD |
| chinese.model | ~18 KB | UD Chinese-GSD |
| RWCP.model | ~22 KB | Original TinySegmenter |
| japanese_pos.model | ~11 MB | UD Japanese-GSD (POS) |
| chinese_pos.model | ~19 MB | UD Chinese-GSD (POS) |
| korean_pos.model | ~8.9 MB | UD Korean-GSD (POS) |
| JEITA_Genpaku_ChaSen_IPAdic.model | ~16 KB | JEITA corpus |

The compact size of the word-segmentation models is a key advantage of Litsea -- they can be embedded directly in applications or served over HTTP with minimal overhead. The joint segmentation + POS models are larger (megabytes) because they carry per-class weights.

## Compatibility

- Model files are **encoding-agnostic** (feature names are stored as-is)
- The format is **deterministic** for the usual training workflow: `save_model` writes features in the learner's feature order, which is sorted (via `BTreeMap`) for learners initialized from a features file or loaded from disk. A learner populated only via `add_instance()` writes features in insertion order instead
- Models are **forward-compatible** -- new features in the input that are not in the model are simply ignored during prediction
