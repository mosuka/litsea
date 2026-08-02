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
bias = -sum(all_model_weights) / 2.0
score = bias + sum(model[feature] for feature in input_attributes)
```

## File Size

Model files are very compact:

| Model | Size | Features |
|-------|------|----------|
| japanese.model | ~2.9 KB | Wikipedia-trained |
| korean.model | ~1.8 KB | Wikipedia-trained |
| chinese.model | ~1.3 KB | Wikipedia-trained |
| RWCP.model | ~22 KB | Original TinySegmenter |
| JEITA_Genpaku_ChaSen_IPAdic.model | ~17 KB | JEITA corpus |

The compact size is a key advantage of Litsea -- models can be embedded directly in applications or served over HTTP with minimal overhead.

## Compatibility

- Model files are **encoding-agnostic** (feature names are stored as-is)
- The format is **deterministic** (features are sorted via BTreeMap)
- Models are **forward-compatible** -- new features in the input that are not in the model are simply ignored during prediction
