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

## Two-Stage Model Format (`litsea-two-stage v1`)

A two-stage model bundles a stage-1 boundary classifier, a candidate-tag
lexicon, and a stage-2 word-level tagger into a single plain-text file with
a magic first line and marker-delimited sections in a fixed order:

```text
litsea-two-stage v1
[params]
dominance\t0.99
[stage1]
<AdaBoost model format: "feature\tweight" lines + one bias line>
[lexicon]
<surface>\t<TAG>:<count>[,<TAG>:<count>...]
[stage2]
<Averaged Perceptron model format: class count, class names, weights>
```

- The `[stage1]` and `[stage2]` sections embed the existing formats
  described above verbatim and are parsed by the existing loaders.
- Each `[lexicon]` line maps a word surface to the UPOS tags observed for
  it in the training corpus with their occurrence counts, most frequent
  first (ties broken by tag name). Surfaces may contain any character
  except tab and newline and are not trimmed, so whitespace tokens stay
  representable.
- The `[params]` section is optional. Its only key, `dominance`, is the
  classifier-skip threshold in `(0.5, 1.0]`: a known surface whose most
  frequent tag covers at least this fraction of its training occurrences is
  tagged without invoking the stage-2 classifier. It defaults to `0.99`
  when the section is absent.
- Stage-2 class names must be valid UPOS tags; together with every weight
  and lexicon line containing a tab, this guarantees no content line can
  collide with a section marker.

The format is **purely additive**: the magic line is neither a valid
AdaBoost weight/bias line nor a perceptron class count, so the existing
loaders reject two-stage files with an explicit error, and existing model
files keep loading unchanged. A future format revision will use a different
magic line (e.g. `litsea-two-stage v2`); the v1 loader rejects it as an
unsupported version. The loader validates section order, the lexicon rules
above, and the parameter range, and reports errors with the section name
(e.g. `[stage2] section: ...`).

## File Size

Model file sizes vary considerably by model type and language:

| Model | Size | Features |
|-------|------|----------|
| japanese.model | ~1.1 MB | UD Japanese-GSD |
| chinese.model | ~2.0 MB | UD Chinese-GSD |
| korean.model | ~86 KB | UD Korean-GSD |
| RWCP.model | ~22 KB | Original TinySegmenter |
| JEITA_Genpaku_ChaSen_IPAdic.model | ~16 KB | JEITA corpus |
| japanese_pos.model | ~5.4 MB | UD Japanese-GSD (two-stage) |
| chinese_pos.model | ~8.0 MB | UD Chinese-GSD (two-stage) |
| korean_pos.model | ~5.0 MB | UD Korean-GSD (two-stage) |

`RWCP.model` and `JEITA_Genpaku_ChaSen_IPAdic.model` are genuinely tiny (kilobyte-scale) and are the easiest to embed directly in applications or serve over HTTP with minimal overhead. The retrained `japanese.model`, `chinese.model`, and `korean.model` (see [Pre-trained Models](../pre-trained-models.md)) trade some of that compactness for substantial quality gains: they are now ~86 KB-2.0 MB rather than kilobyte-scale, though still small compared to the multi-megabyte two-stage (`*_pos.model`) models, which are larger because they carry per-class and per-stage weights.

## Compatibility

- Model files are **encoding-agnostic** (feature names are stored as-is)
- The format is **deterministic** for the usual training workflow: `save_model` writes features in the learner's feature order, which is sorted (via `BTreeMap`) for learners initialized from a features file or loaded from disk. A learner populated only via `add_instance()` writes features in insertion order instead
- Models are **forward-compatible** -- new features in the input that are not in the model are simply ignored during prediction
