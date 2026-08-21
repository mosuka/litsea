# Evaluating Models

Understanding model quality is essential for producing good segmentation results.

## Metrics

The `train` command outputs three key metrics after training. These are
**in-sample** metrics: they are measured on the training data itself, so
they overestimate how the model performs on unseen text. For a realistic
picture, always evaluate on a held-out corpus that was not used for
training (see the benchmarks below).

### Accuracy

```text
Accuracy = (TP + TN) / Total Instances
```

The percentage of all character positions that were correctly classified (both boundaries and non-boundaries). This is the broadest measure of model quality.

### Precision

```text
Precision = TP / (TP + FP)
```

Of the boundaries the model **predicted**, what fraction was **correct**. High precision means few false boundaries (over-segmentation).

### Recall

```text
Recall = TP / (TP + FN)
```

Of the **actual** boundaries, what fraction did the model **find**. High recall means few missed boundaries (under-segmentation).

## Confusion Matrix

| | Predicted Boundary (+1) | Predicted Non-boundary (-1) |
|---|---|---|
| **Actual Boundary** | True Positive (TP) | False Negative (FN) |
| **Actual Non-boundary** | False Positive (FP) | True Negative (TN) |

## Pre-trained Model Benchmarks

The bundled `japanese.model`, `chinese.model`, `korean.model`, and
`english.model` are trained with a binary-perceptron-collapse procedure,
not plain AdaBoost `-t`/`-i` training -- see [Training
Procedure](../pre-trained-models.md#training-procedure) for the exact
recipe. All are evaluated on the held-out test split of their training
treebank. **Word F1** scores exact word matches; **Boundary F1** scores
individual boundary decisions.

| Model | Word F1 | Boundary F1 | Training Corpus |
|-------|---------|-------------|-----------------|
| japanese.model | 96.70% | 98.59% | UD Japanese-GSD |
| korean.model | 99.91% | 99.96% | UD Korean-GSD |
| chinese.model | 90.69% | 95.64% | UD Chinese-GSD |
| english.model | 98.31% | 99.18% | UD English-EWT |

Korean and English are trained and evaluated on text that preserves the
original spaces (space-preserving TSV corpus; space tokens are excluded
from the F1 computation). Since spaces mark most word boundaries in these
two languages, this makes the task easier than for Japanese and Chinese,
which are written without spaces — the scores are not directly comparable
across languages. Korean's near-deterministic 99.91% and English's lower
98.31% both come from this same space-preserving protocol; the remaining
gap is residual ambiguity English keeps even with spaces present
(contractions, hyphenated compounds, abbreviations).

### Reproducing the Benchmarks

Every figure in the table above is reproducible with one command using the
bundled gold data (`resources/eval/`, converted from the UD GSD **test**
splits — held-out for the bundled models, which are trained on the train
splits):

```sh
litsea evaluate -l japanese models/japanese.model resources/eval/japanese_gsd_test.txt
litsea evaluate -l korean --format tsv models/korean.model resources/eval/korean_gsd_test.tsv
litsea evaluate -l chinese models/chinese.model resources/eval/chinese_gsd_test.txt
litsea evaluate -l english --format tsv models/english.model resources/eval/english_ewt_test.tsv
```

See [evaluate](../litsea-cli/evaluate.md) for the command reference. POS
models are evaluated with `--pos`; their held-out figures are listed in
[Pre-trained Models](../pre-trained-models.md).

Which gold file to use depends on how the model was trained, and the two
groups differ:

```sh
# Japanese / Chinese: real text has no spaces, so the space-separated
# `word/POS` gold is also the real-world protocol.
litsea evaluate --pos -l japanese models/japanese_pos.model resources/eval/japanese_gsd_test_pos.txt

# Korean / English: trained on the space-preserving corpus (issue #198),
# so evaluate against the space-preserving POS gold with --format tsv.
litsea evaluate --pos --format tsv -l korean models/korean_pos.model resources/eval/korean_gsd_test_pos_spaced.tsv
litsea evaluate --pos --format tsv -l english models/english_pos.model resources/eval/english_ewt_test_pos_spaced.tsv
```

The `*_test_pos.txt` files (no space tokens) are still present for Korean
and English. They measure the *unspaced* protocol those models were
trained on before #198, so they are useful only for reproducing the older
published numbers -- for current quality, use the `*_pos_spaced.tsv` gold
above, which matches both how the models train and what `segment --pos`
receives in practice.

## Improving Model Quality

If accuracy is unsatisfactory, consider:

1. **More training data** -- A larger and more diverse corpus
2. **Lower threshold** -- Try `-t 0.0001` to allow more boosting iterations
3. **More iterations** -- Try `-i 20000` or higher. AdaBoost selects one
   weak learner (feature) per iteration, so the number of iterations caps
   how many features the model can use; the CLI default (`-i 100`) produces
   very small models with much lower held-out accuracy
4. **Better corpus quality** -- Ensure consistent tokenization and clean text
5. **Retraining** -- Start from an existing model and train with additional data (see [Retraining Models](retraining-models.md))

The threshold/iteration tuning above applies to plain AdaBoost training
(`litsea train` without `--perceptron`/`--pos`). The bundled models' own
+5-13pt held-out quality gains over plain AdaBoost did not come from
tuning `-t`/`-i` -- they came from training a 2-class Averaged Perceptron
and collapsing it losslessly to AdaBoost weights instead. If you are
chasing bundled-model-level quality rather than incremental gains, see
[Training Procedure](../pre-trained-models.md#training-procedure) for that
recipe.
