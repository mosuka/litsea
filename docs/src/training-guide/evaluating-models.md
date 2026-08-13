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

The bundled models are trained with `-t 0.0001 -i 20000` and evaluated on
the held-out test split of their training treebank. **Word F1** scores
exact word matches; **Boundary F1** scores individual boundary decisions.

| Model | Word F1 | Boundary F1 | Training Corpus |
|-------|---------|-------------|-----------------|
| japanese.model | 91.48% | 96.31% | UD Japanese-GSD |
| korean.model | 99.91% | 99.96% | UD Korean-GSD |
| chinese.model | 77.56% | 87.81% | UD Chinese-GSD |

Korean is trained and evaluated on text that preserves the original
inter-eojeol spaces (space-preserving TSV corpus; space tokens are excluded
from the F1 computation). Since spaces mark most Korean word boundaries,
this makes the task far easier than for Japanese and Chinese, which are
written without spaces — the scores are not directly comparable across
languages.

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
