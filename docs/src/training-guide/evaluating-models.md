# Evaluating Models

Understanding model quality is essential for producing good segmentation results.

## Metrics

The `train` command outputs three key metrics after training:

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

| Model | Accuracy | Precision | Recall | Training Corpus |
|-------|----------|-----------|--------|-----------------|
| japanese.model | 94.15% | 95.57% | 94.36% | Wikipedia (Lindera UniDic) |
| korean.model | 85.08% | -- | -- | Wikipedia (Lindera ko-dic) |
| chinese.model | 80.72% | -- | -- | Wikipedia (Lindera CC-CEDICT) |

## Improving Model Quality

If accuracy is unsatisfactory, consider:

1. **More training data** -- A larger and more diverse corpus
2. **Lower threshold** -- Try `-t 0.001` to allow more boosting iterations
3. **More iterations** -- Try `-i 5000` or higher
4. **Better corpus quality** -- Ensure consistent tokenization and clean text
5. **Retraining** -- Start from an existing model and train with additional data (see [Retraining Models](retraining-models.md))
