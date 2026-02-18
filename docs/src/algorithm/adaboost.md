# AdaBoost Binary Classification

Litsea uses the **AdaBoost** (Adaptive Boosting) algorithm for binary classification to determine word boundaries. This chapter explains the algorithm as implemented in Litsea.

## Overview

AdaBoost combines many **weak learners** (simple classifiers) into a strong ensemble classifier. In Litsea:

- **Positive label (+1)** = word boundary
- **Negative label (-1)** = non-boundary (continuation of the current word)
- **Weak learners** = individual features (each feature is a binary "stump" -- present or absent)

## Training Algorithm

The training loop in `AdaBoost::train()` works as follows:

### Initialization

1. Load features and instances from the training file
2. Initialize instance weights uniformly (later adjusted based on initial score)
3. All model weights start at zero

### Iterative Boosting

For each iteration *t* (up to `num_iterations`):

**Step 1: Calculate weighted errors**

For each feature *h*, compute its weighted error over all instances:

```text
error[h] -= D[i] * y[i]   (for each instance i that has feature h)
```

where *D[i]* is the instance weight and *y[i]* is the true label.

**Step 2: Select the best weak learner**

Find the feature with the lowest weighted error rate:

```text
error_rate(h) = (error[h] + positive_weight_sum) / instance_weight_sum
h_best = argmax_h |0.5 - error_rate(h)|
```

The baseline competitor is the "all-negative" classifier (always predicts -1), whose error rate equals the fraction of positive instances. Any real feature must beat this baseline.

**Step 3: Check convergence**

If `|0.5 - best_error_rate| < threshold`, stop early -- no feature can significantly improve the model.

**Step 4: Compute the weak learner weight**

```text
alpha = 0.5 * ln((1 - error_rate) / error_rate)
model[h_best] += alpha
```

A lower error rate produces a higher alpha, giving more influence to better features.

**Step 5: Update instance weights**

```text
For each instance i:
    prediction = +1 if h_best in features(i), else -1

    if y[i] * prediction < 0:  (misclassified)
        D[i] *= exp(alpha)     (increase weight)
    else:                       (correctly classified)
        D[i] /= exp(alpha)     (decrease weight)

Normalize: D[i] /= sum(D)
```

This ensures subsequent iterations focus on the instances that are still difficult to classify.

## Prediction

Given an input set of features (attributes), the prediction is:

```text
score = bias + sum(model[feature] for each feature in attributes)
prediction = +1 if score >= 0, else -1
```

### Bias Term

The bias is computed as:

```text
bias = -sum(all model weights) / 2.0
```

This centers the decision boundary. The empty-string feature (`""`) serves as the bias bucket during training.

## Model File Format

The trained model is saved as a simple text file:

```text
feature1\tweight1
feature2\tweight2
...
bias_value
```

- Each line contains a feature name and its weight (tab-separated)
- Zero-weight features are omitted
- The last line contains the bias term (a single number)

See [Model File Format](../advanced/model-file-format.md) for details.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.01 | Early stopping threshold. Lower values allow more iterations, potentially improving accuracy |
| `num_iterations` | 100 | Maximum number of boosting rounds. Higher values may improve accuracy at the cost of training time and model size |
