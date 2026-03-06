# Averaged Perceptron

Litsea uses the **Averaged Perceptron** algorithm for multiclass classification to perform joint word segmentation and POS tagging. This chapter explains the algorithm as implemented in Litsea.

## Overview

While [AdaBoost](adaboost.md) performs **binary classification** (boundary vs. non-boundary), the Averaged Perceptron performs **multiclass classification** -- predicting one of 18 segment labels for each character position:

- **17 boundary labels**: `B-ADJ`, `B-ADP`, `B-ADV`, `B-AUX`, `B-CCONJ`, `B-DET`, `B-INTJ`, `B-NOUN`, `B-NUM`, `B-PART`, `B-PRON`, `B-PROPN`, `B-PUNCT`, `B-SCONJ`, `B-SYM`, `B-VERB`, `B-X`
- **1 non-boundary label**: `O` (continuation of the current word)

These labels correspond to the 17 [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tags from the Universal Dependencies project, prefixed with `B-` to indicate a word boundary.

## Algorithm

### Weight Representation

The perceptron maintains a **weight vector per class**. Weights are stored as a sparse map:

```text
weights: HashMap<Feature, HashMap<Class, f64>>
```

For a given feature set, the score for each class is the sum of its feature weights:

```text
score(class) = sum(weights[feature][class] for each feature in input)
prediction = argmax(score(class) for all classes)
```

### Update Rule

When the perceptron makes a misclassification:

```text
For each feature in the input:
    weights[feature][correct_class] += 1.0
    weights[feature][predicted_class] -= 1.0
```

This increases the weights for the correct class and decreases them for the incorrectly predicted class, making the correct prediction more likely for similar inputs in the future.

### Averaging

A key improvement over the basic perceptron is **weight averaging**. Rather than using the final weights (which can be unstable), the model averages all weight vectors seen during training. This improves generalization to unseen data.

The implementation uses a **cumulative sum** approach for efficiency:

```text
cumulative[feature][class] += weights[feature][class]  (after each update)

At the end of training:
    averaged[feature][class] = cumulative[feature][class] / total_updates
```

This avoids storing all intermediate weight vectors while producing the same result.

### Training with Epochs

Training iterates over the data multiple times (epochs). Each epoch processes all training instances in order:

```text
For each epoch (1 to num_epochs):
    For each instance in training data:
        features = extract_features(instance)
        predicted = argmax(score(class) for all classes)
        if predicted != correct_label:
            update weights
        accumulate weights for averaging
```

Training supports graceful interruption via `AtomicBool` -- a Ctrl+C signal stops training and saves the model at its current state.

## Model File Format

The Averaged Perceptron model is saved as a text file with the following structure:

```text
18
O
B-ADJ
B-ADP
...
B-X
feature1\tclass1\tweight1
feature2\tclass2\tweight2
...
```

- **Line 1**: Number of classes (18)
- **Lines 2 to N+1**: Class names, one per line
- **Remaining lines**: Feature weights, tab-separated as `feature\tclass\tweight`
- Zero-weight entries are omitted

## Comparison with AdaBoost

| Aspect | AdaBoost | Averaged Perceptron |
|--------|----------|---------------------|
| Classification | Binary (+1/-1) | Multiclass (18 classes) |
| Output | Word boundaries only | Word boundaries + POS tags |
| Training | Iterative boosting with sample reweighting | Online learning with weight averaging |
| Model size | A few KB | ~11 MB (with POS features) |
| Hyperparameters | `threshold`, `num_iterations` | `num_epochs` |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10 | Number of training passes over the data. More epochs can improve accuracy but may overfit |
