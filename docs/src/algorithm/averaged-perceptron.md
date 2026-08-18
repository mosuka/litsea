# Averaged Perceptron

Litsea uses the **Averaged Perceptron** algorithm for multiclass classification to perform joint word segmentation and POS tagging. This chapter explains the algorithm as implemented in Litsea.

## Overview

While [AdaBoost](adaboost.md) performs **binary classification** (boundary vs. non-boundary), the Averaged Perceptron performs **multiclass classification** -- predicting one of 18 segment labels for each character position:

- **17 boundary labels**: `B-ADJ`, `B-ADP`, `B-ADV`, `B-AUX`, `B-CCONJ`, `B-DET`, `B-INTJ`, `B-NOUN`, `B-NUM`, `B-PART`, `B-PRON`, `B-PROPN`, `B-PUNCT`, `B-SCONJ`, `B-SYM`, `B-VERB`, `B-X`
- **1 non-boundary label**: `O` (continuation of the current word)

These labels correspond to the 17 [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tags from the Universal Dependencies project, prefixed with `B-` to indicate a word boundary. This enables simultaneous word boundary detection and POS estimation in a single classification step.

This page describes Averaged Perceptron's primary use as an 18-class joint segmentation + POS tagger. The same algorithm, used in 2-class mode (`B`/`O` labels only), is what actually trains the bundled `japanese.model`, `chinese.model`, and `korean.model` segmentation models before they are losslessly collapsed to the AdaBoost model format for inference -- see [Pre-trained Models: Training Procedure](../pre-trained-models.md#training-procedure) for the full derivation. It is also the algorithm behind the [two-stage architecture](two-stage-tagging.md)'s stage-2 word-level tagger.

## Algorithm

### Weight Representation

The perceptron maintains a **per-class weight vector for each feature**, stored in a single sparse map so that scoring and weight updates need one hashed lookup per feature (not one per feature x class):

```text
slots: FxHashMap<Feature, FeatureSlot>
// FeatureSlot { w: Vec<f64>, acc: Vec<f64>, ts: Vec<usize> } -- one entry per class
```

For example:

```text
weights["UW4:猫"]["B-NOUN"] = 2.5
weights["UC4:H"]["B-NOUN"]  = 1.8
weights["UW4:猫"]["O"]      = -0.3
...
```

For a given feature set, the score for each class is the sum of its feature weights:

```text
score(class) = sum(weights[feature][class] for each feature in input)
prediction = argmax(score(class) for all classes)
```

### Update Rule

When the perceptron makes a misclassification:

```text
For each training instance (features, truth):
    guess = predict(features)

    if guess != truth:
        For each feature f in features:
            weights[f][truth] += 1.0   # increase weight for correct class
            weights[f][guess] -= 1.0   # decrease weight for predicted class
```

This increases the weights for the correct class and decreases them for the incorrectly predicted class, making the correct prediction more likely for similar inputs in the future.

### Averaging

A key improvement over the basic perceptron is **weight averaging**. Rather than using the final weights (which can be unstable and tend to overfit to the tail of the training data), the model averages all weight vectors seen during training. This improves generalization to unseen data.

The implementation uses a **cumulative sum** approach for efficiency:

```text
cumulative[feature][class] += weights[feature][class] * elapsed_steps

At the end of training:
    averaged[feature][class] = cumulative[feature][class] / total_steps
```

This avoids storing all intermediate weight vectors while producing the same result. The averaging reduces dependence on the order of training data and improves generalization performance.

The accumulator and timestamp vectors are materialized lazily, the first time training touches a feature -- a model loaded for inference only carries live weights and pays nothing for averaging state.

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

```rust
use std::sync::atomic::AtomicBool;
use litsea::perceptron::AveragedPerceptron;

let mut perceptron = AveragedPerceptron::new();
// ... add instances ...
let running = AtomicBool::new(true);
perceptron.train(10, &running);  // 10 epochs
```

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
- Weight lines are written in sorted feature order, so saving the same model
  always produces byte-identical files; loading does not depend on the order

## Comparison with AdaBoost

| Aspect | AdaBoost | Averaged Perceptron |
|--------|----------|---------------------|
| Classification | Binary (+1/-1) | Multiclass (18 classes) |
| Output | Word boundaries only | Word boundaries + POS tags |
| Weak learner | Decision stumps per feature | None (linear classifier) |
| Weight management | One weight per feature | Class x feature weight matrix |
| Generalization | Ensemble | Weight averaging |
| Training | Iterative boosting with sample reweighting | Online learning with weight averaging |
| Model size | ~86 KB-2.0 MB (retrained japanese/chinese/korean.model) / ~16-22 KB (legacy RWCP/JEITA) | ~9-19 MB (bundled joint POS models) / ~5-8 MB (two-stage models) |
| Hyperparameters | `threshold`, `num_iterations` | `num_epochs` |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10 | Number of training passes over the data. More epochs can improve accuracy but may overfit |
