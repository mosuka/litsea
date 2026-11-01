# Averaged Perceptron

Litsea uses the **Averaged Perceptron** algorithm for multiclass classification as the training-side learner behind both stages of the two-stage POS architecture and the bundled segmentation models' collapse recipe. This chapter explains the algorithm as implemented in Litsea.

## Overview

While [AdaBoost](adaboost.md) performs **binary classification** (boundary vs. non-boundary), the Averaged Perceptron performs **multiclass classification** over the class labels found in its training data. Litsea trains it with one of two label sets:

- **2 boundary labels** over the character-level templates: `B` (a word starts at this character position) and `O` (continuation of the current word)
- **The 17 [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tags** over the word-level templates, one tag per word: `ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`

POS tags come from the two-stage model: stage 1 is a binary boundary classifier that labels each character position `B` (a word starts here) or `O` (continuation), and stage 2 tags each resulting word with one of the 17 UPOS tags. The earlier joint model, which predicted one of 18 `B-<UPOS>` / `O` labels per character, was removed; loading such a model file fails with an error that asks you to retrain with `litsea train --pos`.

The Averaged Perceptron in 2-class mode (`B`/`O` labels only) is what actually trains the bundled `japanese.model`, `chinese.model`, `korean.model`, and `english.model` segmentation models before they are losslessly collapsed to the AdaBoost model format for inference -- see [Pre-trained Models: Training Procedure](../pre-trained-models.md#training-procedure) for the full derivation. The same collapse trains the [two-stage architecture](two-stage-tagging.md)'s stage-1 boundary classifier, and the multiclass form is the algorithm behind its stage-2 word-level tagger.

## Algorithm

### Weight Representation

The perceptron maintains a **per-class weight vector for each feature**, stored in a single sparse map so that scoring and weight updates need one hashed lookup per feature (not one per feature x class):

```text
slots: FxHashMap<Feature, FeatureSlot>
// FeatureSlot { w: Vec<f64>, acc: Vec<f64>, ts: Vec<usize> } -- one entry per class
```

For example:

```text
weights["UW4:猫"]["B"]    = 2.5     (boundary perceptron)
weights["UW4:猫"]["O"]    = -0.3
weights["WS:猫"]["NOUN"]  = 1.8     (stage-2 word tagger)
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

The Averaged Perceptron model is saved as a text file with the following structure (this example is the start of the `[stage2]` section of the bundled `japanese_pos.model`, whose tagger knows 16 UPOS tags):

```text
16
ADJ
ADP
...
VERB
feature1\tclass1\tweight1
feature2\tclass2\tweight2
...
```

- **Line 1**: Number of classes *N* (2 for a boundary perceptron; 16 or 17 for the bundled stage-2 taggers)
- **Lines 2 to N+1**: Class names, one per line, in sorted order (a boundary perceptron writes `B`, then `O`)
- **Remaining lines**: Feature weights, tab-separated as `feature\tclass\tweight`
- Zero-weight entries are omitted
- Weight lines are written in sorted feature order, so saving the same model
  always produces byte-identical files; loading does not depend on the order

## Comparison with AdaBoost

| Aspect | AdaBoost | Averaged Perceptron |
|--------|----------|---------------------|
| Classification | Binary (+1/-1) | Multiclass (2 classes `B`/`O`, or up to 17 UPOS tags) |
| Output | Word boundaries only | Word boundaries (`B`/`O`, collapsed to the AdaBoost format) or one UPOS tag per word |
| Weak learner | Decision stumps per feature | None (linear classifier) |
| Weight management | One weight per feature | Class x feature weight matrix |
| Generalization | Ensemble | Weight averaging |
| Training | Iterative boosting with sample reweighting | Online learning with weight averaging |
| Model size | ~86 KB-2.0 MB (japanese/chinese/korean/english.model) | ~3.1-8.3 MB (two-stage models) |
| Hyperparameters | `threshold`, `num_iterations` | `num_epochs` |

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10 | Number of training passes over the data. More epochs can improve accuracy but may overfit |
