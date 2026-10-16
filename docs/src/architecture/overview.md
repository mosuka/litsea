# Architecture Overview

Litsea is designed as a compact word segmentation system that needs no dictionary. It treats word segmentation as a **binary classification problem** and uses **AdaBoost** to learn word boundary patterns from character-level features.

## High-Level Data Flow

Litsea has two main workflows: **training** and **segmentation**.

### Training Pipeline

```mermaid
flowchart LR
    A["Corpus (text)"] --> B["Extractor"]
    B --> C["Features File (.txt)"]
    C --> D["Trainer (AdaBoost)"]
    D --> E["Model File (.model)"]
```

1. **Corpus preparation** -- Prepare text with words separated by spaces
2. **Feature extraction** -- The `Extractor` reads the corpus, classifies characters by type, and outputs labeled feature vectors
3. **Model training** -- The `Trainer` feeds features into AdaBoost, which iteratively selects the most informative features and produces a compact model

### Segmentation Pipeline

```mermaid
flowchart LR
    F["Raw text"] --> G["Segmenter (AdaBoost)"]
    H["Model file"] --> G
    G --> I["Segmented words"]
```

1. **Model loading** -- Load a pre-trained model (from file or URL)
2. **Character classification** -- For each character in the input, determine its type code based on language-specific patterns
3. **Feature extraction** -- Stream character n-gram features for each position through a reused buffer using a sliding window
4. **Prediction** -- AdaBoost predicts whether each position is a word boundary

## Design Principles

- **No dictionary required** -- Unlike MeCab or Lindera, Litsea segments with a statistical model learned from character patterns alone; an optional user dictionary can force specific words into single tokens on top of it (`segment --user-words`, issue #263)
- **Compact models** -- the word-segmentation models `japanese`/`chinese`/`korean`/`english.model` are ~86 KB-2.0 MB (see [Pre-trained Models](../pre-trained-models.md) for their accuracy); two-stage POS models are ~3.6-8 MB -- all still small enough to embed directly in applications, containing only the feature weights that matter
- **Language-agnostic framework** -- The core algorithm is the same for all languages; only the character type patterns differ
- **Simple extensibility** -- Adding a new language requires only defining character type patterns and training a model
