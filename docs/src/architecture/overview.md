# Architecture Overview

Litsea is designed as a compact, dictionary-free word segmentation system. It treats word segmentation as a **binary classification problem** and uses **AdaBoost** to learn word boundary patterns from character-level features.

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

- **No dictionary dependency** -- Unlike MeCab or Lindera, Litsea relies solely on a statistical model learned from character patterns
- **Compact models** -- Legacy word-segmentation models (`RWCP.model`, `JEITA_Genpaku_ChaSen_IPAdic.model`) are ~16-22 KB; the retrained `japanese`/`chinese`/`korean.model` are 110 KB-2.0 MB, trading file size for the quality gains documented in [Pre-trained Models](../pre-trained-models.md); two-stage POS models are ~5-8 MB and joint POS models are ~9-19 MB -- all still small enough to embed directly in applications, containing only the feature weights that matter
- **Language-agnostic framework** -- The core algorithm is the same for all languages; only the character type patterns differ
- **Simple extensibility** -- Adding a new language requires only defining character type patterns and training a model
