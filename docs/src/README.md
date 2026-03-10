# Introduction

**Litsea** is an extremely compact word segmentation library implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker).

Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries. Instead, it performs word segmentation using a compact pre-trained model based on the **AdaBoost binary classification** algorithm. Litsea also supports **joint word segmentation and POS (Part-of-Speech) tagging** using the **Averaged Perceptron** multiclass classifier with the [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tagset.

## Key Features

- **Fast and safe Rust implementation** -- built with Rust's safety guarantees and performance
- **Compact pre-trained models** -- model files are only a few kilobytes in size
- **No dictionary dependency** -- segmentation is driven entirely by a statistical model
- **POS tagging** -- joint segmentation and Part-of-Speech tagging with UPOS tags via Averaged Perceptron
- **Multilingual support** -- Japanese, Chinese (Simplified/Traditional), and Korean
- **Model training capabilities** -- train custom models using AdaBoost or Averaged Perceptron with your own corpora
- **Remote model loading** -- load models from HTTP/HTTPS URLs or local files
- **Simple and extensible API** -- easy to integrate into Rust projects as a library

## How It Works

Litsea treats word segmentation as a **binary classification problem**: for each character position in a sentence, the model predicts whether it is a **word boundary** (+1) or **not a boundary** (-1). The classifier uses character n-gram features and character type information specific to each language.

```text
Input:  "LitseaはRust製です"
         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
         O O O O B O B O B   ← boundary predictions
Output: ["Litsea", "は", "Rust製", "です"]
```

### POS Tagging

Litsea also supports **POS (Part-of-Speech) tagging** in addition to word segmentation. Using the **Averaged Perceptron** multiclass classifier, it performs joint segmentation and POS tagging simultaneously.

For each character position, the model predicts one of 18 **SegmentLabel** classes:

- `B-NOUN`, `B-VERB`, ..., `B-X` (boundary labels for 17 POS tags)
- `O` (non-boundary = continuation of the current word)

The POS tags follow the [Universal Dependencies](https://universaldependencies.org/) **UPOS tagset** (17 POS tags).

```text
Input:  "今日はいい天気ですね。"
Output: 今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

## Name Origin

There is a small plant called *Litsea cubeba* (Aomoji) in the same Lauraceae family as *Lindera* (Kuromoji). This is the origin of the name **Litsea**.

## Current Version

Litsea v0.4.0 -- Rust Edition 2024, minimum Rust version 1.87.

## Links

- [GitHub Repository](https://github.com/mosuka/litsea)
- [crates.io](https://crates.io/crates/litsea)
- [API Documentation (docs.rs)](https://docs.rs/litsea)
- [Japanese Documentation (日本語)](../ja/)
