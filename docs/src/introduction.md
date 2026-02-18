# Introduction

**Litsea** is an extremely compact word segmentation library implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker).

Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries. Instead, it performs word segmentation using a compact pre-trained model based on the **AdaBoost binary classification** algorithm.

## Key Features

- **Fast and safe Rust implementation** -- built with Rust's safety guarantees and performance
- **Compact pre-trained models** -- model files are only a few kilobytes in size
- **No dictionary dependency** -- segmentation is driven entirely by a statistical model
- **Multilingual support** -- Japanese, Chinese (Simplified/Traditional), and Korean
- **Model training capabilities** -- train custom models using AdaBoost with your own corpora
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

## Name Origin

There is a small plant called *Litsea cubeba* (Aomoji) in the same Lauraceae family as *Lindera* (Kuromoji). This is the origin of the name **Litsea**.

## Current Version

Litsea v0.4.0 -- Rust Edition 2024, minimum Rust version 1.87.

## Links

- [GitHub Repository](https://github.com/mosuka/litsea)
- [crates.io](https://crates.io/crates/litsea)
- [API Documentation (docs.rs)](https://docs.rs/litsea)
- [Japanese Documentation (日本語)](../ja/)
