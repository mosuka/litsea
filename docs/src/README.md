# Introduction

**Litsea** is an extremely compact word segmentation library implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker).

Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries. Instead, it performs word segmentation using a compact pre-trained **binary classification** model stored in the AdaBoost model format (the bundled segmentation models are trained as 2-class Averaged Perceptrons and converted to that format without loss). Litsea also supports **word segmentation and POS (Part-of-Speech) tagging** with the [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tagset via a two-stage architecture.

## Key Features

- **Fast and safe Rust implementation** -- built with Rust's safety guarantees and performance
- **Compact pre-trained models** -- the `japanese`/`chinese`/`korean`/`english.model` segmentation models are ~86 KB-2.0 MB, small enough to embed directly in applications or serve over HTTP
- **No dictionary required** -- segmentation is driven by a statistical model; an optional user dictionary (`UserDictionary`, `segment --user-words`) forces words of your choice into single tokens without retraining
- **Two-stage POS tagging** -- segments with a binary boundary classifier and tags each word via a candidate-tag lexicon plus a word-level tagger, adding little cost over plain segmentation
- **Multilingual support** -- Japanese, Chinese (Simplified/Traditional), Korean, and English
- **Model training capabilities** -- train custom models using AdaBoost or Averaged Perceptron with your own corpora
- **Remote model loading** -- load models from HTTP/HTTPS URLs (opt-in `remote_model` feature) or local files
- **Simple and extensible API** -- easy to integrate into Rust projects as a library

## How It Works

Litsea treats word segmentation as a **binary classification problem**: for each character position in a sentence, the model predicts whether it is a **word boundary** (+1) or **not a boundary** (-1). The classifier uses character n-gram features and character type information specific to each language.

```text
Input:  "これはテストです。"
         こ れ は テ ス ト で す 。
         B  O  B  B  O  O  B  O  B   ← word-start predictions (japanese.model)
Output: ["これ", "は", "テスト", "です", "。"]
```

### POS Tagging

Litsea also supports **POS (Part-of-Speech) tagging** in addition to word segmentation, through the two-stage architecture: the sentence is segmented by a binary boundary classifier, then each word is tagged through a candidate-tag lexicon plus a word-level tagger.

Stage 1 labels each character position `B` (a word starts here) or `O` (continuation), and stage 2 tags each resulting word with one of the 17 UPOS tags. The earlier joint model, which predicted one of 18 `B-<UPOS>` / `O` labels per character, was removed; loading such a model file fails with an error that asks you to retrain with `litsea train --pos`.

The POS tags follow the [Universal Dependencies](https://universaldependencies.org/) **UPOS tagset** (17 POS tags).

```text
Input:  "今日はいい天気ですね。"
Output: 今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

## Name Origin

There is a small plant called *Litsea cubeba* (Aomoji) in the same Lauraceae family as *Lindera* (Kuromoji). This is the origin of the name **Litsea**.

## Current Version

Litsea v0.15.0 -- Rust Edition 2024, minimum Rust version 1.87.

## Links

- [GitHub Repository](https://github.com/mosuka/litsea)
- [crates.io](https://crates.io/crates/litsea)
- [API Documentation (docs.rs)](https://docs.rs/litsea)
- [Japanese Documentation (日本語)](../ja/)
