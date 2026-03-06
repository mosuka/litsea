# Litsea

Litsea is an extremely compact word segmentation and POS (Part-of-Speech) tagging software implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker). Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries but instead performs segmentation and POS tagging using compact pre-trained models. It features a fast and safe Rust implementation along with learners designed to be simple and highly extensible.

## Key Features

- **Word Segmentation** using AdaBoost binary classification on character n-gram features
- **POS Tagging** using Averaged Perceptron with UPOS (Universal POS) tagset from [Universal Dependencies](https://universaldependencies.org/u/pos/) (17 tags)
- **Multilingual Support** for Japanese, Korean, and Chinese
- **Backward Compatible** — existing segmentation-only workflows continue to work as before

There is a small plant called Litsea cubeba (Aomoji) in the same camphoraceae family as Lindera (Kuromoji). This is the origin of the name Litsea.

## How to build Litsea

Litsea is implemented in Rust. To build it, follow these steps:

### Prerequisites

- Install Rust (stable channel) from [rust-lang.org](https://www.rust-lang.org/).
- Ensure Cargo (Rust’s package manager) is available.

### Build Instructions

1. **Clone the Repository**

   If you haven't already cloned the repository, run:

   ```sh
   git clone https://github.com/mosuka/litsea.git
   cd litsea
   ```

2. **Obtain Dependencies and Build**

   In the project's root directory, run:

   ```sh
   cargo build --release
   ```

   The `--release` flag produces an optimized build.

3. **Verify the Build**

   Once complete, the executable will be in the `target/release` folder. Verify by running:

   ```sh
   ./target/release/litsea --help
   ```

### Additional Notes

- Using the latest stable Rust ensures compatibility with dependencies and allows use of modern features.
- Run `cargo update` to refresh your dependencies if needed.

## How to train models

Prepare a corpus with words separated by spaces in advance.

- corpus.txt

    ```text
    Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。

    ```

Extract the information and features from the corpus. Use the `-l` flag to specify the language (`japanese`, `korean`, or `chinese`):

```sh
./target/release/litsea extract -l japanese ./corpus.txt ./features.txt
```

The output from the `extract` command is similar to:

```text
Feature extraction completed successfully.
```

Train the features output by the above command using AdaBoost. Use `-t` to set the weak classifier accuracy threshold and `-i` to set the maximum number of iterations:

```sh
./target/release/litsea train -t 0.005 -i 1000 ./features.txt ./resources/japanese.model
```

The output from the `train` command is similar to:

```text
Result Metrics:
  Accuracy: 94.15% ( 564133 / 599198 )
  Precision: 95.57% ( 330454 / 345758 )
  Recall: 94.36% ( 330454 / 350215 )
  Confusion Matrix:
    True Positives: 330454
    False Positives: 15304
    False Negatives: 19761
    True Negatives: 233679
```

## How to segment sentences into words

Use the trained model to segment sentences. Specify the language with `-l` and the model file:

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | ./target/release/litsea segment -l japanese ./resources/japanese.model
```

The output will look like:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```

For Korean and Chinese:

```sh
echo "한국어 단어 분할 테스트입니다." | ./target/release/litsea segment -l korean ./resources/korean.model
echo "中文分词测试。" | ./target/release/litsea segment -l chinese ./resources/chinese.model
```

## How to segment sentences with POS tagging

Litsea supports joint word segmentation and POS tagging using the `--pos` flag. POS tags follow the [UPOS tagset](https://universaldependencies.org/u/pos/) from Universal Dependencies (17 tags).

Use the pre-trained POS model to segment sentences with POS tags:

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | ./target/release/litsea segment --pos -l japanese ./resources/japanese_pos.model
```

The output will look like:

```text
Litsea/PROPN は/ADP TinySegmenter/PROPN を/ADP 参考/NOUN に/ADP 開発/VERB さ/AUX れ/AUX た/AUX 、/PUNCT Rust/PROPN で/ADP 実装/VERB さ/AUX れ/AUX た/AUX 極めて/ADV コンパクト/ADJ な/AUX 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

## How to train POS models

POS model training uses [Universal Dependencies](https://universaldependencies.org/) Treebanks as training data. The workflow consists of three steps: convert CoNLL-U data, extract features, and train.

### Step 1: Convert CoNLL-U to Litsea corpus format

Download a UD Treebank (e.g., [UD_Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD)) and convert the CoNLL-U file:

```sh
./target/release/litsea convert-conllu ./ja_gsd-ud-train.conllu ./corpus_pos.txt
```

The output will look like:

```text
Converted 7125 sentences.
```

### Step 2: Extract POS features

Use the `--pos` flag with the `extract` command to extract features from the POS corpus:

```sh
./target/release/litsea extract --pos -l japanese ./corpus_pos.txt ./features_pos.txt
```

### Step 3: Train the POS model

Use the `--pos` flag with the `train` command to train an Averaged Perceptron model. Use `--num-epochs` to set the number of training epochs:

```sh
./target/release/litsea train --pos --num-epochs 10 ./features_pos.txt ./resources/japanese_pos.model
```

The output from the `train` command is similar to:

```text
Result Metrics (POS):
  Accuracy: 98.34% ( 12486 )
  Macro Precision: 93.21%
  Macro Recall: 89.45%
```

## How to split text into sentences

Use the `split-sentences` subcommand to split text into sentences using Unicode UAX #29 rules. Each input line is treated as a paragraph and split into individual sentences:

```sh
echo "これはテストです。次の文です。" | ./target/release/litsea split-sentences
```

The output will look like:

```text
これはテストです。
次の文です。
```

## Pre-trained models

- **japanese.model**
  Trained on Japanese Wikipedia corpus using Lindera (UniDic) tokenization. Accuracy: 94.15%.

- **korean.model**
  Trained on Korean Wikipedia corpus using Lindera (ko-dic) tokenization. Accuracy: 85.08%.

- **chinese.model**
  Trained on Chinese Wikipedia corpus using Lindera (CC-CEDICT) tokenization. Accuracy: 80.72%.

- **japanese_pos.model**
  Joint word segmentation and POS tagging model trained on [UD Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD) Treebank using Averaged Perceptron. Accuracy: 98.34%.

- **JEITA_Genpaku_ChaSen_IPAdic.model**
  This model is trained using the morphologically analyzed corpus published by the Japan Electronics and Information Technology Industries Association (JEITA). It employs data from Project Sugita Genpaku analyzed with ChaSen+IPAdic.

- **RWCP.model**
  Extracted from the original [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/), this model contains only the segmentation component.

## How to retrain existing models

You can further improve performance by resuming training from an existing model with new corpora:

```sh
./target/release/litsea train -t 0.005 -i 1000 -m ./resources/japanese.model ./new_features.txt ./resources/japanese.model
```

## License

This project is distributed under the MIT License.  
It also contains code originally developed by Taku Kudo and released under the BSD 3-Clause License.  
See the LICENSE file for details.
