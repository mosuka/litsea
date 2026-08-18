# Litsea

Litsea is an extremely compact word segmentation and POS (Part-of-Speech) tagging software implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker). Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries but instead performs segmentation and POS tagging using compact pre-trained models. It features a fast and safe Rust implementation along with learners designed to be simple and highly extensible.

## Key Features

- **Word Segmentation** using AdaBoost binary classification on character n-gram features
- **POS Tagging** using Averaged Perceptron with UPOS (Universal POS) tagset from [Universal Dependencies](https://universaldependencies.org/u/pos/) (17 tags), available in two architectures: joint (`--pos`) and the faster two-stage (`--two-stage`)
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
./target/release/litsea train -t 0.0001 -i 20000 ./features.txt ./models/my_model.model
```

(The bundled `japanese.model`/`chinese.model`/`korean.model` are *not* produced this way — see [Pre-trained models](#pre-trained-models) below for their actual training procedure.)

The `train` command reports metrics computed on the training data (with enough iterations the model can fit the training corpus almost perfectly; evaluate on held-out text for a realistic quality estimate):

```text
Result Metrics:
  Accuracy: 100.00% ( 1075868 / 1075869 )
  Precision: 100.00% ( 161283 / 161284 )
  Recall: 100.00% ( 161283 / 161283 )
  Confusion Matrix:
    True Positives: 161283
    False Positives: 1
    False Negatives: 0
    True Negatives: 914585
```

## How to segment sentences into words

Use a trained model to segment sentences. Specify the language with `-l` and the model file. Here we use the bundled `RWCP.model` (the original TinySegmenter model):

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | ./target/release/litsea segment -l japanese ./models/RWCP.model
```

The output is:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```

For Korean and Chinese:

```sh
echo "한국어 단어 분할 테스트입니다." | ./target/release/litsea segment -l korean ./models/korean.model
echo "中文分词测试。" | ./target/release/litsea segment -l chinese ./models/chinese.model
```

## How to segment sentences with POS tagging

Litsea supports joint word segmentation and POS tagging using the `--pos` flag. POS tags follow the [UPOS tagset](https://universaldependencies.org/u/pos/) from Universal Dependencies (17 tags). The `--pos` flag works with both joint (`*_pos.model`) and two-stage (`*_two_stage.model`, see below) model files — the model kind is auto-detected from the file, so the command is identical either way.

Use the pre-trained POS model to segment sentences with POS tags:

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | ./target/release/litsea segment --pos -l japanese ./models/japanese_pos.model
```

The output is:

```text
Litsea/NOUN は/ADP Tiny/PROPN Segmenter/NOUN を/ADP 参考/NOUN に/ADP 開発/VERB さ/AUX れ/AUX た/AUX 、/PUNCT Rust/NOUN で/ADP 実装/VERB さ/AUX れ/AUX た/AUX 極めて/NOUN コンパクト/NOUN な/AUX 単語/NOUN 分割/NOUN ソフトウェア/NOUN です/AUX 。/PUNCT
```

## How to train POS models

POS model training uses [Universal Dependencies](https://universaldependencies.org/) Treebanks as training data. The workflow consists of three steps: prepare corpus, extract features, and train.

### Step 1: Prepare corpus from UD Treebank

Use `scripts/download_udtreebank.sh` to download a UD Treebank and `scripts/corpus_udtreebank.sh` to convert it to Litsea corpus format:

```sh
# Download UD Treebank and get CoNLL-U file path
conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)

# Generate word segmentation corpus
bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt

# Generate POS corpus
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
```

Supported languages: `ja` (Japanese, default), `ko` (Korean), `zh` (Chinese).

### Step 2: Extract POS features

Use the `--pos` flag with the `extract` command to extract features from the POS corpus:

```sh
./target/release/litsea extract --pos -l japanese ./pos_corpus.txt ./features_pos.txt
```

### Step 3: Train the POS model

Use the `--pos` flag with the `train` command to train an Averaged Perceptron model. Use `--num-epochs` to set the number of training epochs:

```sh
./target/release/litsea train --pos --num-epochs 10 ./features_pos.txt ./models/japanese_pos.model
```

The output from the `train` command is similar to:

```text
Result Metrics (POS):
  Accuracy: 98.23% ( 277213 )
  Macro Precision: 96.82%
  Macro Recall: 93.30%
```

## How to train two-stage models

For faster POS tagging, use `--two-stage` instead of `--pos`. It trains a binary boundary classifier (stage 1) plus a word-level tagger (stage 2), assembled with a candidate-tag lexicon into a single `litsea-two-stage v1` model file:

```sh
./target/release/litsea extract --two-stage -l japanese ./pos_corpus.txt ./two_stage_features
./target/release/litsea train --two-stage --num-epochs 50 ./two_stage_features ./models/japanese_two_stage.model
```

`segment --pos` and `evaluate --pos` auto-detect a two-stage model from its file header, so no extra flag is needed to use one once trained. See [Two-Stage vs. Joint Tagging](docs/src/algorithm/two-stage-tagging.md) for the architecture and measured quality/speed comparison.

## How to split text into sentences

Use the `scripts/split_sentences.sh` shell script to split text into sentences using regex-based rules. Each input line is treated as a paragraph and split into individual sentences:

```sh
echo "これはテストです。次の文です。" | bash scripts/split_sentences.sh -l ja
```

The `-l` flag is currently accepted but unused; the splitting rules are language-independent.

The output will look like:

```text
これはテストです。
次の文です。
```

## Pre-trained models

- **japanese.model**
  Trained on the [UD Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD) Treebank as a 2-class Averaged Perceptron, losslessly collapsed to AdaBoost-format scalar weights (see [Pre-trained Models](docs/src/pre-trained-models.md) for the procedure). Held-out word F1: 96.70%.

- **korean.model**
  Trained on the [UD Korean-GSD](https://github.com/UniversalDependencies/UD_Korean-GSD) Treebank with a space-preserving corpus, same collapsed-perceptron procedure as above but tag-free (`extract --tag-free`), making it pointwise so `segment` skips its sequential scoring pass. Held-out word F1: 99.91%.

- **chinese.model**
  Trained on the [UD Chinese-GSD](https://github.com/UniversalDependencies/UD_Chinese-GSD) Treebank, same collapsed-perceptron procedure as above. Held-out word F1: 90.69%.

- **japanese_pos.model**
  Joint word segmentation and POS tagging model trained on the [UD Japanese-GSD](https://github.com/UniversalDependencies/UD_Japanese-GSD) Treebank using Averaged Perceptron. Held-out word/tagged-word F1: 96.56% / 92.51%.

- **chinese_pos.model**
  Joint word segmentation and POS tagging model trained on the [UD Chinese-GSD](https://github.com/UniversalDependencies/UD_Chinese-GSD) Treebank using Averaged Perceptron. Held-out word/tagged-word F1: 90.52% / 81.18%.

- **korean_pos.model**
  Joint word segmentation and POS tagging model trained on the [UD Korean-GSD](https://github.com/UniversalDependencies/UD_Korean-GSD) Treebank using Averaged Perceptron. Held-out word/tagged-word F1: 80.51% / 71.03%.

- **japanese_two_stage.model** / **chinese_two_stage.model** / **korean_two_stage.model**
  Two-stage word segmentation and POS tagging models (see [How to train two-stage models](#how-to-train-two-stage-models)). As bundled, every language beats the corresponding joint model above on both word and tagged-word F1: Japanese 96.78% / 92.95%, Chinese 90.82% / 82.29%, Korean 83.24% / 78.86%.

- **JEITA_Genpaku_ChaSen_IPAdic.model**
  This model is trained using the morphologically analyzed corpus published by the Japan Electronics and Information Technology Industries Association (JEITA). It employs data from Project Sugita Genpaku analyzed with ChaSen+IPAdic.

- **RWCP.model**
  Extracted from the original [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/), this model contains only the segmentation component.

## How to retrain existing models

You can further improve performance by resuming training from an existing model with new corpora:

```sh
./target/release/litsea train -t 0.0001 -i 20000 -m ./models/my_model.model ./new_features.txt ./models/my_model.model
```

(This incremental-retraining path applies to plain AdaBoost models. The bundled `japanese.model`/`chinese.model`/`korean.model` are retrained from scratch via the collapsed-perceptron procedure instead, not incrementally. `train --two-stage` does not support `-m`/incremental training at all.)

## License

This project is distributed under the MIT License.  
It also contains code originally developed by Taku Kudo and released under the BSD 3-Clause License.  
See the LICENSE file for details.
