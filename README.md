# Litsea

Litsea is an extremely compact word segmentation software implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker). Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries but instead performs segmentation using a compact pre-trained model. It features a fast and safe Rust implementation along with a learner designed to be simple and highly extensible.

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
finding instances...: 599198 instances found
loading instances...: 599198/599198 instances loaded
Iteration 999 - margin: 0.005765994648498989
Result Metrics:
  Accuracy: 94.10% ( 563824 / 599198 )
  Precision: 95.44% ( 330556 / 346271 )
  Recall: 94.38% ( 330556 / 350215 )
  Confusion Matrix:
    True Positives: 330556
    False Positives: 15715
    False Negatives: 19659
    True Negatives: 233268
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

## Pre-trained models

- **japanese.model**
  Trained on Japanese Wikipedia corpus using Lindera (UniDic) tokenization. Accuracy: 94.10%.

- **korean.model**
  Trained on Korean Wikipedia corpus using Lindera (ko-dic) tokenization. Accuracy: 84.40%.

- **chinese.model**
  Trained on Chinese Wikipedia corpus using Lindera (CC-CEDICT) tokenization. Accuracy: 80.58%.

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
