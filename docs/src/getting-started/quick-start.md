# Quick Start

## CLI Quick Start

### Segmenting Text

Litsea ships with pre-trained models in the `resources/` directory. Pipe text into the `segment` command:

**Japanese:**

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" \
  | litsea segment -l japanese ./resources/japanese.model
```

Output:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```

**Chinese:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./resources/chinese.model
```

**Korean:**

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./resources/korean.model
```

### POS Tagging

Litsea can perform joint word segmentation and POS tagging using a POS model. Add the `--pos` flag to the `segment` command:

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./resources/japanese_pos.model
```

Output:

```text
今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

Each token is annotated with a [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) tag.

### Splitting Sentences

Split text into sentences using Unicode UAX #29 rules:

```sh
echo "これはテストです。次の文です。" | litsea split-sentences
```

Output:

```text
これはテストです。
次の文です。
```

## Library Quick Start

Here is a minimal Rust program that loads a model and segments text:

```rust
use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the pre-trained model
    let mut learner = AdaBoost::new(0.01, 100);
    learner.load_model("./resources/japanese.model").await?;

    // Create a segmenter
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));

    // Segment text
    let tokens = segmenter.segment("これはテストです。");
    println!("{}", tokens.join(" "));
    // Output: これ は テスト です 。

    Ok(())
}
```

### POS Tagging with the Library

Here is a minimal Rust program that loads a POS model and segments text with POS tags:

```rust
use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the pre-trained POS model
    let mut pos_learner = AveragedPerceptron::new();
    pos_learner.load_model("./resources/japanese_pos.model").await?;

    // Create a segmenter with POS support
    let segmenter = Segmenter::with_pos_learner(Language::Japanese, pos_learner);

    // Segment text with POS tags
    let tokens = segmenter.segment_with_pos("今日はいい天気ですね。");
    for (word, pos) in &tokens {
        print!("{}/{} ", word, pos);
    }
    // Output: 今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT

    Ok(())
}
```

## What's Next

- [CLI Reference](../litsea-cli.md) -- learn all CLI commands and options
- [Training Guide](../training-guide/preparing-corpus.md) -- train your own models
- [Architecture](../architecture/overview.md) -- understand how Litsea works internally
