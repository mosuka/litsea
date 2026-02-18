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

## What's Next

- [CLI Reference](../cli-reference/overview.md) -- learn all CLI commands and options
- [Training Guide](../training-guide/preparing-corpus.md) -- train your own models
- [Architecture](../architecture/overview.md) -- understand how Litsea works internally
