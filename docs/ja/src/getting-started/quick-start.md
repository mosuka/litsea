# クイックスタート

## CLI クイックスタート

### テキストの分割

Litsea には `models/` ディレクトリに学習済みモデルが同梱されています。テキストを `segment` コマンドにパイプで渡します:

**日本語:**

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" \
  | litsea segment -l japanese ./models/japanese.model
```

出力:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```

**中国語:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./models/chinese.model
```

**韓国語:**

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./models/korean.model
```

### 品詞推定付き分割

`--pos` フラグを付けると、単語分割と同時に UPOS 品詞タグを推定します:

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./models/japanese_pos.model
```

出力:

```text
今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

## ライブラリ クイックスタート

モデルを読み込みテキストを分割する最小限の Rust プログラムです:

```rust
use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the pre-trained model
    let mut learner = AdaBoost::new(0.01, 100);
    learner.load_model("./models/japanese.model").await?;

    // Create a segmenter
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));

    // Segment text
    let tokens = segmenter.segment("これはテストです。");
    println!("{}", tokens.join(" "));
    // Output: これ は テスト です 。

    Ok(())
}
```

## ライブラリ クイックスタート（品詞推定）

品詞推定付きモデルを読み込み、単語分割と品詞推定を同時に行う例です:

```rust
use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // POS モデルを読み込み
    let mut pos_learner = AveragedPerceptron::new();
    pos_learner.load_model("./models/japanese_pos.model").await?;

    // POS 対応 Segmenter を作成
    let segmenter = Segmenter::with_pos_learner(Language::Japanese, pos_learner);

    // 品詞推定付き分割
    let tokens = segmenter.segment_with_pos("これはテストです。");
    for (word, pos) in &tokens {
        print!("{}/{} ", word, pos);
    }
    println!();

    Ok(())
}
```

## 次のステップ

- [CLI リファレンス](../litsea-cli.md) -- すべての CLI コマンドとオプションの詳細
- [学習ガイド](../training-guide/preparing-corpus.md) -- 独自モデルの学習方法
- [アーキテクチャ](../architecture/overview.md) -- Litsea の内部動作の理解
