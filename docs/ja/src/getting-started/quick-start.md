# クイックスタート

## CLI クイックスタート

### テキストの分割

Litsea には `resources/` ディレクトリに学習済みモデルが同梱されています。テキストを `segment` コマンドにパイプで渡します:

**日本語:**

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" \
  | litsea segment -l japanese ./resources/japanese.model
```

出力:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```

**中国語:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./resources/chinese.model
```

**韓国語:**

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./resources/korean.model
```

### 文分割

Unicode UAX #29 規則を使用してテキストを文に分割します:

```sh
echo "これはテストです。次の文です。" | litsea split-sentences
```

出力:

```text
これはテストです。
次の文です。
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

## 次のステップ

- [CLI リファレンス](../cli-reference/overview.md) -- すべての CLI コマンドとオプションの詳細
- [学習ガイド](../training-guide/preparing-corpus.md) -- 独自モデルの学習方法
- [アーキテクチャ](../architecture/overview.md) -- Litsea の内部動作の理解
