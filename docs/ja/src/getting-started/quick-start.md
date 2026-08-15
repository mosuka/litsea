# クイックスタート

## CLI クイックスタート

### テキストの分割

Litsea には `models/` ディレクトリに学習済みモデルが同梱されています。テキストを `segment` コマンドにパイプで渡します:

**日本語**（同梱の `RWCP.model`、オリジナルの TinySegmenter モデルを使用）:

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" \
  | litsea segment -l japanese ./models/RWCP.model
```

出力:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
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

Litsea は POS モデルを使うことで、単語分割と品詞推定を同時に行うことができます。`segment` コマンドに `--pos` フラグを追加します:

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./models/japanese_pos.model
```

出力:

```text
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

各トークンには [Universal POS（UPOS）](https://universaldependencies.org/u/pos/) タグが付与されます。

### 二段構成の品詞推定付き分割

Litsea には、より高速な[二段構成](../algorithm/two-stage-tagging.md)の品詞推定アーキテクチャも同梱されています。CLI はファイルからモデルの種類を自動判定するため、コマンドはモデルファイル名以外は上の joint の例と同じです:

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./models/japanese_two_stage.model
```

出力の形式は joint の例と同じです。新規に使う場合は[事前学習済みモデル](../pre-trained-models.md#モデルの選択)で推奨されている二段構成モデルの利用を検討してください。

## ライブラリ クイックスタート

モデルを読み込みテキストを分割する最小限の Rust プログラムです:

```rust
use std::path::Path;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;

fn main() -> litsea::Result<()> {
    // Load the pre-trained model
    let mut learner = AdaBoost::new(0.01, 100);
    learner.load_model_from_path(Path::new("./models/RWCP.model"))?;

    // Create a segmenter
    let segmenter = Segmenter::with_learner(Language::Japanese, learner);

    // Segment text
    let tokens = segmenter.segment("これはテストです。");
    println!("{}", tokens.join(" "));
    // Output: これ は テスト です 。

    Ok(())
}
```

### ライブラリでの品詞推定付き分割

品詞推定付きモデルを読み込み、単語分割と品詞推定を同時に行う最小限の Rust プログラムです:

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;

fn main() -> litsea::Result<()> {
    // Load the pre-trained POS model
    let mut pos_learner = AveragedPerceptron::new();
    pos_learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

    // Create a segmenter with POS support
    let segmenter = Segmenter::with_pos_learner(Language::Japanese, pos_learner);

    // Segment text with POS tags
    let tokens = segmenter.segment_with_pos("今日はいい天気ですね。")?;
    for (word, pos) in &tokens {
        print!("{}/{} ", word, pos);
    }
    // Output: 今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT

    Ok(())
}
```

## 次のステップ

- [CLI リファレンス](../litsea-cli.md) -- すべての CLI コマンドとオプションの詳細
- [学習ガイド](../training-guide/preparing-corpus.md) -- 独自モデルの学習方法
- [アーキテクチャ](../architecture/overview.md) -- Litsea の内部動作の理解
