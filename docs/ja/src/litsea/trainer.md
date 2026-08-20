# Trainer

`Trainer` 構造体は、モデル学習パイプライン全体を制御します。

## 定義

```rust
pub struct Trainer {
    learner: AdaBoost,
}
```

## コンストラクタ

### `Trainer::new`

```rust
pub fn new(
    threshold: f64,
    num_iterations: usize,
    features_path: &Path,
) -> litsea::Result<Self>
```

Trainer を作成し、特徴量ファイルから初期化します。内部で `AdaBoost::initialize_features()` と `AdaBoost::initialize_instances()` を呼び出します。

```rust
use std::path::Path;
use litsea::trainer::Trainer;

let mut trainer = Trainer::new(
    0.0001,                          // threshold
    20000,                           // max iterations
    Path::new("./features.txt"),     // features file
)?;
```

## メソッド

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> litsea::Result<()>
```

再学習用に既存のモデルを読み込みます。ファイルパス、`file://`、および（`remote_model` フィーチャー有効時）`http://`、`https://` URI に対応しています。

`Trainer::new` の後に呼び出すと、読み込んだ重みは特徴名をキーとして、初期化済みの学習データにマージされます。そのため、特徴量インデックスを壊すことなく、既存モデルから増分学習を開始できます。

```rust
trainer.load_model("./models/japanese.model").await?;
```

### `train`

```rust
pub fn train(
    &mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<BinaryMetrics>
```

モデルを学習し、指定したパスに保存します。評価メトリクスを返します。

`running` フラグにより、学習の途中停止が可能です。`false` に設定すると学習を早期終了します。

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

let running = AtomicBool::new(true);
let metrics = trainer.train(&running, Path::new("./model.model"))?;

println!("Accuracy: {:.2}%", metrics.accuracy);
```

## 学習の完全な例

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::Trainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let mut trainer = Trainer::new(
        0.0001,
        20000,
        Path::new("./features.txt"),
    )?;

    // Optionally resume from an existing model
    // trainer.load_model("./models/japanese.model").await?;

    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./model.model"))?;

    println!("Accuracy:  {:.2}%", metrics.accuracy);
    println!("Precision: {:.2}%", metrics.precision);
    println!("Recall:    {:.2}%", metrics.recall);

    Ok(())
}
```

## PerceptronTrainer

`PerceptronTrainer` は `Trainer` の汎用 Averaged Perceptron 版です。特徴量ファイルから、不透明な文字列ラベルに対する多クラスの **Averaged Perceptron** を学習します（`litsea train --perceptron`）。主な用途は、畳み込みレシピ（[事前学習済みモデル](../pre-trained-models.md#学習手順)を参照）が同梱の AdaBoost 形式分割モデルへ変換する、2 クラス（`B`/`O`）境界パーセプトロンの学習です。

### `PerceptronTrainer::new`

```rust
pub fn new(num_epochs: usize, features_path: &Path) -> litsea::Result<Self>
```

特徴量ファイル（各行が `label\tfeature1\tfeature2\t...` の形式で、ラベルは不透明な文字列。例: 境界ラベル `B`/`O`）を読み込み、学習インスタンスを登録します。

### `PerceptronTrainer::load_model`

```rust
pub async fn load_model(&mut self, model_uri: &str) -> litsea::Result<()>
```

既存のパーセプトロンモデルを読み込み、増分学習を行います。学習データからすでに登録済みのクラスは、モデルのクラスとマージされます。

### `PerceptronTrainer::train`

```rust
pub fn train(
    &mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<MulticlassMetrics>
```

設定されたエポック数だけ学習を行い、モデルを保存して、多クラス評価メトリクス（正解率、マクロ平均適合率、マクロ平均再現率）を返します。`running` フラグは `Trainer::train` と同様に、学習の途中停止を可能にします。

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::PerceptronTrainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let mut trainer = PerceptronTrainer::new(10, Path::new("./features.txt"))?;
    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./perceptron.model"))?;
    println!("Accuracy: {:.2}%", metrics.accuracy);
    Ok(())
}
```

## TwoStageTrainer

`TwoStageTrainer` は[二段構成モデル](../algorithm/two-stage-tagging.md)
（issue #147）を学習します: 二値の境界分類器（stage 1）と単語単位の
マルチクラスタガー（stage 2）を、いずれも **Averaged Perceptron** として
学習し、候補タグ語彙表とともに単一の `litsea-two-stage v1` ファイルに
組み立てます。学習後、stage 1 は既存の AdaBoost 形式のスカラー特徴量重みに
畳み込まれます（品質を損なわない変換 -- 導出はこのモジュールのソース
ドキュメントを参照）。これによりランタイムは通常の `segment()` モデルと
全く同じ方法で採点します。`TwoStageTrainer` と `TwoStageMetrics` はどちらも
クレートのルートから `litsea::TwoStageTrainer` / `litsea::TwoStageMetrics`
として再エクスポートされています。

### `TwoStageTrainer::new`

```rust
pub fn new(
    num_epochs: usize,
    dominance: f64,
    features_prefix: &Path,
) -> litsea::Result<Self>
```

[`Extractor::extract_two_stage`](extractor.md) が書き出す 3 つのファイルを
`features_prefix` から読み込みます（`{prefix}.stage1`、`{prefix}.stage2`、
`{prefix}.lexicon`）。両ステージ分の学習インスタンスを登録します。

`dominance` は、組み立て後のモデルにおける分類器スキップの閾値です:
既知の単語のうち最頻タグが学習時の出現のこの割合以上を占めるものは、
stage-2 分類器を呼ばずにタグ付けされます。値は `(0.5, 1.0]` の範囲内で
なければならず、`new()` の時点で即座に検証されます。そのため範囲外の値は
学習が始まる前に失敗し、学習後に失敗することはありません。

```rust
use std::path::Path;
use litsea::trainer::TwoStageTrainer;

let trainer = TwoStageTrainer::new(
    50,                            // num_epochs（両ステージ共通）
    0.99,                          // dominance
    Path::new("./features"),       // 特徴量プレフィックス
)?;
```

### `TwoStageTrainer::train`

```rust
pub fn train(
    mut self,
    running: &AtomicBool,
    model_path: &Path,
) -> litsea::Result<TwoStageMetrics>
```

`Trainer::train` や `PerceptronTrainer::train` と異なり、このメソッドは `self` を
値として受け取ります（Trainer を消費します）。両ステージを Averaged
Perceptron として `num_epochs` エポック分学習し、stage 1 を AdaBoost の
重みへ畳み込み、語彙表とともに 2 つのステージを `litsea-two-stage v1`
モデルへ組み立てて `model_path` に保存し、両ステージの in-sample
メトリクスを返します。`running` フラグは他の Trainer と同様に、学習の
途中停止を可能にします。

```rust
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::TwoStageTrainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let trainer = TwoStageTrainer::new(50, 0.99, Path::new("./features"))?;
    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./model.model"))?;

    println!("Stage 1: {:.2}%, Stage 2: {:.2}%", metrics.stage1.accuracy, metrics.stage2.accuracy);

    Ok(())
}
```

### `TwoStageMetrics`

```rust
pub struct TwoStageMetrics {
    pub stage1: MulticlassMetrics,
    pub stage2: MulticlassMetrics,
}
```

`TwoStageTrainer::train` の実行結果である in-sample メトリクスです。
`stage1` は境界分類器の 2 クラス（`B`/`O`）に対するメトリクス、`stage2` は
単語単位のタガーの UPOS タグクラスに対するメトリクスです。どちらのフィールドも
`MulticlassMetrics` 型で、[`PerceptronTrainer::train`](#perceptrontrainer)（上記）が返すものと
同じ型であり、正解率とマクロ平均の適合率・再現率を保持します。
