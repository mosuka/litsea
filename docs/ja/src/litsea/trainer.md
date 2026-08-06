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
    0.005,                           // threshold
    1000,                            // max iterations
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
        0.005,
        1000,
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

## PosTrainer

`PosTrainer` は `Trainer` の POS モデル版です。POS 特徴量ファイル（`litsea extract --pos` で生成）から、単語分割と品詞タグ付けを同時に行うための **Averaged Perceptron** を学習します。

### `PosTrainer::new`

```rust
pub fn new(num_epochs: usize, features_path: &Path) -> litsea::Result<Self>
```

特徴量ファイル（各行が `label\tfeature1\tfeature2\t...` の形式で、ラベルは `B-NOUN` や `O` のような `SegmentLabel` 文字列）を読み込み、学習インスタンスを登録します。

### `PosTrainer::load_model`

```rust
pub async fn load_model(&mut self, model_uri: &str) -> litsea::Result<()>
```

既存の POS モデルを読み込み、増分学習を行います。学習データからすでに登録済みのクラスは、モデルのクラスとマージされます。

### `PosTrainer::train`

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

use litsea::trainer::PosTrainer;

#[tokio::main]
async fn main() -> litsea::Result<()> {
    let mut trainer = PosTrainer::new(10, Path::new("./features_pos.txt"))?;
    let running = AtomicBool::new(true);
    let metrics = trainer.train(&running, Path::new("./japanese_pos.model"))?;
    println!("Accuracy: {:.2}%", metrics.accuracy);
    Ok(())
}
```
