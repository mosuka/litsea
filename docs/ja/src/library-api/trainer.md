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
) -> io::Result<Self>
```

Trainer を作成し、特徴量ファイルから初期化します。内部で `AdaBoost::initialize_features()` と `AdaBoost::initialize_instances()` を呼び出します。

```rust
use std::path::Path;
use litsea::trainer::Trainer;

let mut trainer = Trainer::new(
    0.005,                           // 閾値
    1000,                            // 最大反復回数
    Path::new("./features.txt"),     // 特徴量ファイル
)?;
```

## メソッド

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

再学習用に既存のモデルを読み込みます。ファイルパス、`file://`、`http://`、`https://` URI に対応しています。

```rust
trainer.load_model("./resources/japanese.model").await?;
```

### `train`

```rust
pub fn train(
    &mut self,
    running: Arc<AtomicBool>,
    model_path: &Path,
) -> Result<Metrics, Box<dyn std::error::Error>>
```

モデルを学習し、指定したパスに保存します。評価メトリクスを返します。

`running` フラグにより、学習の途中停止が可能です。`false` に設定すると学習を早期終了します。

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::path::Path;

let running = Arc::new(AtomicBool::new(true));
let metrics = trainer.train(running, Path::new("./model.model"))?;

println!("Accuracy: {:.2}%", metrics.accuracy);
```

## 学習の完全な例

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::path::Path;

use litsea::trainer::Trainer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = Trainer::new(
        0.005,
        1000,
        Path::new("./features.txt"),
    )?;

    // 必要に応じて既存モデルから再開
    // trainer.load_model("./resources/japanese.model").await?;

    let running = Arc::new(AtomicBool::new(true));
    let metrics = trainer.train(running, Path::new("./model.model"))?;

    println!("Accuracy:  {:.2}%", metrics.accuracy);
    println!("Precision: {:.2}%", metrics.precision);
    println!("Recall:    {:.2}%", metrics.recall);

    Ok(())
}
```
