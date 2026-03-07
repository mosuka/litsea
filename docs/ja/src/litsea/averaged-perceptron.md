# Averaged Perceptron

`AveragedPerceptron` 構造体は、品詞推定のための多クラス分類を実装しています。

## 定義

```rust
pub struct AveragedPerceptron {
    // internal fields: weights, accumulated, timestamps, step, classes, instances
}
```

## コンストラクタ

### `AveragedPerceptron::new`

```rust
pub fn new() -> Self
```

新しい Averaged Perceptron インスタンスを作成します。

```rust
use litsea::perceptron::AveragedPerceptron;

let mut learner = AveragedPerceptron::new();
```

## インスタンスの追加

### `add_instance`

```rust
pub fn add_instance(&mut self, features: HashSet<String>, label: String)
```

特徴量セットとラベルを持つ学習インスタンスを追加します。未知のクラスは自動的に登録されます。

```rust
use std::collections::HashSet;
use litsea::perceptron::AveragedPerceptron;

let mut learner = AveragedPerceptron::new();
let mut feats = HashSet::new();
feats.insert("UW4:猫".to_string());
feats.insert("UC4:H".to_string());
learner.add_instance(feats, "B-NOUN".to_string());
```

## 学習

### `train`

```rust
pub fn train(&mut self, num_epochs: usize, running: Arc<AtomicBool>)
```

指定されたエポック数でモデルを学習します。`running` を `false` に設定すると早期終了します。学習終了時に重みの平均化が自動的に行われます。

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

let running = Arc::new(AtomicBool::new(true));
learner.train(10, running);
```

## 予測

### `predict`

```rust
pub fn predict(&self, features: &HashSet<String>) -> String
```

与えられた特徴量セットに対してラベルを予測します。各クラスのスコアを計算し、最大スコアのクラス名を返します。クラスが未登録の場合は空文字列を返します。

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:猫".to_string());
attrs.insert("UC4:H".to_string());

let label = learner.predict(&attrs);
// label == "B-NOUN" など
```

## モデルの入出力

### `save_model`

```rust
pub fn save_model(&self, path: &Path) -> io::Result<()>
```

モデルをファイルに保存します。モデルが空の場合はエラーを返します。

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

URI からモデルを読み込みます。以下の形式に対応しています:

- ローカルファイルパス: `./resources/japanese_pos.model`
- File URI: `file:///path/to/model`
- HTTP: `http://example.com/model`
- HTTPS: `https://example.com/model`

```rust
learner.load_model("./resources/japanese_pos.model").await?;
```

## 評価

### `get_metrics`

```rust
pub fn get_metrics(&self) -> Metrics
```

学習データに対する評価メトリクスを算出します。

## Metrics

```rust
pub struct Metrics {
    pub accuracy: f64,                            // 正解率（パーセント）
    pub macro_precision: f64,                     // マクロ平均適合率（パーセント）
    pub macro_recall: f64,                        // マクロ平均再現率（パーセント）
    pub num_instances: usize,                     // インスタンス数
    pub correct_per_class: HashMap<String, usize>,   // クラスごとの正解数
    pub predicted_per_class: HashMap<String, usize>,  // クラスごとの予測数
    pub gold_per_class: HashMap<String, usize>,       // クラスごとの正解ラベル数
}
```
