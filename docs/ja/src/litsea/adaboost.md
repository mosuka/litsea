# AdaBoost

`AdaBoost` 構造体は、単語境界検出のための二値分類を実装しています。

## 定義

```rust
pub struct AdaBoost {
    pub threshold: f64,
    pub num_iterations: usize,
    // internal fields: model weights, features, instances, etc.
}
```

## コンストラクタ

### `AdaBoost::new`

```rust
pub fn new(threshold: f64, num_iterations: usize) -> Self
```

指定したハイパーパラメータで新しい AdaBoost インスタンスを作成します。

```rust
use litsea::adaboost::AdaBoost;

let mut learner = AdaBoost::new(0.01, 100);
```

## モデルの読み込み

### `load_model`

```rust
pub async fn load_model(&mut self, uri: &str) -> io::Result<()>
```

URI からモデルの重みを読み込みます。以下の形式に対応しています:

- ローカルファイルパス: `./resources/japanese.model`
- File URI: `file:///path/to/model`
- HTTP: `http://example.com/model`
- HTTPS: `https://example.com/model`

```rust
learner.load_model("./resources/japanese.model").await?;
learner.load_model("https://example.com/model").await?;
```

### `save_model`

```rust
pub fn save_model(&self, filename: &Path) -> io::Result<()>
```

モデルの重みをファイルに保存します。モデルが空の場合はエラーを返します。

## 学習メソッド

### `initialize_features`

```rust
pub fn initialize_features(&mut self, filename: &Path) -> io::Result<()>
```

特徴量ファイルを読み込み、特徴量インデックスを構築します。`initialize_instances` の前に呼び出す必要があります。

### `initialize_instances`

```rust
pub fn initialize_instances(&mut self, filename: &Path) -> io::Result<()>
```

同じ特徴量ファイルを読み込み、ラベル付きインスタンスとその重みを初期化します。

### `train`

```rust
pub fn train(&mut self, running: Arc<AtomicBool>)
```

AdaBoost の学習ループを実行します。`running` を `false` に設定すると早期終了します。

### `add_instance`

```rust
pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8)
```

特徴量セットとラベルを持つ単一の学習インスタンスを追加します。

## 予測

### `predict`

```rust
pub fn predict(&self, attributes: HashSet<String>) -> i8
```

与えられた特徴量セットに対してラベルを予測します。`+1`（境界）または `-1`（非境界）を返します。

```rust
use std::collections::HashSet;

let mut attrs = HashSet::new();
attrs.insert("UW4:は".to_string());
attrs.insert("UC4:I".to_string());
// ... その他の特徴量

let label = learner.predict(attrs);
// label == 1 (境界) or -1 (非境界)
```

### `get_bias`

```rust
pub fn get_bias(&self) -> f64
```

バイアス項を返します: `-sum(all model weights) / 2.0`

## 評価

### `get_metrics`

```rust
pub fn get_metrics(&self) -> Metrics
```

学習データに対する評価メトリクスを算出します。

## Metrics

```rust
pub struct Metrics {
    pub accuracy: f64,          // 正解率（パーセント）
    pub precision: f64,         // 適合率（パーセント）
    pub recall: f64,            // 再現率（パーセント）
    pub num_instances: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
}
```
