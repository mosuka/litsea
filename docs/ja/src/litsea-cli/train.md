# train

AdaBoostを使用して単語分割モデルを学習します。

## 使い方

```sh
litsea train [OPTIONS] <FEATURES_FILE> <MODEL_FILE>
```

## 引数

| Argument | Description |
|----------|------------|
| `FEATURES_FILE` | 入力特徴量ファイルのパス（`extract` の出力） |
| `MODEL_FILE` | 出力モデルファイルのパス |

## オプション

| Option | Default | Description |
|--------|---------|------------|
| `-t`, `--threshold <THRESHOLD>` | `0.01` | 早期停止のための弱分類器精度の閾値。値を小さくするとより多くの反復が可能になる |
| `-i`, `--num-iterations <NUM_ITERATIONS>` | `100` | ブースティング反復の最大回数 |
| `-m`, `--load-model-uri <LOAD_MODEL_URI>` | None | 学習を再開するための既存モデルのURI（ファイルパスまたはHTTP/HTTPS URL） |
| `--pos` | off | 品詞（POS）学習モードを有効にする（Averaged Perceptron を使用） |
| `-e`, `--num-epochs <NUM_EPOCHS>` | `10` | 学習エポック数（POS モードのみ） |

## 出力

学習メトリクスはstderrに出力されます。

メトリクスは学習データに対して計算されます。反復回数が十分であれば、モデルは学習コーパスにほぼ完全に適合できてしまうため、現実的な品質を見積もるにはホールドアウトされたテキストで評価してください。

```text
Result Metrics:
  Accuracy: 100.00% ( 1075868 / 1075869 )
  Precision: 100.00% ( 161283 / 161284 )
  Recall: 100.00% ( 161283 / 161283 )
  Confusion Matrix:
    True Positives: 161283
    False Positives: 1
    False Negatives: 0
    True Negatives: 914585
```

## Ctrl+C のハンドリング

学習は優雅な中断をサポートしています。

- **1回目のCtrl+C**: 学習を停止し、現在の状態でモデルを保存する
- **2回目のCtrl+C**: 保存せずに即座に終了する

これにより、長時間の学習セッションを進捗を失うことなく停止できます。

## 使用例

基本的な学習:

```sh
litsea train -t 0.0001 -i 20000 ./features.txt ./models/japanese.model
```

高精度な学習（低い閾値、多い反復回数）:

```sh
litsea train -t 0.001 -i 5000 ./features.txt ./model.model
```

既存モデルからの再学習:

```sh
litsea train -t 0.0001 -i 20000 -m ./models/japanese.model \
    ./new_features.txt ./models/japanese_v2.model
```

## ハイパーパラメータの調整

| Parameter | 値を小さくした場合の効果 | 値を大きくした場合の効果 |
|-----------|---------------------|---------------------|
| `threshold` | 反復回数が増加、精度が向上する可能性あり、学習時間が長くなる | 反復回数が減少、学習が高速化、アンダーフィットの可能性あり |
| `num_iterations` | ブースティングラウンドが減少、モデルが小さくなる、アンダーフィットの可能性あり | ラウンドが増加、モデルが大きくなる、精度が向上する可能性あり |

## 品詞モデルの学習（`--pos`）

`--pos` フラグを指定すると、AdaBoost の代わりに **Averaged Perceptron** アルゴリズムを使用します。単語分割と品詞タグ付けを同時に行うマルチクラス分類器を学習します。

### 使い方

```sh
litsea train --pos [OPTIONS] <FEATURES_FILE> <MODEL_FILE>
```

### POS 学習固有のオプション

| Option | Default | Description |
|--------|---------|------------|
| `--pos` | off | 品詞推定モデル（Averaged Perceptron）を学習する |
| `-e`, `--num-epochs <NUM_EPOCHS>` | `10` | 学習エポック数 |

### 使用例

```sh
# 品詞モデルの学習（10エポック）
litsea train --pos -e 10 ./pos_features.txt ./models/japanese_pos.model
```

### 出力

学習メトリクスはstderrに出力されます（マクロ平均の適合率・再現率）。

```text
Result Metrics:
  Accuracy: 98.23% ( 277213 )
  Macro Precision: 96.82%
  Macro Recall: 93.30%
```

### Ctrl+C のハンドリング

AdaBoost と同様に、品詞モデルの学習も優雅な中断をサポートしています。1回目の Ctrl+C で学習を停止し、現在の状態でモデルを保存します。

### POS ハイパーパラメータ

| Parameter | 値を小さくした場合の効果 | 値を大きくした場合の効果 |
|-----------|---------------------|---------------------|
| `num_epochs` | 学習が高速化、アンダーフィットの可能性あり | 精度が向上、学習時間が長くなる、オーバーフィットの可能性あり |
