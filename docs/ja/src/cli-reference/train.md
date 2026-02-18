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

## 出力

学習メトリクスはstderrに出力されます。

```text
Result Metrics:
  Accuracy: 94.15% ( 564133 / 599198 )
  Precision: 95.57% ( 330454 / 345758 )
  Recall: 94.36% ( 330454 / 350215 )
  Confusion Matrix:
    True Positives: 330454
    False Positives: 15304
    False Negatives: 19761
    True Negatives: 233679
```

## Ctrl+C のハンドリング

学習は優雅な中断をサポートしています。

- **1回目のCtrl+C**: 学習を停止し、現在の状態でモデルを保存する
- **2回目のCtrl+C**: 保存せずに即座に終了する

これにより、長時間の学習セッションを進捗を失うことなく停止できます。

## 使用例

基本的な学習:

```sh
litsea train -t 0.005 -i 1000 ./features.txt ./resources/japanese.model
```

高精度な学習（低い閾値、多い反復回数）:

```sh
litsea train -t 0.001 -i 5000 ./features.txt ./model.model
```

既存モデルからの再学習:

```sh
litsea train -t 0.005 -i 1000 -m ./resources/japanese.model \
    ./new_features.txt ./resources/japanese_v2.model
```

## ハイパーパラメータの調整

| Parameter | 値を小さくした場合の効果 | 値を大きくした場合の効果 |
|-----------|---------------------|---------------------|
| `threshold` | 反復回数が増加、精度が向上する可能性あり、学習時間が長くなる | 反復回数が減少、学習が高速化、アンダーフィットの可能性あり |
| `num_iterations` | ブースティングラウンドが減少、モデルが小さくなる、アンダーフィットの可能性あり | ラウンドが増加、モデルが大きくなる、精度が向上する可能性あり |
