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
| `--num-epochs <NUM_EPOCHS>` | `10` | 学習エポック数（POS モードおよび `--two-stage` モード） |
| `--two-stage` | off | 代わりに[二段構成](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)モデルを学習する。`{FEATURES_FILE}.stage1`/`.stage2`/`.lexicon`（`extract --two-stage` の出力）を読み込む。`--pos` および `-m`/`--load-model-uri`（増分学習は非対応）とは併用できない |
| `--dominance <DOMINANCE>` | `0.99` | `--two-stage` 用の分類器スキップ閾値、範囲は `(0.5, 1.0]`。既知の単語のうち最頻タグが学習時の出現のこの割合以上を占めるものは、stage-2 分類器を呼ばずにタグ付けされる |

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
| `--num-epochs <NUM_EPOCHS>` | `10` | 学習エポック数 |

### 使用例

```sh
# 品詞特徴量から品詞モデルを学習
litsea train --pos --num-epochs 10 ./pos_features.txt ./models/japanese_pos.model
```

### 出力

学習メトリクスはstderrに出力されます（マクロ平均の適合率・再現率）。

```text
Result Metrics (POS):
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

## 二段構成モデルの学習

`--two-stage` を指定すると、
[二段構成モデル](../advanced/model-file-format.md#two-stage-model-format-litsea-two-stage-v1)
を構築します: 二値の境界分類器（stage 1）と単語単位の品詞タガー（stage 2）を、
候補タグ語彙表とともに単一の `litsea-two-stage v1` ファイルに組み立てます。
両ステージとも `--num-epochs` エポック分 Averaged Perceptron として学習し、
その後 stage 1 は既存の AdaBoost 形式のスカラー重みに畳み込まれます
（品質を損なわない変換 — 導出は `litsea::trainer` のモジュールドキュメントを参照）。
これによりランタイムは通常の `segment()` モデルと全く同じ方法で採点します。

### 使い方

```sh
litsea train --two-stage [OPTIONS] <FEATURES_PREFIX> <MODEL_FILE>
```

`FEATURES_PREFIX` は `extract --two-stage` に渡したものと同じプレフィックスです。

### 例

```sh
litsea extract --two-stage -l japanese ./pos_corpus.txt ./two_stage_features
litsea train --two-stage --num-epochs 10 ./two_stage_features ./models/japanese_two_stage.model
```

### 出力

```text
Result Metrics (Two-Stage):
  Stage 1 (boundary) Accuracy: 99.36% ( 277213 )
  Stage 1 Macro Precision: 99.30%
  Stage 1 Macro Recall: 99.35%
  Stage 2 (tagging) Accuracy: 98.53% ( 168333 )
  Stage 2 Macro Precision: 98.39%
  Stage 2 Macro Recall: 97.59%
```

他のモードと同様、これらは in-sample のメトリクスです。現実的な品質を見積もるには
`litsea evaluate --pos` でホールドアウトされたテキストを評価してください。
`segment --pos` と `evaluate --pos` はファイルヘッダから二段構成モデルを自動判別するため、
利用に追加のフラグは不要です。
