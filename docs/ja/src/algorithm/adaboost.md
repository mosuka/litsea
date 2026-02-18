# AdaBoost 二値分類

Litsea は、単語境界を判定するために **AdaBoost**（Adaptive Boosting）アルゴリズムによる二値分類を使用します。本章では、Litsea に実装されているアルゴリズムについて説明します。

## 概要

AdaBoost は、多数の**弱学習器**（単純な分類器）を組み合わせて強力なアンサンブル分類器を構築します。Litsea では:

- **正ラベル（+1）** = 単語境界
- **負ラベル（-1）** = 非境界（現在の単語の継続）
- **弱学習器** = 個々の特徴量（各特徴量は二値の「切り株」-- 存在するか否か）

## 学習アルゴリズム

`AdaBoost::train()` の学習ループは以下のように動作します:

### 初期化

1. 学習ファイルから特徴量とインスタンスを読み込み
2. インスタンスの重みを均一に初期化（後に初期スコアに基づいて調整）
3. すべてのモデルの重みをゼロで初期化

### 反復ブースティング

各イテレーション *t*（最大 `num_iterations` 回）について:

**ステップ 1: 重み付き誤差の計算**

各特徴量 *h* について、全インスタンスに対する重み付き誤差を計算します:

```text
error[h] -= D[i] * y[i]   (for each instance i that has feature h)
```

ここで *D[i]* はインスタンスの重み、*y[i]* は真のラベルです。

**ステップ 2: 最良の弱学習器の選択**

重み付き誤差率が最も低い特徴量を選択します:

```text
error_rate(h) = (error[h] + positive_weight_sum) / instance_weight_sum
h_best = argmax_h |0.5 - error_rate(h)|
```

基準となる競合対象は「全て負」分類器（常に -1 を予測）であり、その誤差率は正のインスタンスの割合に等しくなります。実際の特徴量はこの基準を上回る必要があります。

**ステップ 3: 収束判定**

`|0.5 - best_error_rate| < threshold` の場合、早期停止します -- どの特徴量もモデルを大幅に改善できないためです。

**ステップ 4: 弱学習器の重みの計算**

```text
alpha = 0.5 * ln((1 - error_rate) / error_rate)
model[h_best] += alpha
```

誤差率が低いほど alpha が高くなり、より良い特徴量により大きな影響力を与えます。

**ステップ 5: インスタンスの重みの更新**

```text
For each instance i:
    prediction = +1 if h_best in features(i), else -1

    if y[i] * prediction < 0:  (misclassified)
        D[i] *= exp(alpha)     (increase weight)
    else:                       (correctly classified)
        D[i] /= exp(alpha)     (decrease weight)

Normalize: D[i] /= sum(D)
```

これにより、後続のイテレーションは分類が困難なインスタンスに集中するようになります。

## 予測

入力された特徴量（属性）のセットに対して、予測は以下のように行われます:

```text
score = bias + sum(model[feature] for each feature in attributes)
prediction = +1 if score >= 0, else -1
```

### バイアス項

バイアスは以下のように計算されます:

```text
bias = -sum(all model weights) / 2.0
```

これにより決定境界が中心化されます。空文字列特徴量（`""`）は学習中のバイアスバケットとして機能します。

## モデルファイル形式

学習されたモデルはシンプルなテキストファイルとして保存されます:

```text
feature1\tweight1
feature2\tweight2
...
bias_value
```

- 各行には特徴量名とその重み（タブ区切り）が含まれる
- 重みがゼロの特徴量は省略される
- 最終行にはバイアス項（単一の数値）が含まれる

詳細は[モデルファイル形式](../advanced/model-file-format.md)を参照してください。

## ハイパーパラメータ

| Parameter | Default | 説明 |
|-----------|---------|------|
| `threshold` | 0.01 | 早期停止の閾値。低い値はより多くのイテレーションを許可し、精度が向上する可能性がある |
| `num_iterations` | 100 | ブースティングの最大ラウンド数。高い値は学習時間とモデルサイズを犠牲にして精度を向上させる可能性がある |
