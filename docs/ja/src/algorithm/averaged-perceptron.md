# Averaged Perceptron

Litsea は、単語分割と品詞推定を同時に行うために **Averaged Perceptron** アルゴリズムによる多クラス分類を使用します。本章では、Litsea に実装されているアルゴリズムについて説明します。

## 概要

Averaged Perceptron は、**多クラス分類**を行う線形分類器です。AdaBoost が二値分類（境界 / 非境界）を行うのに対し、Averaged Perceptron は 18 クラスの **SegmentLabel** を直接予測します:

- `B-ADJ`, `B-ADP`, `B-ADV`, `B-AUX`, `B-CCONJ`, `B-DET`, `B-INTJ`, `B-NOUN`, `B-NUM`, `B-PART`, `B-PRON`, `B-PROPN`, `B-PUNCT`, `B-SCONJ`, `B-SYM`, `B-VERB`, `B-X`（17 品詞の境界ラベル）
- `O`（非境界 = 単語の継続）

これにより、単語境界の検出と品詞の推定を 1 つの分類ステップで同時に行えます。

### AdaBoost との比較

| 項目 | AdaBoost | Averaged Perceptron |
|------|----------|---------------------|
| 分類方式 | 二値分類（+1 / -1） | 多クラス分類（18クラス） |
| 出力 | 境界 / 非境界 | 境界+品詞 / 非境界 |
| 弱学習器 | 特徴量の決定株 | なし（線形分類器） |
| 重みの管理 | 特徴量ごとに1つの重み | クラス×特徴量の重み行列 |
| 汎化手法 | アンサンブル | 重みの平均化 |

## 重みベクトル

各クラスは独立した重みベクトルを持ちます。特徴量は疎（sparse）な二値表現で、存在する特徴量の重みのみを合算してスコアを計算します。

```text
weights["B-NOUN"]["UW4:猫"] = 2.5
weights["B-NOUN"]["UC4:H"]  = 1.8
weights["O"]["UW4:猫"]      = -0.3
...
```

予測時には、すべてのクラスについてスコアを計算し、最大スコアのクラスを予測ラベルとします:

```text
score(class) = sum(weights[class][feature] for feature in input_features)
prediction = argmax_class score(class)
```

## 学習アルゴリズム

### 更新規則

各学習インスタンスについて、予測が正解と異なる場合に重みを更新します:

```text
For each training instance (features, truth):
    guess = predict(features)

    if guess != truth:
        For each feature f in features:
            weights[truth][f] += 1.0   # 正解クラスの重みを増加
            weights[guess][f] -= 1.0   # 誤予測クラスの重みを減少
```

この単純な更新規則により、正解クラスの特徴量が強化され、誤予測クラスの特徴量が弱められます。

### 平均化による汎化

学習の各ステップで重みが変動するため、最終的な重みは学習データの末尾に過適合する傾向があります。これを防ぐために、全ステップにわたる**累積平均重み**を最終モデルとして使用します。

```text
# 各ステップの重みを累積
accumulated[class][feature] += weight[class][feature] * elapsed_steps

# 学習終了時に平均化
final_weight[class][feature] = accumulated[class][feature] / total_steps
```

この平均化により、学習データの順序への依存が軽減され、汎化性能が向上します。

### エポック数と早期停止

学習は指定されたエポック数（`num_epochs`）だけ学習データを繰り返します。`AtomicBool` フラグにより、Ctrl+C などで学習を中断し、その時点でのモデルを保存することも可能です。

```rust
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use litsea::perceptron::AveragedPerceptron;

let mut perceptron = AveragedPerceptron::new();
// ... インスタンスを追加 ...
let running = Arc::new(AtomicBool::new(true));
perceptron.train(10, running);  // 10エポック
```

## モデルファイル形式

学習されたモデルはテキストファイルとして保存されます:

```text
18
B-ADJ
B-ADP
B-ADV
...
O
UW4:猫	B-NOUN	2.5
UC4:H	B-NOUN	1.8
UW4:猫	O	-0.3
...
```

- 1行目: クラス数
- 続くN行: クラス名（1行に1つ）
- 残りの行: `特徴名\tクラス名\t重み`（タブ区切り、重みがゼロの特徴量は省略）

## ハイパーパラメータ

| Parameter | Default | 説明 |
|-----------|---------|------|
| `num_epochs` | 10 | 学習エポック数。高い値は精度を向上させる可能性があるが、学習時間が増加する |
