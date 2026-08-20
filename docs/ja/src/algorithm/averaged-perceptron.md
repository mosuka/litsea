# Averaged Perceptron

Litsea は、二段構成品詞推定アーキテクチャの両ステージと、同梱分割モデルの畳み込み（collapse）レシピの学習側の学習器として、**Averaged Perceptron** アルゴリズムによる多クラス分類を使用します。本章では、Litsea に実装されているアルゴリズムについて説明します。

## 概要

[AdaBoost](adaboost.md) が**二値分類**（境界 / 非境界）を行うのに対し、Averaged Perceptron は 18 クラスの**多クラス分類**を行い、各文字位置に対してセグメントラベルを予測します:

- **17 個の境界ラベル**: `B-ADJ`, `B-ADP`, `B-ADV`, `B-AUX`, `B-CCONJ`, `B-DET`, `B-INTJ`, `B-NOUN`, `B-NUM`, `B-PART`, `B-PRON`, `B-PROPN`, `B-PUNCT`, `B-SCONJ`, `B-SYM`, `B-VERB`, `B-X`
- **1 個の非境界ラベル**: `O`（単語の継続）

これらのラベルは Universal Dependencies プロジェクトの 17 個の [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) タグに対応し、`B-` プレフィックスで単語境界を示します。これにより、単語境界の検出と品詞の推定を 1 つの分類ステップで同時に行えます。

Averaged Perceptron を 2 クラスモード（`B`/`O` ラベルのみ）で使うと、実際には同梱の `japanese.model`・`chinese.model`・`korean.model`・`english.model` の各分割モデルを学習しています。学習後、推論用に AdaBoost モデル形式へ無損失で畳み込まれます — 完全な導出は[事前学習済みモデル: 学習手順](../pre-trained-models.md#学習手順)を参照してください。同じ畳み込みが[二段構成アーキテクチャ](two-stage-tagging.md)の stage-1 境界分類器の学習にも使われ、多クラス形式は stage-2 単語タガーのアルゴリズムです。

## アルゴリズム

### 重み表現

パーセプトロンは**特徴量ごとにクラス別の重みベクトル**を保持し、単一の疎なマップに格納します。これにより、スコア計算と重みの更新は
特徴量ごとに 1 回のハッシュ検索で済みます（特徴量 × クラスごとに 1 回ではありません）:

```text
slots: FxHashMap<Feature, FeatureSlot>
// FeatureSlot { w: Vec<f64>, acc: Vec<f64>, ts: Vec<usize> } -- one entry per class
```

例:

```text
weights["UW4:猫"]["B-NOUN"] = 2.5
weights["UC4:H"]["B-NOUN"]  = 1.8
weights["UW4:猫"]["O"]      = -0.3
...
```

ある特徴量集合に対し、各クラスのスコアはその特徴量の重みの合計です:

```text
score(class) = sum(weights[feature][class] for each feature in input)
prediction = argmax(score(class) for all classes)
```

### 更新規則

各学習インスタンスについて、予測が正解と異なる場合に重みを更新します:

```text
For each training instance (features, truth):
    guess = predict(features)

    if guess != truth:
        For each feature f in features:
            weights[f][truth] += 1.0   # increase weight for correct class
            weights[f][guess] -= 1.0   # decrease weight for predicted class
```

この単純な更新規則により、正解クラスの特徴量が強化され、誤予測クラスの特徴量が弱められます。これにより、将来の類似入力に対して正しい予測がより起こりやすくなります。

### 平均化

基本的なパーセプトロンに対する重要な改善点が**重みの平均化**です。最終的な重みは学習データの末尾に過適合する傾向があるため、学習中に観測されたすべての重みベクトルの平均を最終モデルとして使用します。これにより、未知データへの汎化性能が向上します。

実装では効率のために**累積和**アプローチを使用します:

```text
cumulative[feature][class] += weights[feature][class] * elapsed_steps

At the end of training:
    averaged[feature][class] = cumulative[feature][class] / total_steps
```

これにより、すべての中間重みベクトルを保存することなく同じ結果が得られます。この平均化により、学習データの順序への依存が軽減され、汎化性能が向上します。

累積和とタイムスタンプのベクトルは、学習がその特徴量に初めて触れた時点で遅延生成されます。そのため、推論専用に読み込まれたモデルは
現在の重みのみを保持し、平均化用の状態には一切コストがかかりません。

### エポックによる学習

学習は指定されたエポック数だけ学習データを繰り返します。各エポックでは、すべての学習インスタンスを順に処理します:

```text
For each epoch (1 to num_epochs):
    For each instance in training data:
        features = extract_features(instance)
        predicted = argmax(score(class) for all classes)
        if predicted != correct_label:
            update weights
        accumulate weights for averaging
```

`AtomicBool` フラグにより、Ctrl+C などで学習を中断し、その時点でのモデルを保存することも可能です。

```rust
use std::sync::atomic::AtomicBool;
use litsea::perceptron::AveragedPerceptron;

let mut perceptron = AveragedPerceptron::new();
// ... add instances ...
let running = AtomicBool::new(true);
perceptron.train(10, &running);  // 10 epochs
```

## モデルファイル形式

Averaged Perceptron のモデルは、以下の構造を持つテキストファイルとして保存されます:

```text
18
O
B-ADJ
B-ADP
...
B-X
feature1\tclass1\tweight1
feature2\tclass2\tweight2
...
```

- **1行目**: クラス数（18）
- **2 行目から N+1 行目**: クラス名（1行に1つ）
- **残りの行**: 特徴量の重み、タブ区切り `feature\tclass\tweight`
- 重みがゼロのエントリは省略される
- 重みの行は特徴量のソート順で書き出されるため、同じモデルを保存すると常にバイト単位で同一のファイルになる。
  読み込み時は順序に依存しない

## AdaBoost との比較

| 項目 | AdaBoost | Averaged Perceptron |
|------|----------|---------------------|
| 分類方式 | 二値分類（+1 / -1） | 多クラス分類（18クラス） |
| 出力 | 単語境界のみ | 単語境界 + 品詞タグ |
| 弱学習器 | 特徴量の決定株 | なし（線形分類器） |
| 重みの管理 | 特徴量ごとに1つの重み | クラス×特徴量の重み行列 |
| 汎化手法 | アンサンブル | 重みの平均化 |
| 学習方式 | サンプル再重み付けによる反復ブースティング | 重み平均化によるオンライン学習 |
| モデルサイズ | 約 86 KB〜2.0 MB（再学習済み japanese/chinese/korean/english.model）／約 16〜22 KB（レガシー RWCP/JEITA） | 約 3.6-8 MB（二段構成モデル） |
| ハイパーパラメータ | `threshold`, `num_iterations` | `num_epochs` |

## ハイパーパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `num_epochs` | 10 | 学習エポック数。高い値は精度を向上させる可能性があるが、過学習のリスクがある |
