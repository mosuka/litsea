# ベンチマーク

Litsea には、パフォーマンス測定のための Criterion ベンチマークスイートが含まれています。

## ベンチマークの実行

```sh
cargo bench --bench bench
```

または Makefile を使用:

```sh
make bench
```

## ベンチマークスイート

ベンチマークは `litsea/benches/bench.rs` で定義されています:

| ベンチマーク | 説明 |
|-----------|------------|
| `segment_short/adaboost/{ja,zh,ko}` | 短い文の分割（AdaBoost） |
| `segment_short/averaged_perceptron/{ja,zh,ko}` | 短い文の分割+品詞付与 |
| `segment_long_japanese/{adaboost,averaged_perceptron}` | 坊っちゃん全文の処理（約 300 KB） |
| `get_type_hiragana` | 文字種分類 |
| `add_corpus` | 学習用コーパスの取り込み |
| `predict_adaboost` | 単一の AdaBoost 予測 |

モデルは `load_model_from_path` で同期的に読み込まれます。ベンチマークに非同期ランタイムは関与しません。

## HTML レポート

Criterion は、統計情報と比較グラフを含む詳細な HTML レポートを以下の場所に生成します:

```text
target/criterion/report/index.html
```

ベンチマーク実行後にこのファイルをブラウザで開くと、以下を確認できます:

- 信頼区間付きの反復時間
- スループット測定
- 前回実行との比較（自動回帰検出）

## 結果の解釈

パフォーマンスに影響する主な要因:

- **分割処理**は入力長に対して線形（O(n)）
- **文字種分類**は文字範囲に対する `match` で直接行われる（数ナノ秒、セットアップコストなし）
- 各位置での**予測**は特徴量の数に依存（38-42個、定数）
- **モデル読み込み**時間はモデルファイルサイズに比例
