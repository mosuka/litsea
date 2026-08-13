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
| `segment_short/adaboost/{japanese,chinese,korean}` | 短い文の分割（AdaBoost） |
| `segment_short/averaged_perceptron/{japanese,chinese,korean}` | 短い文の分割+品詞付与 |
| `segment_long_japanese/{adaboost,averaged_perceptron}` | 坊っちゃん全文の処理（約 300 KB） |
| `char_type_hiragana` | 文字種分類 |
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

## リリースプロファイル

`cargo bench` はリリースプロファイルを継承します。このプロファイルでは thin LTO と単一のコード生成ユニット（single codegen unit）が有効になっています（ワークスペースの `Cargo.toml` を参照）。そのため、ベンチマーク結果はリリースバイナリが実際に使用する最適化済みの構成を反映しています。単なる `cargo build`（開発プロファイル）は大幅に低速であり、代表的な数値にはなりません。

## 結果の解釈

パフォーマンスに影響する主な要因:

- **分割処理**は入力長に対して線形（O(n)）
- **文字種分類**は文字範囲に対する `match` で直接行われる（数ナノ秒、セットアップコストなし）
- 各位置での**予測**は特徴量の数に依存（38-42個、定数）
- **モデル読み込み**時間はモデルファイルサイズに比例
