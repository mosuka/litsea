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
| `segment_japanese_short` | 短い日本語文の分割 |
| `segment_japanese_long` | 坊っちゃんの全文の分割（約 300 KB） |
| `segment_chinese_short` | 短い中国語文の分割 |
| `segment_korean_short` | 短い韓国語文の分割 |
| `get_type_hiragana` | 文字種分類 |
| `add_corpus` | 学習用コーパスの取り込み |
| `char_type_patterns_japanese` | パターンコンパイルのコスト |
| `predict` | 単一の AdaBoost 予測 |

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
- **パターンコンパイル**（正規表現）は最もコストが高い初回処理 -- `Segmenter::new()` がパターンをキャッシュする
- 各位置での**予測**は特徴量の数に依存（38-42個、定数）
- **モデル読み込み**時間はモデルファイルサイズに比例
