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
| `external_corpus/*` | tokenizer-speed-bench と同一のコーパススループット計測（後述） |
| `char_type_hiragana` | 文字種分類 |
| `add_corpus` | 学習用コーパスの取り込み |
| `predict_adaboost` | 単一の AdaBoost 予測 |

モデルは `load_model_from_path` で同期的に読み込まれます。ベンチマークに非同期ランタイムは関与しません。

## コーパススループット（`external_corpus`）

`external_corpus` グループは、外部の
[tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
にある litsea の 7 ベンチをリポジトリ内で再現します。これにより、
スループットの回帰を `cargo bench` だけで検出できます:

```sh
cargo bench --bench bench -- external_corpus
```

| ベンチ ID | モデル | コーパス |
|----------|-------|--------|
| `japanese` | japanese.model | wagahaiwa_nekodearu.txt |
| `japanese-rwcp` | RWCP.model | wagahaiwa_nekodearu.txt |
| `japanese-two-stage` | japanese_pos.model | wagahaiwa_nekodearu.txt |
| `korean` | korean.model | mujeong.txt |
| `korean-two-stage` | korean_pos.model | mujeong.txt |
| `chinese` | chinese.model | rulin_waishi.txt |
| `chinese-two-stage` | chinese_pos.model | rulin_waishi.txt |

`*-two-stage` ベンチは[二段構成アーキテクチャ](../algorithm/two-stage-tagging.md)
（#147/#169）と合わせて追加したもので、上記の元々の tokenizer-speed-bench
を再現する 7 ベンチには含まれません。

1 イテレーションでコーパス全行を分割し（外部ベンチと同様、行のフィルタなし）、
グループの `Throughput::Elements` にコーパスの改行を除く文字数を設定しているため、
Criterion の `elem/s` 表示がそのまま **chars/sec** として読めます。

コーパスは `resources/` に外部ベンチとバイト同一で同梱しています:

| コーパス | サイズ | 出典 |
|--------|------|--------|
| wagahaiwa_nekodearu.txt | 約 1.1 MB | 吾輩は猫である（夏目漱石）、青空文庫、パブリックドメイン |
| mujeong.txt | 約 786 KB | 무정（李光洙、1917）、ko.wikisource、パブリックドメイン — 分かち書きされた現代表記の韓国語で、空白対応 korean.model の想定入力 |
| rulin_waishi.txt | 約 985 KB | 儒林外史（呉敬梓）、zh.wikisource、パブリックドメイン — UD Chinese-GSD と同じ繁体字 |

数値は公表されている tokenizer-speed-bench の値と比較可能ですが、方法論の違いに
より完全には一致しません: Criterion はプロセス内のウォームアップ + サンプリング
（外部ベンチは 101 回のプロセスインターリーブ実行）であり、`cargo bench` は litsea の
チューニング済み release プロファイル（thin LTO、codegen-units=1）を継承します
（外部ベンチのクレートはデフォルトの release プロファイル）。

## API 比較（`segment_into`）

`segment_into` グループは、所有出力の `segment()` API とバッファ再利用の
`segment_into()` API（issue #184）を、`external_corpus` と同じ 3 つの
分割コーパス（同じ行単位ワークロード、`Throughput::Elements` による
chars/sec 表示）でペア比較します:

| ベンチ ID | API |
|----------|-----|
| `japanese-strings` / `korean-strings` / `chinese-strings` | `segment()`（トークンごとに `String` 1 つ、呼び出しごとに新規スクラッチ） |
| `japanese-ranges` / `korean-ranges` / `chinese-ranges` | 1 つの `SegmentBuffer` を再利用する `segment_into()` |

同一 run 内で同じ言語の 2 つの ID を比較してください: その差が、バッファ
再利用 API が取り除く呼び出しごとのアロケーションコストです。採点処理は
同一です（`segment()` は `segment_into()` のラッパー）。

```sh
cargo bench -- segment_into
```

### エンジンの数値と CLI の数値

本章の計測はすべて**シングルスレッドのエンジンスループット**です。
CLI の `segment --threads N`（issue #185）はこれに加えてプロセスレベルで
バッチの実時間をコア数にスケールさせますが、この 2 種類の数値は比較
できません — `--threads 8` の実時間はエンジンの高速化ではなく、エンジンの
chars/sec は CLI のスレッドスケーリングについて何も語りません。CLI レベルの
スケーリングを報告する際は、スレッド数を明記し、後述のペア計測の規律で
測定してください。

### 実行間のばらつき

本ドキュメントに掲載している数値（[二段構成タグ付け](../algorithm/two-stage-tagging.md)や
[事前学習済みモデル](../pre-trained-models.md)ページのスループット数値を含む）は、
専用のアイドルなベンチマーク用ハードウェアではなく、本プロジェクトの開発機で
計測しています。同一ビルドで `external_corpus` を 3 回連続実行したところ、
個々のベンチ ID で 10〜20% の振れ幅が見られました -- 1 回の実行を精密な数値と
読むには大きすぎる幅です。ページ内で範囲や「N 回計測」の注記がある場合は
この振れ幅をそのまま反映したものであり、単一の数値のみが示されている場合も
概ね同程度の誤差があるものとして扱ってください。（別の実行で計測した過去の
公表値と比較するのではなく）**同一実行内で**2 つのモデルを比較すると、
両方が同じマシン状態を経験するため、この振れ幅の大半が相殺されます。

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
