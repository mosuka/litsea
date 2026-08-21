# 評価

セグメンテーションおよび品詞タグ付けの held-out 品質評価
（`litsea::evaluation`）です。
[`litsea evaluate`](../litsea-cli/evaluate.md) サブコマンドの背後にある
ライブラリ API です。`evaluate_pos` は
[`with_two_stage_learner`](segmenter.md#with_two_stage_learner) で作成した
Segmenter（[二段構成タグ付け](../algorithm/two-stage-tagging.md)を参照）を
`segment_with_pos` を通じて評価します。

## メトリクス型

```rust
pub struct SegmentationMetrics {
    pub word_precision: f64,     // %
    pub word_recall: f64,        // %
    pub word_f1: f64,            // %
    pub boundary_precision: f64, // %
    pub boundary_recall: f64,    // %
    pub boundary_f1: f64,        // %
    pub sentences: usize,
    pub gold_words: usize,
    pub predicted_words: usize,
}

pub struct PosMetrics {
    pub segmentation: SegmentationMetrics,
    pub tagged_precision: f64, // %: span and tag both match
    pub tagged_recall: f64,    // %
    pub tagged_f1: f64,        // %
}
```

いずれの型もクレートルートで再エクスポートされています。トークンは、
ゴールドトークンを連結した文字列上の文字オフセットスパンの完全一致で
対応付けます。空白のみのトークンはスコア計算から除外されます（韓国語/英語の
空白保持プロトコル。空白を使わずに表記される言語では no-op です）。

## 関数

### `evaluate_segmentation`

```rust
pub fn evaluate_segmentation<I, S>(segmenter: &Segmenter, gold: I) -> SegmentationMetrics
where
    I: IntoIterator<Item = Vec<S>>,
    S: Into<String>,
```

各ゴールド文のトークンを連結したものを [`Segmenter::segment`] で分割し、
その結果をスコアリングします。空の文はスキップされます。

### `evaluate_pos`

```rust
pub fn evaluate_pos<I, S>(segmenter: &Segmenter, gold: I) -> litsea::Result<PosMetrics>
where
    I: IntoIterator<Item = Vec<(S, Upos)>>,
    S: Into<String>,
```

`evaluate_segmentation` と同様ですが、[`Segmenter::segment_with_pos`] を
実行し、タグ付き単語も追加でスコアリングします。Segmenter に POS 学習器も
二段構成学習器も設定されていない場合は `LitseaError::PosLearnerNotSet` を
返します。

### ゴールド行パーサ

```rust
pub fn parse_gold_line(line: &str, tsv: bool) -> Vec<String>
pub fn parse_gold_pos_line(line: &str, tsv: bool) -> Vec<(String, Upos)>
```

どちらもスペースで分割します（`tsv = true` の場合はタブで分割し、
トークンとして空白文字そのもの（`" "`）を含められます — スペース区切りで
書かれる言語で重要になる理由は[英語](../language-support/english.md)を
参照）。`parse_gold_pos_line` はさらに各トークンを**最後の** `/` で分割し
（学習パイプラインと同じルール）、タグが欠落しているか解析できない場合は
`Upos::X` を既定値とします。`tsv = true` の場合、空白リテラルトークンには
`/POS` サフィックスがなく常に `Upos::X` になりますが、これは無害です --
空白トークンはタグ付きスコアリングから内容によって除外され、割り当てられた
タグでは判定しません。これは二段構成 POS モデルの実運用精度を、スペース
区切りで書かれる言語で測定するための仕組みです（issue #196）。韓国語・英語の
二段構成モデルは現在この空白保持コーパスで学習しているため（issue #198）、
これらの言語ではこの形式が学習時と実際の入力の双方に一致するプロトコルに
なっています。

## 使用例

```rust
use litsea::adaboost::AdaBoost;
use litsea::evaluation::{evaluate_segmentation, parse_gold_line};
use litsea::language::Language;
use litsea::segmenter::Segmenter;

let mut learner = AdaBoost::new(0.01, 100);
learner.load_model_from_path(std::path::Path::new("./models/japanese.model"))?;
let segmenter = Segmenter::with_learner(Language::Japanese, learner);

let gold = std::fs::read_to_string("./resources/eval/japanese_gsd_test.txt")?;
let sentences = gold.lines().map(|l| parse_gold_line(l, false));
let metrics = evaluate_segmentation(&segmenter, sentences);
println!("word F1: {:.2}%", metrics.word_f1);
# Ok::<(), Box<dyn std::error::Error>>(())
```
