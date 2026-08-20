# Segmenter

`Segmenter` 構造体は、単語分割のための主要なインターフェースです。

## 定義

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: two_stage: Option<PackedTwoStageModel>（コンパイル済み stage-2 モデル）
    // internal: packed cache (see below)
}
```

フィールドは非公開です。アクセサメソッド `language()`、`learner()`、`learner_mut()` を使ってアクセスしてください。

これらに加えて、構造体は `packed` も保持しています。これは学習器の重みを `segment()` が採点に使う整数インデックスのテーブル群へコンパイルした、遅延再構築されるキャッシュです（[予測パイプライン](../algorithm/prediction-pipeline.md#コンパイル済みスコアリングテーブル)を参照）。これは内部実装の詳細でありアクセサはなく、学習器が変更されると自動的に無効化されます。`two_stage` は [`with_two_stage_learner`](#with_two_stage_learner)（後述）が設定するコンパイル済み stage-2 タグ付けモデルを保持します。二段構成モデルから作成された Segmenter でなければ `None` です。キャッシュとは異なり、保持された学習器から導出されるものではありません — 生の stage-2 パーツはコンパイル後に破棄され、このフィールドへの変更経路は存在しません。

## コンストラクタ

### `Segmenter::new`

```rust
pub fn new(language: Language) -> Self
```

デフォルト（未学習）の `AdaBoost` 学習器を持つ Segmenter を作成します。学習や特徴量抽出に適しています。モデルが読み込まれるか学習データが追加されるまでは、`segment` は文字ごとに1単語を返します。二段構成モデルは未設定のままで、`segment_with_pos` は `Err(LitseaError::PosLearnerNotSet)` を返します — 単語分割と品詞タグ付けを行うには [`with_two_stage_learner`](#with_two_stage_learner) を使用してください。

### `Segmenter::with_learner`

```rust
pub fn with_learner(language: Language, learner: AdaBoost) -> Self
```

指定した学習器を使って Segmenter を作成します。通常は学習済みモデルを読み込んだ学習器を渡します。

```rust
use litsea::language::Language;
use litsea::segmenter::Segmenter;

// With a pre-trained model
let segmenter = Segmenter::with_learner(Language::Japanese, learner);

// Without a model (for training or feature extraction)
let segmenter = Segmenter::new(Language::Japanese);
```

## メソッド

### `segment`

```rust
pub fn segment(&self, sentence: &str) -> Vec<String>
```

文を単語に分割します。空の入力に対しては空のベクターを返します。

```rust
let tokens = segmenter.segment("これはテストです。");
// ["これ", "は", "テスト", "です", "。"]
```

内部的には [`segment_into`](#segment_into--segmentbuffer) の薄いラッパーで、
呼び出しごとに新しいバッファを使い、各 range を所有 `String` として
実体化します — 採点実装は 1 つだけです。

### `segment_into` / `SegmentBuffer`

```rust
pub struct SegmentBuffer { /* 内部のスクラッチ + 出力ストレージ */ }

impl SegmentBuffer {
    pub fn new() -> Self
}

impl Segmenter {
    pub fn segment_into<'b>(
        &self,
        sentence: &str,
        buf: &'b mut SegmentBuffer,
    ) -> &'b [(usize, usize)]
}
```

`segment` のアロケーションフリー版です（issue #184）。返される各
`(start, end)` ペアは `sentence` へのバイト範囲（`&sentence[start..end]`
がトークン）で、出現順に並び、文を過不足なく敷き詰めます。バッファは
呼び出しごとに必要なアロケーション（コンテキスト配列・スコアバッファ・
タグスクラッチ・出力 range）をすべて所有するため、バッチ処理で 1 つの
バッファを使い回すと、定常状態では分割が一切アロケートしなくなります。
空の入力には空のスライスを返します。

公開スループット水準ではこれが効きます: `segment` はトークンごとに
`String` を 1 つ確保し（バッチ処理では毎秒数百万個）、さらに呼び出し
ごとのスクラッチも確保します — 計測ではバッチプロファイルの約 1/4 が
アロケータ系でした。バッファは借用を持たないプレーンなデータなので、
文・モデル・言語をまたいで再利用できます。並列処理ではスレッドごとに
1 つのバッファを使ってください。

```rust
use litsea::segmenter::{SegmentBuffer, Segmenter};

let mut buf = SegmentBuffer::new();
for line in lines {
    for &(start, end) in segmenter.segment_into(line, &mut buf) {
        let token: &str = &line[start..end];
        // アロケーションなしでトークンを参照・出力
    }
}
```

### `char_type`

```rust
pub fn char_type(&self, c: char) -> &'static str
```

文字を言語固有の文字種コードに分類します（`Language::char_type` に委譲します）。

```rust
let segmenter = Segmenter::new(Language::Japanese);
assert_eq!(segmenter.char_type('あ'), "I");  // Hiragana
assert_eq!(segmenter.char_type('漢'), "H");  // Kanji
assert_eq!(segmenter.char_type('A'), "A");   // ASCII
```

### `add_corpus`

```rust
pub fn add_corpus(&mut self, corpus: &str)
```

スペース区切りのコーパスを処理し、内部の AdaBoost 学習器にインスタンスを追加します。

```rust
let mut segmenter = Segmenter::new(Language::Japanese);
segmenter.add_corpus("テスト です");
```

### `add_corpus_with_writer`

```rust
pub fn add_corpus_with_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, i8),
```

コーパスを処理し、各文字位置の特徴量セットとラベルをコールバックに渡します。

```rust
segmenter.add_corpus_with_writer("テスト です", |attrs, label| {
    println!("Features: {:?}, Label: {}", attrs, label);
});
```

### `add_corpus_tsv` / `add_corpus_tsv_with_writer`

```rust
pub fn add_corpus_tsv(&mut self, corpus: &str)
pub fn add_corpus_tsv_with_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, i8),
```

`add_corpus` / `add_corpus_with_writer` のタブ区切りバリアントです。トークンはタブ文字で区切られ、トークンとして空白文字そのもの（`" "`）を含められます。これにより学習テキスト内に元の文の空白が保持され、モデルは空白文字を境界のコンテキストとして学習できます（韓国語モデルで使用。issue #152）。

```rust
let mut segmenter = Segmenter::new(Language::Korean);
segmenter.add_corpus_tsv("나는\t \t고양이");
```

### アクセサ

```rust
pub fn language(&self) -> Language
pub fn learner(&self) -> &AdaBoost
pub fn learner_mut(&mut self) -> &mut AdaBoost
```

Segmenter の言語と内部の学習器（二段構成の Segmenter では stage-1 境界分類器）へのアクセスを提供します。

> 文字位置ごとの特徴量抽出（韓国語では38個、日本語・中国語では42個の特徴量）は内部実装の詳細です。以前の `get_attributes` メソッドは非公開になりました。

## POS-Mode API

Segmenter は、二段構成モデル（issue #147）を使った **単語分割と品詞タグ付け** もサポートしています。

### `with_two_stage_learner`

```rust
pub fn with_two_stage_learner(language: Language, learner: TwoStageLearner) -> Self
```

二段構成モデル（`litsea-two-stage v1` ファイルを読み込んだ `TwoStageLearner`）を持つ
Segmenter を作成します。モデルの stage-1 境界分類器はそのまま Segmenter の
AdaBoost 経路の学習器になり（`segment` は自然に動作します）、`segment_with_pos` は
分割された各単語を候補タグ語彙表（lexicon）でタグ付けします — 単一候補・優勢候補の
表層は分類器を完全にスキップし、曖昧な表層は候補マスク付き argmax、未知の表層は
全クラスの argmax を stage-2 の単語単位タガーが決定します。
二段構成形式については[モデルファイル形式](../advanced/model-file-format.md)を
参照してください。

### `segment_with_pos`

```rust
pub fn segment_with_pos(&self, sentence: &str) -> Result<Vec<(String, Upos)>>
```

stage-1 境界分類器で文を単語に分割し（`segment` と全く同じ）、二段構成のタグ付け経路で各単語に UPOS タグを付与します。空の文に対しては、空のベクターを持つ `Ok` を返します。

**エラー**: 二段構成学習器が設定されていない場合は `LitseaError::PosLearnerNotSet` を返します — まず `with_two_stage_learner()` で Segmenter を作成してください。

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_two_stage.model"))?;

let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);
let tokens = segmenter.segment_with_pos("これはテストです。")?;
// [("これ", Upos::PRON), ("は", Upos::ADP), ("テスト", Upos::NOUN),
//  ("です", Upos::AUX), ("。", Upos::PUNCT)]
```

### `add_corpus_with_pos_writer`

```rust
pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, SegmentLabel)
```

POS タグ付きコーパス（`word/POS word/POS ...`）の文字レベル学習用特徴量を、最初の位置を含めて、Segmenter を変更することなくカスタムライターへストリーミングします。これは `Extractor::extract_two_stage` が stage-1 境界特徴量を構築する際の基盤となっています。
