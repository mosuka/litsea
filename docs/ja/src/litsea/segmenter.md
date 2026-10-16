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

`add_corpus` / `add_corpus_with_writer` のタブ区切りバリアントです。トークンはタブ文字で区切られ、トークンとして空白文字そのもの（`" "`）を含められます。これにより学習テキスト内に元の文の空白が保持され、モデルは空白文字を境界のコンテキストとして学習できます（韓国語モデルおよび英語モデルで使用。issue #152）。

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

> 文字位置ごとの特徴量抽出（韓国語/英語では38個、日本語・中国語では42個の特徴量）は内部実装の詳細です。以前の `get_attributes` メソッドは非公開になりました。

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
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

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

## ユーザー辞書

`UserDictionary`（issue #263）は、入力に現れたら必ず 1 トークンとして出力する語の集合で、任意に設定できます。モデルを変えるものではなく、境界判定への制約です。モデルは引き続きすべての位置を採点し、辞書は出現箇所の両端（境界を強制）と内側（非境界を強制）の判定を上書きします。本節の出力例は `japanese.model` によるものです。

### `UserDictionary`

```rust
pub struct UserDictionary { /* private */ }

impl UserDictionary {
    pub fn new() -> Self
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>
    pub fn add(&mut self, word: &str) -> bool
    pub fn contains(&self, word: &str) -> bool
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn set_gate(&mut self, gate: Option<f64>) -> Result<()>
    pub fn gate(&self) -> Option<f64>
}
```

語は Unicode スカラー値をキーとする trie に格納され、与えられたまま照合されます — 大文字小文字の同一視や、全角半角・Unicode の正規化は行いません。語に空白を含めることもできます。モデルが空白をトークンとして出力する言語（韓国語・英語）では、そのような語は空白を含む 1 トークンになります。1 文字の語には結合すべき内側がないため、`add` は格納せずに `false` を返します（重複の場合も `false` を返します）。この型は `litsea::UserDictionary` として再エクスポートされています。

### 意味論

- 登録語の出現箇所は厳密に 1 トークンとして出力されます。先頭文字の前と末尾文字の後に境界を強制し、内側には境界を置きません。
- 強制した判定は、モデルの判定と全く同じようにモデルの境界タグ履歴（`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*` 特徴量）に入るため、語の後ろの位置は一貫した特徴量で採点されます。`korean.model` や `english.model` のようなタグ非依存（pointwise）モデルでは、辞書の影響は出現箇所そのものに限られます。
- 重なった出現箇所は leftmost-longest で解決します。左から走査し、現在位置から始まる最長の登録語が勝ち、その後ろから照合を再開します。
- 照合は文脈を見ません。登録語はその文字列が現れる場所すべてに適用されるため、`って` や `いま` のような 2 文字の断片を語として登録すると、それを含む他の語をすべて壊します。語全体を登録し、短い語を含む辞書では後述の gate を検討してください。
- 辞書は品詞情報を持ちません。`segment_with_pos` では、語彙表に未知の強制語は stage-2 タガーの全候補 argmax でタグ付けされます。
- 空の辞書、または登録語が一度も現れない辞書は、辞書なしの分割とバイト単位で同一の出力を残します。

### Gate

`set_gate(Some(ratio))` を設定すると、出現箇所は **flip ratio** が `ratio` 以下のときだけ適用されます。flip ratio は、語が制約する位置のうち制約なしの判定を覆す位置のモデルスコアの絶対値の和を、語が制約するすべての位置の絶対値の和で割ったものです。`0.0..=1.0` に収まり、モデルごとの調整は不要です。`0.0` はモデルがすでに同意している場所だけに語を適用し、`1.0` はすべてに適用し、`None`（既定）は判定を省きます。`0.0..=1.0` の外の値（または NaN）は `LitseaError::InvalidInput` で拒否されます。

gate は短い語や曖昧な語を含む辞書のためのものです。`中野` を登録して gate を `0.0` にすると、`山中野球部` は `山中 | 野球 | 部` のまま（そこではモデルの `中野` に対する margin が大きい）ですが、`私の名前は中野です` では引き続き `中野` が 1 トークンになります（モデルが同意している）。gate なしでは両方の文に語が適用されます。

タグ依存モデルでは、候補を含む文に限り逐次採点パスが 1 回余分にかかります — 比較対象となる制約なしの判定は、前の位置が強制するかもしれない語を考慮せずに計算されるため、近似であることを明記しておきます。pointwise モデルでは追加コストはありません。

### `with_user_dictionary` / `set_user_dictionary` / `user_dictionary`

```rust
pub fn with_user_dictionary(self, dictionary: UserDictionary) -> Self
pub fn set_user_dictionary(&mut self, dictionary: Option<UserDictionary>)
pub fn user_dictionary(&self) -> Option<&UserDictionary>
```

辞書の設定・差し替え・解除と参照を行います。コンパイル済みのスコア表には関与しないためいつでも低コストで変更でき、辞書なしの分割ホットパスは 1 文につき 1 回の判定だけです。

```rust
use std::path::Path;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::user_dictionary::UserDictionary;

let mut learner = AdaBoost::new(0.01, 100);
learner.load_model_from_path(Path::new("./models/japanese.model"))?;

let segmenter = Segmenter::with_learner(Language::Japanese, learner)
    .with_user_dictionary(UserDictionary::from_words(["雪だるま", "新幹線"]));
let tokens = segmenter.segment("雪だるまを作る");
// ["雪だるま", "を", "作る"]   （辞書なしでは ["雪", "だる", "ま", "を", "作る"]）
```
