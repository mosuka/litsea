# Segmenter

`Segmenter` 構造体は、単語分割のための主要なインターフェースです。

## 定義

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: two_stage: Option<PackedTwoStageModel>（コンパイル済み stage-2 モデル）
    // private: user_dictionary: Option<UserDictionary>
    // internal: packed cache (see below)
}
```

フィールドは非公開です。アクセサメソッド `language()`、`learner()`、`learner_mut()`、`user_dictionary()` を使ってアクセスしてください。ユーザー辞書は `with_user_dictionary` か `set_user_dictionary` で設定します（[ユーザー辞書](#ユーザー辞書)を参照）。

これらに加えて、構造体は `packed` も保持しています。これは学習器の重みを `segment()` が採点に使う整数インデックスのテーブル群へコンパイルした、遅延再構築されるキャッシュです（[予測パイプライン](../algorithm/prediction-pipeline.md#コンパイル済みスコアリングテーブル)を参照）。これは内部実装の詳細でありアクセサはなく、学習器が変更されると自動的に無効化されます。`two_stage` は [`with_two_stage_learner`](#with_two_stage_learner)（後述）が設定するコンパイル済み stage-2 タグ付けモデルを保持します。二段構成モデルから作成された Segmenter でなければ `None` です。キャッシュとは異なり、保持された学習器から導出されるものではありません — 生の stage-2 パーツはコンパイル後に破棄され、このフィールドへの変更経路は存在しません。

## コンストラクタ

### `Segmenter::new`

```rust
pub fn new(language: Language) -> Self
```

デフォルト（未学習）の `AdaBoost` 学習器を持つ Segmenter を作成します。学習や特徴量抽出に適しています。モデルが読み込まれるか学習データが追加されるまでは、`segment` は文字ごとに1単語を返します。二段構成モデルは未設定のままで、`segment_with_pos` は空でない文に対して `Err(LitseaError::PosLearnerNotSet)` を返します — 単語分割と品詞タグ付けを行うには [`with_two_stage_learner`](#with_two_stage_learner) を使用してください。

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

スペース区切りのコーパスを処理し、先頭を除く文字位置ごとに 1 つずつ、内部の AdaBoost 学習器にインスタンスを追加します。

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

コーパスを処理し、先頭を除く各文字位置の特徴量セットと境界ラベル（`1` = 単語の開始、`-1` = 単語の続き）をコールバックに渡します。先頭の位置のラベルは常に単語の開始なので渡しません。例えば `"テスト です"` は 5 文字なので、コールバックは 4 回呼ばれます。

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

`add_corpus` / `add_corpus_with_writer` のタブ区切りバリアントです。トークンはタブ文字で区切られ、トークンとして空白文字そのもの（`" "`）を含められます。これにより学習テキスト内に元の文の空白が保持され、モデルは空白文字を境界のコンテキストとして学習できます（韓国語モデルおよび英語モデルで使用。issue #152）。`add_corpus` / `add_corpus_with_writer` と同じく、先頭の文字位置は除きます。

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

stage-1 境界分類器で文を単語に分割し（`segment` と全く同じ）、二段構成のタグ付け経路で各単語に UPOS タグを付与します。ユーザー辞書がタグ付きで登録した語には、代わりにそのタグを付けます（[タグ](#タグ)を参照）。空の文に対しては、空のベクターを持つ `Ok` を返します。

**エラー**: 二段構成学習器が設定されておらず、文が空でない場合は `LitseaError::PosLearnerNotSet` を返します（空の文には、学習器を確かめる前に空のベクターを持つ `Ok` を返します） — まず `with_two_stage_learner()` で Segmenter を作成してください。

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

### `add_corpus_tsv_with_pos_writer`

```rust
pub fn add_corpus_tsv_with_pos_writer<F>(&self, corpus: &str, writer: F)
where
    F: FnMut(HashSet<String>, SegmentLabel)
```

`add_corpus_with_pos_writer` のタブ区切りバリアントです（`word/POS\tword/POS\t...`、issue #198）。トークンとして空白文字そのもの（`" "`）を含められるため、学習テキストに元の空白が残ります。空白のトークンには `/POS` が付かないので `B(Upos::X)` のラベルになります。二段構成の extractor は stage-1 のラベルをすべて `B`/`O` にまとめるので、空白に付いたタグはそこでは効きません。スペース区切り版と同じく、最初の位置を含み、Segmenter を変更しません。`Extractor::extract_two_stage` は、タブ区切りのコーパスにこのメソッドを使います。

## ユーザー辞書

`UserDictionary`（issue #263）は、入力に現れたら必ず 1 トークンとして出力する語の集合で、任意に設定できます。モデルを変えるものではなく、境界判定への制約です。モデルは引き続きすべての位置を採点し、辞書は出現箇所の両端（境界を強制）と内側（非境界を強制）の判定を上書きします。語には、`segment_with_pos` がそのトークンに付ける UPOS タグを持たせることもできます（issue #264、[タグ](#タグ)を参照）。本節の出力例は `japanese.model` によるもので、タグの例は `japanese_pos.model` によるものです。

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
    pub fn add_with_pos(&mut self, word: &str, pos: Upos) -> bool
    pub fn contains(&self, word: &str) -> bool
    pub fn pos(&self, word: &str) -> Option<Upos>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn set_gate(&mut self, gate: Option<f64>) -> Result<()>
    pub fn gate(&self) -> Option<f64>
}
```

語は Unicode スカラー値をキーとする trie に格納され、与えられたまま照合されます — 大文字小文字の同一視や、全角半角・Unicode の正規化は行いません。語に空白を含めることもできます。モデルが空白をトークンとして出力する言語（韓国語・英語）では、そのような語は空白を含む 1 トークンになります。1 文字の語には結合すべき内側がないため、`add` は格納せずに `false` を返し、`add_with_pos` はタグだけのエントリとして格納します（[1 文字のエントリ](#1-文字のエントリ)を参照）。どちらも、登録済みの語に対しては `false` を返します。`len` と `contains` は 1 文字のエントリも数えます。この型は `litsea::UserDictionary` として再エクスポートされています。

### 意味論

- 登録語の出現箇所は厳密に 1 トークンとして出力されます。先頭文字の前と末尾文字の後に境界を強制し、内側には境界を置きません。
- 強制した判定は、モデルの判定と全く同じようにモデルの境界タグ履歴（`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*` 特徴量）に入るため、語の後ろの位置は一貫した特徴量で採点されます。`korean.model` や `english.model` のようなタグ非依存（pointwise）モデルでは、辞書の影響は出現箇所そのものに限られます。
- 重なった出現箇所は leftmost-longest で解決します。左から走査し、現在位置から始まる最長の登録語が勝ち、その後ろから照合を再開します。
- 照合は文脈を見ません。登録語はその文字列が現れる場所すべてに適用されるため、`って` や `いま` のような 2 文字の断片を語として登録すると、それを含む他の語をすべて壊します。語全体を登録し、短い語を含む辞書では後述の gate を検討してください。
- タグなしで登録した語は、`segment_with_pos` で他の語と同じようにタグ付けされます。語彙表がその表層を知っていれば語彙表で、知らなければ stage-2 タガーの全候補 argmax で決まります。`add_with_pos` で登録したタグは、そのどちらよりも優先します（[タグ](#タグ)を参照）。
- 空の辞書、登録語が一度も現れない辞書、1 文字のエントリだけの辞書では、`segment` の出力は辞書なしの分割とバイト単位で同一です。

### Gate

`set_gate(Some(ratio))` を設定すると、出現箇所は **flip ratio** が `ratio` 以下のときだけ適用されます。flip ratio は、語が制約する位置のうち制約なしの判定を覆す位置のモデルスコアの絶対値の和を、語が制約するすべての位置の絶対値の和で割ったものです。`0.0..=1.0` に収まり、モデルごとの調整は不要です。`0.0` はモデルがすでに同意している場所だけに語を適用し、`1.0` はすべてに適用し、`None`（既定）は判定を省きます。`0.0..=1.0` の外の値（または NaN）は `LitseaError::InvalidInput` で拒否されます。

gate は短い語や曖昧な語を含む辞書のためのものです。`中野` を登録して gate を `0.0` にすると、`山中野球部` は `山中 | 野球 | 部` のまま（そこではモデルの `中野` に対する margin が大きい）ですが、`私の名前は中野です` では引き続き `中野` が 1 トークンになります（モデルが同意している）。gate なしでは両方の文に語が適用されます。

タグ依存モデルでは、候補を含む文に限り逐次採点パスが 1 回余分にかかります — 比較対象となる制約なしの判定は、前の位置が強制するかもしれない語を考慮せずに計算されるため、近似であることを明記しておきます。pointwise モデルでは追加コストはありません。

### タグ

`add_with_pos(word, pos)` は、2 文字以上の語を `add` と全く同じように登録し（分割への制約も同じです）、その語に UPOS タグを付けます（issue #264）。1 文字の語はタグだけのエントリになります（[1 文字のエントリ](#1-文字のエントリ)を参照）。登録済みの語に対して呼ぶとタグを設定し、前のタグがあれば置き換えます。`add` がタグを消すことはありません。`pos(word)` は登録されたタグを返し、未登録の語やタグなしの語には `None` を返します。

`segment_with_pos` では、表層がタグ付きの語と一致するトークンすべてに、登録されたタグを付けます。語彙表の固定タグと stage-2 タガーのどちらよりも優先し、それ以外のトークンはタガーのタグのままです。stage-2 タガーは各語を、その語自身の表層と前後の文字から採点し、他の語のタグは使わないため、登録したタグが他のトークンのタグを変えることはありません。

- gate なしでは、タグが付くトークンは、辞書が強制した出現箇所とちょうど一致します。
- gate ありでは、gate が却下した出現箇所でも、モデルが結局登録どおりに切り出せばタグが付きます。たとえば、受理された隣の語の強制した境界で、その語が切り出される場合です。トークンとして出てこない出現箇所にはタグは付きません。
- `segment` と `segment_into` はタグを無視します。タグを持たない辞書では、`segment_with_pos` の処理は増えません。

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;
use litsea::upos::Upos;
use litsea::user_dictionary::UserDictionary;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

let mut dictionary = UserDictionary::new();
dictionary.add_with_pos("チャットGPT", Upos::PROPN);
let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner)
    .with_user_dictionary(dictionary);
let tokens = segmenter.segment_with_pos("チャットGPTに聞く")?;
// [("チャットGPT", Upos::PROPN), ("に", Upos::ADP), ("聞く", Upos::VERB)]
// タグなしで登録すると ("チャットGPT", Upos::NOUN)、
// 辞書なしでは ("チャット", Upos::NOUN), ("GPT", Upos::NOUN) になります。
```

### 1 文字のエントリ

1 文字の語には結合すべき内側がないので、分割を制約できず、`add` は無視します。`add_with_pos` は、タグだけのエントリとして登録します（issue #276）。

- 照合の対象になりません。境界を強制せず、`segment` と `segment_into` の出力はエントリがない場合と同じで、gate も関係しません。
- `segment_with_pos` では、その 1 文字ちょうどのトークンに登録したタグが付きます。長いトークンの中にある同じ文字はタガーのタグのままです。そのため、エントリが効くかどうかは、モデルがその文字を 1 トークンとして切り出すかで決まります。
- ほかのタグと同じく文脈を見ないので、文によってタグが変わらない語に向いています。1 文字の名前や、学習データにない記号がその例です。`で` や `に` のような機能語を登録すると、文の中での役割にかかわらず、すべての出現のタグが置き換わります。UD の test セットからこの方法でタグを選ぶと、held-out の文では効果がなく、中国語と韓国語ではタグの正しいトークンが減りました（issue #276）。

```rust
use std::path::Path;

use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;
use litsea::upos::Upos;
use litsea::user_dictionary::UserDictionary;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_pos.model"))?;

// 上の例と同じ準備で、辞書だけをこれに替えた場合:
let mut dictionary = UserDictionary::new();
dictionary.add_with_pos("楓", Upos::PROPN);
let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner)
    .with_user_dictionary(dictionary);
let tokens = segmenter.segment_with_pos("楓が笑った。")?;
// [("楓", Upos::PROPN), ("が", Upos::ADP), ("笑っ", Upos::VERB),
//  ("た", Upos::AUX), ("。", Upos::PUNCT)]
// エントリがなければ、名前は ("楓", Upos::NOUN) になります。
```

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
