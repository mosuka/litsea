# Segmenter

`Segmenter` 構造体は、単語分割のための主要なインターフェースです。

## 定義

```rust
pub struct Segmenter {
    // private: language: Language,
    // private: learner: AdaBoost,
    // private: pos_learner: Option<AveragedPerceptron>,
}
```

フィールドは非公開です。アクセサメソッド `language()`、`learner()`、`learner_mut()`、`pos_learner()`、`pos_learner_mut()` を使ってアクセスしてください。

## コンストラクタ

### `Segmenter::new`

```rust
pub fn new(language: Language, learner: Option<AdaBoost>) -> Self
```

新しい Segmenter を作成します。

- `language` -- 文字種分類に使用する言語
- `learner` -- 学習済みの `AdaBoost` モデル（オプション）。`None` の場合、デフォルト（未学習）のインスタンスが作成されます。

```rust
use litsea::language::Language;
use litsea::segmenter::Segmenter;

// 学習済みモデルを使用する場合
let segmenter = Segmenter::new(Language::Japanese, Some(learner));

// モデルなし（学習や特徴量抽出用）
let segmenter = Segmenter::new(Language::Japanese, None);
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

### `char_type`

```rust
pub fn char_type(&self, ch: &str) -> &str
```

言語固有のルールを使用して、1文字をその文字種コードに分類します。`&str` の先頭の文字が分類対象になります。空文字列の場合は `"O"` を返します。

```rust
let segmenter = Segmenter::new(Language::Japanese, None);
assert_eq!(segmenter.char_type("あ"), "I");  // ひらがな
assert_eq!(segmenter.char_type("漢"), "H");  // 漢字
assert_eq!(segmenter.char_type("A"), "A");   // ASCII
```

### `add_corpus`

```rust
pub fn add_corpus(&mut self, corpus: &str)
```

スペース区切りのコーパスを処理し、内部の AdaBoost 学習器にインスタンスを追加します。

```rust
let mut segmenter = Segmenter::new(Language::Japanese, None);
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

### アクセサ

```rust
pub fn language(&self) -> Language
pub fn learner(&self) -> &AdaBoost
pub fn learner_mut(&mut self) -> &mut AdaBoost
pub fn pos_learner(&self) -> Option<&AveragedPerceptron>
pub fn pos_learner_mut(&mut self) -> Option<&mut AveragedPerceptron>
```

Segmenter の言語と内部の学習器へのアクセスを提供します。

> 文字位置ごとの特徴量抽出（韓国語では38個、日本語・中国語では42個の特徴量）は内部実装の詳細です。以前の `get_attributes` メソッドは非公開になりました。
