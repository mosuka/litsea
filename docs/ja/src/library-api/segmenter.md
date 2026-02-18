# Segmenter

`Segmenter` 構造体は、単語分割のための主要なインターフェースです。

## 定義

```rust
pub struct Segmenter {
    pub language: Language,
    pub learner: AdaBoost,
    // internal: char_types: CharTypePatterns
}
```

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

### `get_type`

```rust
pub fn get_type(&self, ch: &str) -> &str
```

言語固有のパターンを使用して、1文字をその文字種コードに分類します。

```rust
let segmenter = Segmenter::new(Language::Japanese, None);
assert_eq!(segmenter.get_type("あ"), "I");  // ひらがな
assert_eq!(segmenter.get_type("漢"), "H");  // 漢字
assert_eq!(segmenter.get_type("A"), "A");   // ASCII
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

### `get_attributes`

```rust
pub fn get_attributes(
    &self,
    i: usize,
    tags: &[String],
    chars: &[String],
    types: &[String],
) -> HashSet<String>
```

特定の文字位置における特徴量セットを抽出します。韓国語では38個、日本語・中国語では42個の特徴量を返します。

> これは主に `segment()` と `process_corpus()` の内部で使用されます。
