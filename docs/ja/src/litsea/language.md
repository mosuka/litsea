# Language

`Language` 列挙型は、文字種分類を含む言語固有の動作を定義します。

## Language 列挙型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    English,
}
```

この列挙型に `#[non_exhaustive]` が付いているのは、新しい言語が破壊的変更なしに追加されることを想定しているためです。したがって、外部クレートで `Language` に対する `match` 式を書く場合はワイルドカードアーム（`_ => ...`）が必要です。

### トレイト

- `Default` -- `Language::Japanese` を返す
- `Display` -- 小文字の名前を返す（`"japanese"`、`"chinese"`、`"korean"`、`"english"`）
- `FromStr` -- 完全名または ISO 639-1 コードから解析（大文字・小文字を区別しない）

### パース

```rust
use litsea::language::Language;

// Full names
let ja: Language = "japanese".parse().unwrap();
let zh: Language = "chinese".parse().unwrap();
let ko: Language = "korean".parse().unwrap();
let en: Language = "english".parse().unwrap();

// ISO 639-1 codes
let ja: Language = "ja".parse().unwrap();
let zh: Language = "zh".parse().unwrap();
let ko: Language = "ko".parse().unwrap();
let en: Language = "en".parse().unwrap();

// Case-insensitive
let ko: Language = "KOREAN".parse().unwrap();

// Invalid
assert!("french".parse::<Language>().is_err());
```

### `char_type`

```rust
pub fn char_type(&self, c: char) -> &'static str
```

文字をその言語固有の文字種コードに分類します。どのクラスにも属さない文字には `"O"`（その他）を返します。

分類は文字範囲に対する直接の `match` で行われます -- アロケーション不要、O(1) で、正規表現は使用しません。

```rust
use litsea::language::Language;

let lang = Language::Japanese;
assert_eq!(lang.char_type('あ'), "I");
assert_eq!(lang.char_type('漢'), "H");
assert_eq!(lang.char_type('@'), "O");
```

内部的には、`char_type` は言語ごとの非公開関数（`japanese_char_type_id`、`chinese_char_type_id`、`korean_char_type_id`、`english_char_type_id`）が返す数値の type id に対するテーブル参照になっており、文字列コードと数値 id が食い違うことはありません。全言語に共通のクラス -- `"P"`（句読点）、`"A"`（ラテン文字）、`"N"`（数字） -- は、言語固有のクラスの後にチェックされる共通ヘルパーで処理されます（英語では `"P"` を ASCII の句読点全般まで広げています。詳細は[英語](../language-support/english.md)を参照）。

## ParseLanguageError

文字列からの `Language` のパースに失敗すると `ParseLanguageError` が返されます。この型はクレートルートから再エクスポートされています（`litsea::ParseLanguageError`）:

```rust
use litsea::language::{Language, ParseLanguageError};

let err: ParseLanguageError = "french".parse::<Language>().unwrap_err();
assert_eq!(err.input(), "french");
```

- `input()` -- パースに失敗した文字列を返す
- エラーメッセージにはサポートされている言語が列挙されます: `Unsupported language: 'french'. Supported: japanese (ja), chinese (zh), korean (ko), english (en)`
