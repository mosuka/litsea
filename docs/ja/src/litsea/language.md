# Language

`Language` 列挙型は、文字種分類を含む言語固有の動作を定義します。

## Language 列挙型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
}
```

### トレイト

- `Default` -- `Language::Japanese` を返す
- `Display` -- 小文字の名前を返す（`"japanese"`、`"chinese"`、`"korean"`）
- `FromStr` -- 完全名または ISO 639-1 コードから解析（大文字・小文字を区別しない）

### パース

```rust
use litsea::language::Language;

// 完全名
let ja: Language = "japanese".parse().unwrap();
let zh: Language = "chinese".parse().unwrap();
let ko: Language = "korean".parse().unwrap();

// ISO 639-1 コード
let ja: Language = "ja".parse().unwrap();
let zh: Language = "zh".parse().unwrap();
let ko: Language = "ko".parse().unwrap();

// 大文字・小文字を区別しない
let ko: Language = "KOREAN".parse().unwrap();

// 無効な値
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

内部的には、`char_type` は言語ごとの非公開関数（`japanese_char_type`、`chinese_char_type`、`korean_char_type`）にディスパッチします。全言語に共通のクラス -- `"P"`（句読点）、`"A"`（ラテン文字）、`"N"`（数字） -- は、言語固有のクラスの後にチェックされる共通ヘルパーで処理されます。
