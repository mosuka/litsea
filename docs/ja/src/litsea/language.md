# Language

`Language` 列挙型と `CharTypePatterns` 構造体は、言語固有の動作を定義します。

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

### `char_type_patterns`

```rust
pub fn char_type_patterns(&self) -> CharTypePatterns
```

この言語に対応する文字種パターンを作成します。呼び出しのたびに正規表現パターンをコンパイルするため、パフォーマンスのために結果をキャッシュすることを推奨します（`Segmenter::new` は自動的にキャッシュします）。

## CharTypePatterns

```rust
pub struct CharTypePatterns {
    // internal: Vec<(CharMatcher, &'static str)>
}
```

### `get_type`

```rust
pub fn get_type(&self, ch: &str) -> &str
```

文字をその文字種コードに分類します。一致するパターンがない場合は `"O"`（その他）を返します。

```rust
let patterns = Language::Japanese.char_type_patterns();
assert_eq!(patterns.get_type("あ"), "I");
assert_eq!(patterns.get_type("漢"), "H");
assert_eq!(patterns.get_type("@"), "O");
```

### `new`

```rust
pub fn new(patterns: Vec<(Regex, &'static str)>) -> Self
```

正規表現と文字種コードのペアのリストからパターンを作成します。パターンは順番にチェックされ、最初に一致したものが使用されます。
