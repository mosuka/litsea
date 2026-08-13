# Language

The `Language` enum defines language-specific behavior, including character type classification.

## Language Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
}
```

The enum is marked `#[non_exhaustive]` because new languages are expected to be added without a breaking change; external `match` expressions over `Language` therefore need a wildcard arm (`_ => ...`).

### Traits

- `Default` -- Returns `Language::Japanese`
- `Display` -- Returns lowercase name (`"japanese"`, `"chinese"`, `"korean"`)
- `FromStr` -- Parses from full name or ISO 639-1 code (case-insensitive)

### Parsing

```rust
use litsea::language::Language;

// Full names
let ja: Language = "japanese".parse().unwrap();
let zh: Language = "chinese".parse().unwrap();
let ko: Language = "korean".parse().unwrap();

// ISO 639-1 codes
let ja: Language = "ja".parse().unwrap();
let zh: Language = "zh".parse().unwrap();
let ko: Language = "ko".parse().unwrap();

// Case-insensitive
let ko: Language = "KOREAN".parse().unwrap();

// Invalid
assert!("french".parse::<Language>().is_err());
```

### `char_type`

```rust
pub fn char_type(&self, c: char) -> &'static str
```

Classifies a character into its language-specific type code. Returns `"O"` (Other) if the character does not belong to any class.

Classification is a direct `match` on character ranges -- allocation-free, O(1), and with no regex involved.

```rust
use litsea::language::Language;

let lang = Language::Japanese;
assert_eq!(lang.char_type('あ'), "I");
assert_eq!(lang.char_type('漢'), "H");
assert_eq!(lang.char_type('@'), "O");
```

Internally, `char_type` is a table lookup over the numeric type id returned by a private per-language function (`japanese_char_type_id`, `chinese_char_type_id`, `korean_char_type_id`), so string codes and numeric ids cannot drift apart. The classes common to all languages -- `"P"` (punctuation), `"A"` (Latin), and `"N"` (digits) -- are handled by a shared helper that is checked after the language-specific classes.

## ParseLanguageError

Parsing a `Language` from a string fails with `ParseLanguageError`, which is re-exported at the crate root (`litsea::ParseLanguageError`):

```rust
use litsea::language::{Language, ParseLanguageError};

let err: ParseLanguageError = "french".parse::<Language>().unwrap_err();
assert_eq!(err.input(), "french");
```

- `input()` -- Returns the string that failed to parse
- The error message enumerates the supported languages: `Unsupported language: 'french'. Supported: japanese (ja), chinese (zh), korean (ko)`
