# Language

The `Language` enum and `CharTypePatterns` struct define language-specific behavior.

## Language Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
}
```

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

### `char_type_patterns`

```rust
pub fn char_type_patterns(&self) -> CharTypePatterns
```

Creates the character type patterns for this language. Compiles regex patterns on each call -- for performance, cache the result (as `Segmenter::new` does automatically).

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

Classifies a character into its type code. Returns `"O"` (Other) if no pattern matches.

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

Creates patterns from a list of regex + type code pairs. Patterns are checked in order; first match wins.
