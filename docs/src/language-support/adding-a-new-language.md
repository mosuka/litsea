# Adding a New Language

Litsea's multilingual framework is designed to be easily extensible. This guide explains how to add support for a new language.

## Steps Overview

1. Add a variant to the `Language` enum
2. Implement `Display` and `FromStr` match arms
3. Create a character type pattern function
4. Register the pattern function
5. Decide on WC feature inclusion
6. Prepare a training corpus and train a model
7. Add tests

## Step 1: Add a Variant to `Language`

In `litsea/src/language.rs`, add a new variant to the `Language` enum:

```rust
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    Thai,       // ← new language
}
```

## Step 2: Implement Display and FromStr

Add match arms for the new language:

```rust
// In Display impl
Language::Thai => write!(f, "thai"),

// In FromStr impl
"thai" | "th" => Ok(Language::Thai),
```

## Step 3: Create Character Type Patterns

Define a function that returns `CharTypePatterns` for the new language:

```rust
fn thai_patterns() -> CharTypePatterns {
    CharTypePatterns::from_matchers(vec![
        // Thai characters (U+0E01-U+0E3A)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E01}-\u{0E3A}]").unwrap()
        ), "T"),
        // Thai vowels (U+0E40-U+0E4E)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E40}-\u{0E4E}]").unwrap()
        ), "V"),
        // Thai digits (U+0E50-U+0E59)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E50}-\u{0E59}]").unwrap()
        ), "N"),
        // ASCII + Full-width Latin
        (CharMatcher::Regex(
            Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap()
        ), "A"),
        // Digits
        (CharMatcher::Regex(
            Regex::new(r"[0-9０-９]").unwrap()
        ), "N"),
    ])
}
```

### Design Tips for Character Types

- **Identify linguistically distinct categories** that correlate with word boundary patterns
- **Order matters** -- first match wins, so put more specific patterns before general ones
- **Consider high-frequency function words** as a separate type (as Chinese does with "F")
- **Use closures** for complex logic that cannot be expressed as a single regex

## Step 4: Register the Pattern Function

Add a match arm in `Language::char_type_patterns()`:

```rust
pub fn char_type_patterns(&self) -> CharTypePatterns {
    match self {
        Language::Japanese => japanese_patterns(),
        Language::Chinese => chinese_patterns(),
        Language::Korean => korean_patterns(),
        Language::Thai => thai_patterns(),    // ← new
    }
}
```

## Step 5: Decide on WC Feature Inclusion

In `segmenter.rs`, `get_attributes()` has a `match` on the language to decide whether to include WC features:

```rust
match self.language {
    Language::Japanese | Language::Chinese => {
        // Include WC features
        attrs.insert(format!("WC1:{}{}", w3, c4));
        attrs.insert(format!("WC2:{}{}", c3, w4));
        attrs.insert(format!("WC3:{}{}", w3, c3));
        attrs.insert(format!("WC4:{}{}", w4, c4));
    }
    _ => {}
}
```

If your language's character types have enough variety to make WC features informative, add it to the match arm. If your type system is low-entropy (like Korean's SN/SF), it is better to exclude WC features.

## Step 6: Prepare Corpus and Train a Model

1. **Prepare a corpus** with words separated by spaces:
   ```text
   word1 word2 word3 word4
   ```

2. **Extract features**:
   ```sh
   litsea extract -l thai ./corpus.txt ./features.txt
   ```

3. **Train a model**:
   ```sh
   litsea train -t 0.005 -i 1000 ./features.txt ./resources/thai.model
   ```

## Step 7: Add Tests

Add tests in both `language.rs` and `segmenter.rs`:

```rust
// In language.rs tests
#[test]
fn test_thai_patterns() {
    let p = Language::Thai.char_type_patterns();
    assert_eq!(p.get_type("ก"), "T");   // Thai consonant
    assert_eq!(p.get_type("A"), "A");   // ASCII
    assert_eq!(p.get_type("@"), "O");   // Other
}

// In segmenter.rs tests
#[test]
fn test_get_type_thai() {
    let segmenter = Segmenter::new(Language::Thai, None);
    assert_eq!(segmenter.get_type("ก"), "T");
}
```

Run all tests to verify:

```sh
cargo test --workspace
```
