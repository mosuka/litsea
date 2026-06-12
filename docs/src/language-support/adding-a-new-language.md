# Adding a New Language

Litsea's multilingual framework is designed to be easily extensible. This guide explains how to add support for a new language.

## Steps Overview

1. Add a variant to the `Language` enum
2. Implement `Display` and `FromStr` match arms
3. Create a character classification function
4. Register the classification function
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

## Step 3: Create a Character Classification Function

Define a function that classifies a `char` into a type code for the new language. Classification is a direct `match` on character ranges (no regex), so each class is an arm; the **first matching arm wins**:

```rust
fn thai_char_type(c: char) -> &'static str {
    match c {
        // Thai consonants and sequential vowels (U+0E01-U+0E3A)
        '\u{0E01}'..='\u{0E3A}' => "T",
        // Thai vowels and tone marks (U+0E40-U+0E4E)
        '\u{0E40}'..='\u{0E4E}' => "V",
        // Thai digits (U+0E50-U+0E59)
        '\u{0E50}'..='\u{0E59}' => "N",
        // Shared classes: "P" (punctuation), "A" (Latin), "N" (digits)
        _ => punct_latin_digit(c).unwrap_or("O"),
    }
}
```

### Design Tips for Character Types

- **Identify linguistically distinct categories** that correlate with word boundary patterns
- **Order matters** -- match arms are tried top to bottom, so put more specific classes before general ones
- **Consider high-frequency function words** as a separate type (as Chinese does with "F")
- **Use match guards** for logic beyond plain ranges (as Korean does to split syllables with/without 받침)
- Reuse the shared `punct_latin_digit()` helper for the common "P"/"A"/"N" classes

## Step 4: Register the Classification Function

Add a match arm in `Language::char_type()`:

```rust
pub fn char_type(&self, c: char) -> &'static str {
    match self {
        Language::Japanese => japanese_char_type(c),
        Language::Chinese => chinese_char_type(c),
        Language::Korean => korean_char_type(c),
        Language::Thai => thai_char_type(c),    // ← new
    }
}
```

## Step 5: Decide on WC Feature Inclusion

In `segmenter.rs`, the internal attribute builder (`write_attributes()`) has a `match` on the language to decide whether to include WC features:

```rust
match self.language {
    Language::Japanese | Language::Chinese => {
        // Include WC features
        attr!("WC1:{}{}", w3, c4);
        attr!("WC2:{}{}", c3, w4);
        attr!("WC3:{}{}", w3, c3);
        attr!("WC4:{}{}", w4, c4);
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
   litsea train -t 0.005 -i 1000 ./features.txt ./models/thai.model
   ```

## Step 7: Add Tests

Add tests in both `language.rs` and `segmenter.rs`:

```rust
// In language.rs tests
#[test]
fn test_thai_char_types() {
    let lang = Language::Thai;
    assert_eq!(lang.char_type('ก'), "T");   // Thai consonant
    assert_eq!(lang.char_type('A'), "A");   // ASCII
    assert_eq!(lang.char_type('@'), "O");   // Other
}

// In segmenter.rs tests
#[test]
fn test_char_type_thai() {
    let segmenter = Segmenter::new(Language::Thai, None);
    assert_eq!(segmenter.char_type("ก"), "T");
}
```

Run all tests to verify:

```sh
cargo test --workspace
```
