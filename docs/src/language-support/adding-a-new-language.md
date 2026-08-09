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

Define a function that classifies a `char` into a **type id** for the new language. Ids are indices into the language's ordered `type_codes()` table (Step 4): the shared classes occupy fixed indices ("O" = 0, "P" = 1, "A" = 2, "N" = 3) and language-specific classes follow from 4. Classification is a direct `match` on character ranges (no regex), so each class is an arm; the **first matching arm wins**:

```rust
fn thai_char_type_id(c: char) -> u8 {
    match c {
        // Thai consonants and sequential vowels (U+0E01-U+0E3A)
        '\u{0E01}'..='\u{0E3A}' => 4, // "T"
        // Thai vowels and tone marks (U+0E40-U+0E4E)
        '\u{0E40}'..='\u{0E4E}' => 5, // "V"
        // Thai digits (U+0E50-U+0E59)
        '\u{0E50}'..='\u{0E59}' => DIGIT_TYPE_ID, // "N"
        // Shared classes: "P" (punctuation), "A" (Latin), "N" (digits)
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
    }
}
```

### Design Tips for Character Types

- **Identify linguistically distinct categories** that correlate with word boundary patterns
- **Order matters** -- match arms are tried top to bottom, so put more specific classes before general ones
- **Consider high-frequency function words** as a separate type (as Chinese does with "F")
- **Use match guards** for logic beyond plain ranges (as Korean does to split syllables with/without 받침)
- Reuse the shared `punct_latin_digit()` helper for the common "P"/"A"/"N" classes
- **Keep the code set prefix-free** -- no code may be a prefix of another (Korean's `SN`/`SF` work because `S` alone is not a code). The model loader decodes concatenated codes left to right when compiling packed feature keys, and prefix-freeness is what makes that decoding unambiguous (a unit test pins this per language)

## Step 4: Register the Type-Code Table and Classification Function

Add the language's ordered code table to `Language::type_codes()` (index = type id; shared codes first) and a dispatch arm in `Language::char_type_id()`. `char_type()` itself is derived from these two, so string codes and numeric ids cannot drift apart:

```rust
pub(crate) fn type_codes(self) -> &'static [&'static str] {
    match self {
        // ...
        Language::Thai => &["O", "P", "A", "N", "T", "V"],    // ← new
    }
}

pub(crate) fn char_type_id(self, c: char) -> u8 {
    match self {
        // ...
        Language::Thai => thai_char_type_id(c),    // ← new
    }
}
```

## Step 5: Decide on WC Feature Inclusion

The feature template is defined once in `packed_model.rs` (`TEMPLATES`), and `templates_for()` decides whether a language uses the trailing `WC1`--`WC4` char/type mixed templates:

```rust
pub(crate) fn templates_for(language: Language) -> &'static [Template] {
    match language {
        Language::Japanese | Language::Chinese => &TEMPLATES[..],
        _ => &TEMPLATES[..BASE_TEMPLATE_COUNT], // 38 base templates
    }
}
```

If your language's character types have enough variety to make WC features informative, add it to the first match arm. If your type system is low-entropy (like Korean's SN/SF), it is better to exclude WC features.

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
    let segmenter = Segmenter::new(Language::Thai);
    assert_eq!(segmenter.char_type("ก"), "T");
}
```

Run all tests to verify:

```sh
cargo test --workspace
```
