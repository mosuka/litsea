# UPOS

The `upos` module defines the Universal POS (UPOS) tagset and segment label types used for POS tagging.

## Upos

### Definition

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Upos {
    ADJ,    // Adjective
    ADP,    // Adposition
    ADV,    // Adverb
    AUX,    // Auxiliary
    CCONJ,  // Coordinating conjunction
    DET,    // Determiner
    INTJ,   // Interjection
    NOUN,   // Noun
    NUM,    // Numeral
    PART,   // Particle
    PRON,   // Pronoun
    PROPN,  // Proper noun
    PUNCT,  // Punctuation
    SCONJ,  // Subordinating conjunction
    SYM,    // Symbol
    VERB,   // Verb
    X,      // Other
}
```

Litsea supports all 17 UPOS tags from the [Universal Dependencies](https://universaldependencies.org/u/pos/) project:

| Tag | Description | Example (Japanese) |
|-----|-------------|-------------------|
| `ADJ` | Adjective | いい, 大きい |
| `ADP` | Adposition | は, が, を, に |
| `ADV` | Adverb | とても, まだ |
| `AUX` | Auxiliary | です, ます, た |
| `CCONJ` | Coordinating conjunction | と, や |
| `DET` | Determiner | この, その |
| `INTJ` | Interjection | ああ, はい |
| `NOUN` | Noun | 天気, 本 |
| `NUM` | Numeral | 一, 二, 100 |
| `PART` | Particle | ね, よ |
| `PRON` | Pronoun | これ, それ |
| `PROPN` | Proper noun | 東京, 太郎 |
| `PUNCT` | Punctuation | 。, 、 |
| `SCONJ` | Subordinating conjunction | ので, から |
| `SYM` | Symbol | %, $ |
| `VERB` | Verb | 読む, 書く |
| `X` | Other | (unclassified tokens) |

### Constant

#### `Upos::ALL`

```rust
pub const ALL: [Upos; 17]
```

Returns an array of all 17 UPOS tags.

### Trait Implementations

- `Display`: Converts to a string such as `"NOUN"`, `"VERB"`, etc.
- `FromStr`: Parses a string into `Upos`. Returns a `ParseUposError` for invalid strings.

```rust
use litsea::upos::Upos;

let pos: Upos = "NOUN".parse().unwrap();
assert_eq!(pos.to_string(), "NOUN");
```

### ParseUposError

`ParseUposError` (re-exported at the crate root as `litsea::ParseUposError`) is returned when a string is not a valid UPOS tag. Its `input()` accessor returns the string that failed to parse, and the message reads `Unknown UPOS tag: '<input>'`.

## SegmentLabel

### Definition

The `SegmentLabel` type combines word boundary detection with POS tagging. Each character position is assigned one of 18 labels:

- **`B(Upos)`** (17 labels): Word boundary with the given UPOS tag (e.g., `B-NOUN`, `B-VERB`)
- **`O`** (1 label): Non-boundary (continuation of the current word)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    B(Upos),  // Start of a word (boundary). Carries POS information.
    O,        // Continuation of a word (non-boundary).
}
```

```rust
use litsea::upos::SegmentLabel;

// Segment labels for "今日は" (kyou wa)
// 今 → B-NOUN  (start of "今日", tagged as NOUN)
// 日 → O       (continuation of "今日")
// は → B-ADP   (start of "は", tagged as ADP)
```

### Methods

#### `all_labels`

```rust
pub fn all_labels() -> Vec<SegmentLabel>
```

Returns a vector of all 18 `SegmentLabel` values (not strings): the 17 `B(Upos)` labels followed by `O`.

#### `is_boundary`

```rust
pub fn is_boundary(&self) -> bool
```

Returns whether this is a boundary label (`B-*`).

#### `pos`

```rust
pub fn pos(&self) -> Option<Upos>
```

Returns the UPOS tag. Returns `None` for the non-boundary label (`O`).

### Trait Implementations

- `Display`: Converts to a string such as `"B-NOUN"`, `"O"`, etc.
- `FromStr`: Parses a string into `SegmentLabel`. Returns a `ParseSegmentLabelError` for invalid strings.

```rust
use litsea::upos::{SegmentLabel, Upos};

let label: SegmentLabel = "B-NOUN".parse().unwrap();
assert!(label.is_boundary());
assert_eq!(label.pos(), Some(Upos::NOUN));

let label_o: SegmentLabel = "O".parse().unwrap();
assert!(!label_o.is_boundary());
assert_eq!(label_o.pos(), None);
```

### ParseSegmentLabelError

`ParseSegmentLabelError` (re-exported at the crate root as `litsea::ParseSegmentLabelError`) is returned when a string is not a valid segment label. It has two variants:

- `InvalidFormat` -- the string is neither `O` nor of the form `B-<UPOS>` (message: `Invalid segment label: '<input>'. Expected 'O' or 'B-<UPOS>'`)
- `InvalidPos` -- the `B-` prefix was present but the POS part failed to parse (wraps the underlying `ParseUposError`)
