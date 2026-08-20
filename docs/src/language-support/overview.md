# Language Support Overview

Litsea supports word segmentation for four languages through a unified framework based on the `Language` enum.

## Supported Languages

| Language | Enum Variant | CLI Values | Feature Count | Word F1 (held-out) |
|----------|-------------|------------|---------------|--------------------|
| Japanese | `Language::Japanese` | `japanese`, `ja` | 42 | 96.70% |
| Chinese | `Language::Chinese` | `chinese`, `zh` | 42 | 90.69% |
| Korean | `Language::Korean` | `korean`, `ko` | 38 | 99.91% |
| English | `Language::English` | `english`, `en` | 38 | 98.31% |

## The Language Enum

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

- **Default** is `Japanese`
- Marked `#[non_exhaustive]` -- new languages can be added without a breaking change, so external `match` expressions need a wildcard arm
- Implements `FromStr` -- parses from full name or ISO 639-1 code (case-insensitive)
- Implements `Display` -- outputs the lowercase full name

### Parsing Examples

```rust
use litsea::language::Language;

let ja: Language = "japanese".parse().unwrap();
let zh: Language = "zh".parse().unwrap();
let ko: Language = "Korean".parse().unwrap();   // case-insensitive
let err = "french".parse::<Language>();          // Err(...)
```

## How Languages Differ

Each language defines its own **character type patterns** that classify characters into type codes. These type codes are used as features for the AdaBoost classifier.

| Aspect | Japanese | Chinese | Korean | English |
|--------|----------|---------|--------|---------|
| Character types | 8 (M, H, I, K, P, A, N, O) | 9 (F, C, X, R, P, B, A, N, O) | 10 (E, SN, SF, J, G, H, P, A, N, O) | 7 (U, W, Q, P, A, N, O) |
| WC features | Yes (4 extra) | Yes (4 extra) | No | No |
| Total features | 42 | 42 | 38 | 38 |
| Matching method | `match` on char ranges | `match` on char ranges | `match` on char ranges + codepoint test | `match` on char ranges |

### Why Korean and English Have Fewer Features

Korean Hangul syllables are classified into only two types: **SN** (without 받침/final consonant) and **SF** (with 받침). This binary distinction means WC features (word + character-type combinations) would produce redundant information with little discriminative power. Excluding them reduces noise and keeps the model compact.

English shares the same conclusion for a related reason: the dominant
boundary signal is whitespace, and a dev-split comparison measured the
38-template (no WC) tag-free model at 98.68% Word F1 versus 98.65% with
all 42 templates — WC features measured *worse*, not just unhelpful. See
[English](english.md#no-wc-features) for the full comparison.
