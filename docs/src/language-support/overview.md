# Language Support Overview

Litsea supports word segmentation for three languages through a unified framework based on the `Language` enum.

## Supported Languages

| Language | Enum Variant | CLI Values | Feature Count | Pre-trained Model Accuracy |
|----------|-------------|------------|---------------|---------------------------|
| Japanese | `Language::Japanese` | `japanese`, `ja` | 42 | 94.15% |
| Chinese | `Language::Chinese` | `chinese`, `zh` | 42 | 80.72% |
| Korean | `Language::Korean` | `korean`, `ko` | 38 | 85.08% |

## The Language Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
}
```

- **Default** is `Japanese`
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

| Aspect | Japanese | Chinese | Korean |
|--------|----------|---------|--------|
| Character types | 8 (M, H, I, K, P, A, N, O) | 9 (F, C, X, R, P, B, A, N, O) | 10 (E, SN, SF, J, G, H, P, A, N, O) |
| WC features | Yes (4 extra) | Yes (4 extra) | No |
| Total features | 42 | 42 | 38 |
| Matching method | Regex only | Regex only | Regex + Closure |

### Why Korean Has Fewer Features

Korean Hangul syllables are classified into only two types: **SN** (without 받침/final consonant) and **SF** (with 받침). This binary distinction means WC features (word + character-type combinations) would produce redundant information with little discriminative power. Excluding them reduces noise and keeps the model compact.
