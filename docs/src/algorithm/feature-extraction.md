# Feature Extraction

Litsea uses character n-gram features to capture the local context around each potential word boundary. This chapter catalogs all feature types.

## Feature Categories

For each character position *i* in the input, the segmenter extracts features from a sliding window of characters, their type codes, and previous boundary decisions.

### Base Features (38 features)

| Category | IDs | Description | Window |
|----------|-----|-------------|--------|
| **UW** (Unary Word) | UW1--UW6 | Individual characters at positions i-3 to i+2 | 6 |
| **BW** (Bigram Word) | BW1--BW3 | Adjacent character pairs | 3 |
| **UC** (Unary Char-type) | UC1--UC6 | Character type codes at positions i-3 to i+2 | 6 |
| **BC** (Bigram Char-type) | BC1--BC3 | Adjacent type code pairs | 3 |
| **TC** (Trigram Char-type) | TC1--TC4 | Type code triples | 4 |
| **UP** (Unary Previous-tag) | UP1--UP3 | Previous 3 boundary decisions | 3 |
| **BP** (Bigram Previous-tag) | BP1--BP2 | Boundary decision pairs | 2 |
| **UQ** (Unary tag+type) | UQ1--UQ3 | Combined boundary decision + type code | 3 |
| **BQ** (Bigram tag+type) | BQ1--BQ4 | Combined decision + type code bigrams | 4 |
| **TQ** (Trigram tag+type) | TQ1--TQ4 | Combined decision + type code trigrams | 4 |

### Language-Specific Features (4 features, Japanese and Chinese only)

| Category | IDs | Description | Count |
|----------|-----|-------------|-------|
| **WC** (Word+Char-type) | WC1--WC4 | Character + type code mixed features | 4 |

- `WC1`: character at i-1 + type code at i
- `WC2`: type code at i-1 + character at i
- `WC3`: character at i-1 + type code at i-1
- `WC4`: character at i + type code at i

> **Why no WC for Korean?** Korean Hangul syllables are classified into only two types (SN and SF), so WC features would add noise rather than useful signal.

### Total Feature Count

| Language | Base | WC | Total |
|----------|------|----|-------|
| Japanese | 38 | 4 | **42** |
| Chinese | 38 | 4 | **42** |
| Korean | 38 | 0 | **38** |

## Feature Format

Each feature is represented as a string in the format `PREFIX:VALUE`:

```text
UW4:は        ← The character at position i is "は"
UC4:I         ← The type code at position i is "I" (Hiragana)
BW2:はテ      ← The bigram at position i-1..i is "はテ"
BC2:IK        ← The type bigram is Hiragana + Katakana
UP3:B         ← The previous boundary decision was "B" (boundary)
WC1:はK       ← Character "は" combined with type "K"
```

## Sliding Window Layout

The segmenter pads the input with sentinel characters:

```text
Index:   0    1    2    3    4    5    ...  n+2  n+3  n+4  n+5
Chars:   B3   B2   B1   c1   c2   c3  ...  cn   E1   E2   E3
Types:   O    O    O    t1   t2   t3  ...  tn   O    O    O
Tags:    U    U    U    U    ?    ?   ...  ?
```

- **B3, B2, B1** -- Begin sentinels (padding)
- **E1, E2, E3** -- End sentinels (padding)
- **O** -- "Other" type for padding positions
- **U** -- "Unknown" tag for initial positions
- **B** -- "Boundary" tag (word start)
- **O** -- "Other" tag (continuation)

Features are extracted for positions 4 through len-3, where the full window of i-3 to i+2 is available.

## Training Data Format

The `extract` command writes features to a file in this format:

```text
1	UW1:B2 UW2:B1 UW3:L UW4:i UW5:t UC1:O UC2:O UC3:A UC4:A ...
-1	UW1:B1 UW2:L UW3:i UW4:t UW5:s UC1:O UC2:A UC3:A UC4:A ...
```

Each line contains:
1. A label (`1` for boundary, `-1` for non-boundary)
2. Tab-separated feature strings
