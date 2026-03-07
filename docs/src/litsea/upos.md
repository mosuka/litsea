# UPOS

The `upos` module defines the Universal POS (UPOS) tagset and segment label types used for POS tagging.

## UPOS Tags

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

## Segment Labels

The `SegmentLabel` type combines word boundary detection with POS tagging. Each character position is assigned one of 18 labels:

- **`B-{TAG}`** (17 labels): Word boundary with the given UPOS tag (e.g., `B-NOUN`, `B-VERB`)
- **`O`** (1 label): Non-boundary (continuation of the current word)

```rust
use litsea::upos::SegmentLabel;

// Segment labels for "今日は" (kyou wa)
// 今 → B-NOUN  (start of "今日", tagged as NOUN)
// 日 → O       (continuation of "今日")
// は → B-ADP   (start of "は", tagged as ADP)
```

## Functions

### `segment_label_to_upos`

Converts a segment label string (e.g., `"B-NOUN"`) to its UPOS tag (e.g., `"NOUN"`). Returns `None` for the `"O"` label.

### `all_segment_labels`

Returns a vector of all 18 segment label strings.
