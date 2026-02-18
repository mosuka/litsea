# Character Type Classification

Each language in Litsea defines a set of **character type patterns** that classify individual characters into linguistically meaningful categories. These type codes are used as features for the AdaBoost classifier.

## How It Works

The `CharTypePatterns` struct holds an ordered list of `(CharMatcher, type_code)` pairs. For each character, the **first matching pattern** determines the type code. If no pattern matches, the character is classified as `"O"` (Other).

`CharMatcher` supports two matching strategies:

- **Regex** -- Compiled regex patterns for Unicode range matching
- **Closure** -- Custom functions for complex logic (e.g., Korean Hangul syllable structure)

## Japanese Character Types

| Code | Name | Pattern / Range | Examples |
|------|------|----------------|----------|
| **M** | Kanji Numbers | `[一二三四五六七八九十百千万億兆]` | 一, 千, 億 |
| **H** | Kanji / CJK Ideographs | `[一-龠々〆ヵヶ]` | 漢, 字, 学 |
| **I** | Hiragana | `[ぁ-ん]` | あ, い, う |
| **K** | Katakana | `[ァ-ヴーｱ-ﾝﾞﾟ]` | ア, カ, ー |
| **P** | Punctuation | CJK Symbols (U+3000-303F), Full-width (U+FF01-FF65) | 。, 、, 「 |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z, Ｂ |
| **N** | Digits | `[0-9０-９]` | 0, ５ |
| **O** | Other | Fallback | @, # |

> **Note:** "M" (Kanji numbers) is checked before "H" (general Kanji), so characters like 一 and 百 are classified as numbers rather than generic ideographs.

## Chinese Character Types

| Code | Name | Pattern / Range | Examples |
|------|------|----------------|----------|
| **F** | Function Words | High-frequency grammatical words | 的, 了, 在, 是 |
| **C** | CJK Unified | U+4E00--U+9FFF | 中, 国, 人 |
| **X** | CJK Extension A | U+3400--U+4DBF | Rare characters |
| **R** | CJK Radicals | U+2E80--U+2FDF | Kangxi radicals |
| **P** | Punctuation | CJK Symbols + Full-width | 。, ，, 《 |
| **B** | Bopomofo | U+3100--U+312F, U+31A0--U+31BF | Zhuyin symbols |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z |
| **N** | Digits | `[0-9０-９]` | 0, ５ |
| **O** | Other | Fallback | @, # |

**Chinese function words** include:
- Structural particles: 的, 地, 得
- Aspect/modal particles: 了, 着, 过, 吗, 呢, 吧, 啊, 嘛
- Conjunctions: 和, 与, 或, 但, 而, 且, 及
- Prepositions: 在, 从, 到, 把, 被, 对, 向, 给
- Common grammatical verbs/adverbs: 是, 有, 不, 也, 都, 就, 要, 会, 能, 可

## Korean Character Types

| Code | Name | Pattern / Range | Examples |
|------|------|----------------|----------|
| **E** | Particles/Endings | High-frequency grammatical particles | 은, 는, 을, 를, 의, 에 |
| **SN** | Hangul (no batchim) | Hangul Syllable without final consonant | 가, 나, 하 |
| **SF** | Hangul (with batchim) | Hangul Syllable with final consonant | 한, 글, 각 |
| **J** | Hangul Jamo | U+1100--U+11FF | Individual consonants/vowels |
| **G** | Compatibility Jamo | U+3130--U+318F | ㄱ, ㅏ, ㅎ |
| **H** | Hanja | U+4E00--U+9FFF | CJK Ideographs |
| **P** | Punctuation | CJK Symbols + Full-width | 。, ， |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z |
| **N** | Digits | `[0-9０-９]` | 0, ５ |
| **O** | Other | Fallback | @, # |

### Korean Hangul Syllable Detection

Korean uses **closure-based matching** for SN and SF types. This leverages Unicode's systematic Hangul encoding:

- Hangul Syllables occupy U+AC00--U+D7AF
- Each syllable is encoded as: `(initial * 21 + medial) * 28 + final + 0xAC00`
- If `(codepoint - 0xAC00) % 28 == 0`, the syllable has **no final consonant** (SN)
- Otherwise, it **has a final consonant** (SF, "받침")

This distinction is important because the presence of a final consonant (받침) affects Korean word boundary patterns and particle attachment.

## Cross-Language Comparison

| Feature | Japanese | Chinese | Korean |
|---------|----------|---------|--------|
| Total types | 8 | 9 | 10 |
| Unique types | M, H, I, K | F, C, X, R, B | E, SN, SF, J, G |
| Shared types | P, A, N, O | P, A, N, O | P, A, N, O (H shared with JP) |
| Matching method | Regex only | Regex only | Regex + Closure |
| WC features used | Yes | Yes | No |
