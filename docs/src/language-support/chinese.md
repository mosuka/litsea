# Chinese

Litsea supports Chinese word segmentation covering both Simplified and Traditional Chinese.

## Character Types

| Code | Name | Pattern | Examples |
|------|------|---------|----------|
| **F** | Function Words | High-frequency grammatical words | 的, 了, 在, 是, 和 |
| **C** | CJK Unified | U+4E00--U+9FFF | 中, 国, 人 |
| **X** | CJK Extension A | U+3400--U+4DBF | Rare characters |
| **R** | CJK Radicals | U+2E80--U+2FDF | Kangxi radicals |
| **P** | Punctuation | CJK Symbols + Full-width | 。, ，, 《, 》 |
| **B** | Bopomofo | U+3100--U+312F, U+31A0--U+31BF | Zhuyin symbols |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z |
| **N** | Digits | `[0-9０-９]` | 0, 5, ５ |
| **O** | Other | Fallback | @, #, $ |

### Chinese Function Words (虚词)

The "F" type captures high-frequency grammatical words that are critical for segmentation:

| Category | Characters |
|----------|-----------|
| Structural particles | 的, 地, 得 |
| Aspect/modal particles | 了, 着, 过, 吗, 呢, 吧, 啊, 嘛 |
| Conjunctions | 和, 与, 或, 但, 而, 且, 及 |
| Prepositions | 在, 从, 到, 把, 被, 对, 向, 给 |
| Grammatical verbs/adverbs | 是, 有, 不, 也, 都, 就, 要, 会, 能, 可 |

These characters appear overwhelmingly in grammatical roles and signal word boundaries differently from content words.

## Pre-trained Model

### chinese.model

- **Training corpus**: Chinese Wikipedia articles
- **Tokenizer**: Lindera with CC-CEDICT dictionary
- **Accuracy**: 80.72%

## Example

```sh
echo "中文分词测试。" | litsea segment -l chinese ./resources/chinese.model
```
