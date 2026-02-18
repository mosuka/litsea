# Korean

Litsea supports Korean word segmentation with specialized Hangul character type detection.

## Character Types

| Code | Name | Pattern | Examples |
|------|------|---------|----------|
| **E** | Particles/Endings | `[은는을를의에]` | 은, 는, 을, 를, 의, 에 |
| **SN** | Hangul (no 받침) | Codepoint arithmetic | 가, 나, 하, 모 |
| **SF** | Hangul (with 받침) | Codepoint arithmetic | 한, 글, 각, 붙 |
| **J** | Hangul Jamo | U+1100--U+11FF | Individual consonants/vowels |
| **G** | Compatibility Jamo | U+3130--U+318F | ㄱ, ㅏ, ㅎ |
| **H** | Hanja | U+4E00--U+9FFF | CJK Ideographs |
| **P** | Punctuation | CJK Symbols + Full-width | 。, ， |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z |
| **N** | Digits | `[0-9０-９]` | 0, 5, ５ |
| **O** | Other | Fallback | @, #, $ |

### Korean Particles (조사)

The "E" type captures six high-frequency grammatical particles:

| Character | Role | Name |
|-----------|------|------|
| 은/는 | Topic marker | 주격 조사 |
| 을/를 | Object marker | 목적격 조사 |
| 의 | Possessive | 관형격 조사 |
| 에 | Locative | 부사격 조사 |

These particles frequently appear at word boundaries and are given a distinct type code to improve segmentation accuracy.

### Hangul Syllable Structure (받침 Detection)

Korean uses **closure-based matching** instead of regex for SN and SF types. This exploits the systematic Unicode Hangul encoding:

- Hangul Syllables: U+AC00--U+D7AF (11,172 syllables)
- Each syllable = `(initial * 21 + medial) * 28 + final + 0xAC00`
- **SN** (no 받침): `(codepoint - 0xAC00) % 28 == 0`
- **SF** (with 받침): `(codepoint - 0xAC00) % 28 != 0`

The 받침 (final consonant) distinction is linguistically significant because it affects how particles attach to words and where boundaries occur.

### No WC Features

Korean does **not** use WC (word + character-type) features. Since most Hangul syllables fall into only two types (SN and SF), WC features would produce low-entropy, noisy combinations that hurt model accuracy.

## Pre-trained Model

### korean.model

- **Training corpus**: Korean Wikipedia articles
- **Tokenizer**: Lindera with ko-dic dictionary
- **Accuracy**: 85.08%

## Example

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./resources/korean.model
```
