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

Korean uses a **range arm with a codepoint test in its body** for the SN and SF types. This exploits the systematic Unicode Hangul encoding:

- Hangul Syllables: U+AC00--U+D7AF (11,172 syllables)
- Each syllable = `(initial * 21 + medial) * 28 + final + 0xAC00`
- **SN** (no 받침): `(codepoint - 0xAC00) % 28 == 0`
- **SF** (with 받침): `(codepoint - 0xAC00) % 28 != 0`

The 받침 (final consonant) distinction is linguistically significant because it affects how particles attach to words and where boundaries occur.

### No WC Features

Korean does **not** use WC (word + character-type) features. Since most Hangul syllables fall into only two types (SN and SF), WC features would produce low-entropy, noisy combinations that hurt model accuracy.

### Space-Preserving Training

Korean is written with spaces between eojeol (word phrases), and those
spaces mark most word boundaries. The Korean model is therefore trained on
a **space-preserving TSV corpus**: tokens are tab-separated and each
inter-eojeol space is kept as its own token, so the training text contains
the space characters of the original sentence and the model can use them as
boundary context. Generate the corpus with `corpus_udtreebank.sh -s`
(which reconstructs spacing from the treebank's `SpaceAfter` annotations)
and extract features with `litsea extract --format tsv`. At inference no
special handling is needed: `segment()` receives the spaced text as-is and
emits each space as its own token.

## Pre-trained Model

### korean.model

- **Training corpus**: UD Korean-GSD (space-preserving TSV corpus)
- **Training options**: `--format tsv`, `-t 0.0001 -i 20000`
- **Word F1 (held-out)**: 99.91%
- **Boundary F1 (held-out)**: 99.96%

Held-out metrics are computed on the original spaced text with space tokens
excluded from scoring.

## Example

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./models/korean.model
```
