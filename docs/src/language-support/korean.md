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

Because each space is its own single-character token, the character-level
labeling (see [AdaBoost](../algorithm/adaboost.md)) marks two separate
boundaries around it: the space itself starts a new token (label `B`), and
the character immediately following the space starts the next token (also
label `B`). Both are deterministic given the corpus construction -- a space
is always exactly one token, and whatever follows it always begins the next
token -- so the model learns them as a near-trivial rule. Only the second of
these (the boundary that starts the following real word) affects the
held-out Word F1 score, since pure-whitespace tokens are excluded from
scoring (see [Evaluation](../litsea/evaluation.md)).

## Pre-trained Models

### korean.model

- **Training corpus**: UD Korean-GSD (space-preserving TSV corpus)
- **Training options**: `--format tsv`, 30 epochs of Averaged Perceptron
  training, collapsed to AdaBoost scalar weights, not pruned (3,994
  features) -- see [Training
  Procedure](../pre-trained-models.md#training-procedure) for the full recipe
- **Word F1 (held-out)**: 99.90%
- **Boundary F1 (held-out)**: 99.95%

Held-out metrics are computed on the original spaced text with space tokens
excluded from scoring.

### korean_pos.model

- **Algorithm**: Averaged Perceptron (joint segmentation + POS tagging)
- **Details**: see [Pre-trained Models](../pre-trained-models.md#korean_posmodel)

### korean_two_stage.model

- **Algorithm**: two-stage segmentation + POS tagging (faster than the joint model, same output shape)
- **Note**: unlike `korean.model` above, this model is trained on the
  *unspaced* `word/POS` corpus (the same protocol as `korean_pos.model`),
  not the space-preserving protocol described on this page
- **Details**: see [Pre-trained Models](../pre-trained-models.md#korean_two_stagemodel)

## Example

```sh
echo "한국어 단어 분할 테스트입니다." | litsea segment -l korean ./models/korean.model
```
