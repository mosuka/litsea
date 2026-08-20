# Japanese

Japanese is the default language in Litsea.

## Character Types

| Code | Name | Pattern | Examples |
|------|------|---------|----------|
| **M** | Kanji Numbers | `[一二三四五六七八九十百千万億兆]` | 一, 三, 千, 億 |
| **H** | Kanji / CJK | U+4E00--U+9FFF, plus 々〆ヵヶ | 漢, 字, 学, 々 |
| **I** | Hiragana | `[ぁ-ん]` | あ, い, う, を |
| **K** | Katakana | `[ァ-ヴーｱ-ﾝﾞﾟ]` | ア, カ, ー, ﾊ |
| **P** | Punctuation | CJK Symbols + Full-width | 。, 、, 「, 」 |
| **A** | ASCII/Latin | `[a-zA-Zａ-ｚＡ-Ｚ]` | A, z, Ｂ |
| **N** | Digits | `[0-9０-９]` | 0, 5, ５ |
| **O** | Other | Fallback | @, #, $ |

### Pattern Priority

Patterns are evaluated in order. Notably:

- **M before H**: Characters like 一 and 百 are classified as "Kanji Numbers" (M), not generic "Kanji" (H)
- This distinction helps the model learn number-specific boundary patterns

## Pre-trained Models

### japanese.model

- **Training corpus**: UD Japanese-GSD
- **Training options**: 50 epochs of Averaged Perceptron training, collapsed
  to AdaBoost scalar weights, then pruned to the top 40,000 features by
  `|weight|` -- see [Training
  Procedure](../pre-trained-models.md#training-procedure) for the full recipe
- **Word F1 (held-out)**: 96.70%
- **Boundary F1 (held-out)**: 98.59%

### japanese_pos.model

- **Algorithm**: two-stage segmentation + POS tagging (a binary boundary
  classifier plus a word-level tagger with a candidate-tag lexicon)
- **Details**: see [Pre-trained Models](../pre-trained-models.md#japanese_posmodel)

### RWCP.model

- **Source**: Extracted from the original TinySegmenter
- **License**: BSD 3-Clause (Taku Kudo)
- **Size**: ~22 KB

### JEITA_Genpaku_ChaSen_IPAdic.model

- **Training corpus**: JEITA Project Sugita Genpaku corpus
- **Tokenizer**: ChaSen with IPAdic dictionary
- **Size**: ~16 KB

## Example

```sh
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" \
  | litsea segment -l japanese ./models/RWCP.model
```

Output:

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
```
