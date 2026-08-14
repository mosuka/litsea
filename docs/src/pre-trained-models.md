# Pre-trained Models

Litsea ships with several pre-trained models in the `models/` directory.

## Model Catalog

The word segmentation models are evaluated on the held-out test split of
their training treebank (sentences never seen during training). **Word F1**
scores exact word matches; **Boundary F1** scores individual boundary
decisions. Note that the `train` command prints *in-sample* metrics
(measured on the training data itself), which are higher than these
held-out figures.

### japanese.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | UD Japanese-GSD |
| Training Options | `-t 0.0001 -i 20000` |
| Word F1 (held-out) | 91.48% |
| Boundary F1 (held-out) | 96.31% |
| File Size | ~20 KB |

### korean.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | UD Korean-GSD (space-preserving TSV corpus) |
| Training Options | `--format tsv`, `-t 0.0001 -i 20000` |
| Word F1 (held-out) | 99.91% |
| Boundary F1 (held-out) | 99.96% |
| File Size | ~9.4 KB |

The Korean model is trained and evaluated on text that preserves the
original inter-eojeol spaces (each space is its own token; space tokens are
excluded from the F1 computation). Spaces mark most word boundaries in
Korean, so a model that sees them during training resolves the UD
Korean-GSD standard almost deterministically. Japanese and Chinese are
written without spaces, so their protocol is unchanged.

### chinese.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | UD Chinese-GSD |
| Training Options | `-t 0.0001 -i 20000` |
| Word F1 (held-out) | 77.56% |
| Boundary F1 (held-out) | 87.81% |
| File Size | ~18 KB |

### RWCP.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Source | Extracted from the original [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) |
| License | BSD 3-Clause (Taku Kudo) |
| File Size | ~22 KB |

### JEITA_Genpaku_ChaSen_IPAdic.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | JEITA Project Sugita Genpaku corpus |
| Tokenizer | ChaSen with IPAdic |
| File Size | ~16 KB |

## POS Tagging Models

In-sample rows are the `train` command's metrics on the training data;
held-out rows are word / tagged-word F1 measured with `litsea evaluate
--pos` on the UD GSD test splits (see
[Evaluating Models](training-guide/evaluating-models.md)). The Korean POS
gold follows the POS pipeline's convention (no space tokens), so it is
evaluated on unspaced text.

### japanese_pos.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Japanese-GSD (7,050 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 98.23% |
| Macro Precision (in-sample) | 96.82% |
| Macro Recall (in-sample) | 93.30% |
| Word F1 (held-out) | 96.56% |
| Tagged Word F1 (held-out) | 92.51% |
| File Size | ~11 MB |

### chinese_pos.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 97.04% |
| Macro Precision (in-sample) | 97.17% |
| Macro Recall (in-sample) | 96.14% |
| Word F1 (held-out) | 90.52% |
| Tagged Word F1 (held-out) | 81.18% |
| File Size | ~19 MB |

### korean_pos.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Korean-GSD (4,400 sentences) |
| Epochs | 10 |
| Accuracy (in-sample) | 95.14% |
| Macro Precision (in-sample) | 95.00% |
| Macro Recall (in-sample) | 86.15% |
| Word F1 (held-out) | 80.51% |
| Tagged Word F1 (held-out) | 71.03% |
| File Size | ~8.9 MB |

#### Usage

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_pos.model
```

Output:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## Choosing a Model

- For **Japanese**, use `japanese.model` for the best accuracy, or `RWCP.model` for compatibility with the original TinySegmenter
- For **Chinese**, use `chinese.model`
- For **Korean**, use `korean.model`
- For **POS tagging**, use the corresponding `*_pos.model` (`japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`) for joint word segmentation and POS tagging
- For **domain-specific** needs, consider [training your own model](training-guide/preparing-corpus.md) or [retraining](training-guide/retraining-models.md) an existing one

## Sample Data

The `resources/` directory also contains sample data used for benchmarking:

- **bocchan.txt** -- 坊っちゃん (Natsume Soseki), ~307 KB. Used by the `segment_long_japanese` benchmarks and differential tests.
- **wagahaiwa_nekodearu.txt** -- 吾輩は猫である (Natsume Soseki), ~1.1 MB, Aozora Bunko.
- **mujeong.txt** -- 무정 (Yi Kwang-su, 1917), ~786 KB, ko.wikisource.
- **rulin_waishi.txt** -- 儒林外史 (Wu Jingzi), ~985 KB, zh.wikisource.

The last three are byte-identical to the corpora of the external
[tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
harness and feed the `external_corpus` benchmark group (see
[Benchmarking](advanced/benchmarking.md)). All are public domain.
