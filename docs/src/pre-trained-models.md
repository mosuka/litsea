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
| Training Corpus | UD Korean-GSD |
| Training Options | `-t 0.0001 -i 20000` |
| Word F1 (held-out) | 65.37% |
| Boundary F1 (held-out) | 82.32% |
| File Size | ~20 KB |

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
| File Size | ~17 KB |

## POS Tagging Models

### japanese_pos.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Japanese-GSD (7,050 sentences) |
| Epochs | 10 |
| Accuracy | 98.23% |
| Macro Precision | 96.82% |
| Macro Recall | 93.30% |
| File Size | ~11 MB |

### chinese_pos.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 10 |
| Accuracy | 97.04% |
| Macro Precision | 97.17% |
| Macro Recall | 96.14% |
| File Size | ~19 MB |

### korean_pos.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Korean-GSD (4,400 sentences) |
| Epochs | 10 |
| Accuracy | 95.14% |
| Macro Precision | 95.00% |
| Macro Recall | 86.15% |
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

The `resources/` directory also contains sample data:

- **bocchan.txt** -- Sample Japanese corpus from the novel "Botchan" by Natsume Soseki (~307 KB). Used for benchmarking.
