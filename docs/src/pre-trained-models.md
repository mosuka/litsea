# Pre-trained Models

Litsea ships with several pre-trained models in the `resources/` directory.

## Model Catalog

### japanese.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | UD Japanese-GSD |
| Accuracy | 94.15% |
| Precision | 95.57% |
| Recall | 94.36% |
| File Size | ~2.9 KB |

### korean.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | UD Korean-GSD |
| Accuracy | 85.08% |
| File Size | ~1.8 KB |

### chinese.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | UD Chinese-GSD |
| Accuracy | 80.72% |
| File Size | ~1.3 KB |

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
| Accuracy | 98.34% |
| Macro Precision | 97.87% |
| Macro Recall | 91.67% |
| File Size | ~11 MB |

### chinese_pos.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Chinese-GSD (3,997 sentences) |
| Epochs | 10 |
| Accuracy | 97.09% |
| Macro Precision | 97.31% |
| Macro Recall | 96.23% |
| File Size | ~19 MB |

### korean_pos.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Algorithm | Averaged Perceptron |
| Training Corpus | UD Korean-GSD (4,400 sentences) |
| Epochs | 10 |
| Accuracy | 95.33% |
| Macro Precision | 95.30% |
| Macro Recall | 87.69% |
| File Size | ~8.4 MB |

#### Usage

```sh
echo "これはテストです。" | litsea segment --pos -l japanese resources/japanese_pos.model
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

The `resources/` directory also contains:

- **bocchan.txt** -- Sample Japanese corpus from the novel "Botchan" by Natsume Soseki (~307 KB). Used for benchmarking.
