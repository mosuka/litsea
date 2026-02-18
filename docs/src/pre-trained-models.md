# Pre-trained Models

Litsea ships with several pre-trained models in the `resources/` directory.

## Model Catalog

### japanese.model

| Property | Value |
|----------|-------|
| Language | Japanese |
| Training Corpus | Japanese Wikipedia articles |
| Tokenizer | Lindera (UniDic) |
| Accuracy | 94.15% |
| Precision | 95.57% |
| Recall | 94.36% |
| File Size | ~2.9 KB |

### korean.model

| Property | Value |
|----------|-------|
| Language | Korean |
| Training Corpus | Korean Wikipedia articles |
| Tokenizer | Lindera (ko-dic) |
| Accuracy | 85.08% |
| File Size | ~1.8 KB |

### chinese.model

| Property | Value |
|----------|-------|
| Language | Chinese (Simplified & Traditional) |
| Training Corpus | Chinese Wikipedia articles |
| Tokenizer | Lindera (CC-CEDICT) |
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

## Choosing a Model

- For **Japanese**, use `japanese.model` for the best accuracy, or `RWCP.model` for compatibility with the original TinySegmenter
- For **Chinese**, use `chinese.model`
- For **Korean**, use `korean.model`
- For **domain-specific** needs, consider [training your own model](training-guide/preparing-corpus.md) or [retraining](training-guide/retraining-models.md) an existing one

## Sample Data

The `resources/` directory also contains:

- **bocchan.txt** -- Sample Japanese corpus from the novel "Botchan" by Natsume Soseki (~307 KB). Used for benchmarking.
