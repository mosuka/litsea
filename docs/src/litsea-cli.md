# CLI Reference Overview

The `litsea` CLI provides commands for word segmentation, model training, and text processing.

## Usage

```sh
litsea <COMMAND> [OPTIONS] [ARGS]
```

## Commands

| Command | Description |
|---------|------------|
| [`extract`](litsea-cli/extract.md) | Extract features from a corpus for training |
| [`train`](litsea-cli/train.md) | Train a word segmentation model |
| [`segment`](litsea-cli/segment.md) | Segment text into words using a trained model |

## Global Options

| Option | Description |
|--------|------------|
| `-h`, `--help` | Show help information |
| `-V`, `--version` | Show version number |

## Typical Workflow

### AdaBoost Workflow (Word Segmentation Only)

```mermaid
flowchart LR
    A["1. scripts/download_udtreebank.sh"] --> B["2. scripts/corpus_udtreebank.sh"]
    B --> C["3. litsea extract"]
    C --> D["4. litsea train"]
    D --> E["5. litsea segment"]
```

1. Download a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)`
2. Convert to corpus format: `bash scripts/corpus_udtreebank.sh "$conllu_file" corpus.txt`
3. Extract features: `litsea extract -l japanese corpus.txt features.txt`
4. Train a model: `litsea train -t 0.005 -i 1000 features.txt model.model`
5. Segment text: `echo "text" | litsea segment -l japanese model.model`

### POS Workflow (Word Segmentation with POS Tagging)

```mermaid
flowchart LR
    A["1. scripts/download_udtreebank.sh"] --> B["2. scripts/corpus_udtreebank.sh -p"]
    B --> C["3. litsea extract --pos"]
    C --> D["4. litsea train --pos"]
    D --> E["5. litsea segment --pos"]
```

1. Download a UD Treebank: `conllu_file=$(bash scripts/download_udtreebank.sh -l ja -o /tmp)`
2. Convert to POS corpus format: `bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt`
3. Extract POS features: `litsea extract --pos -l japanese pos_corpus.txt features_pos.txt`
4. Train a POS model: `litsea train --pos --num-epochs 10 features_pos.txt model_pos.model`
5. Segment with POS tags: `echo "text" | litsea segment --pos -l japanese model_pos.model`
