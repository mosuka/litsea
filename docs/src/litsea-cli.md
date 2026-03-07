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
| [`convert-conllu`](litsea-cli/convert-conllu.md) | Convert CoNLL-U (Universal Dependencies) files to Litsea corpus format |
| [`split-sentences`](litsea-cli/split-sentences.md) | Split text into sentences using Unicode UAX #29 |

## Global Options

| Option | Description |
|--------|------------|
| `-h`, `--help` | Show help information |
| `-V`, `--version` | Show version number |

## Typical Workflow

```mermaid
flowchart LR
    A["1. Prepare corpus"] --> B["2. litsea extract"]
    B --> C["3. litsea train"]
    C --> D["4. litsea segment"]
```

1. Prepare a corpus from a UD Treebank: `litsea convert-conllu ud_data.conllu corpus.txt`
2. Extract features: `litsea extract -l japanese corpus.txt features.txt`
3. Train a model: `litsea train -t 0.005 -i 1000 features.txt model.model`
4. Segment text: `echo "text" | litsea segment -l japanese model.model`

### POS Tagging Workflow

```mermaid
flowchart LR
    A["1. UD Treebank\n(CoNLL-U)"] --> B["2. litsea convert-conllu"]
    B --> C["3. litsea extract --pos"]
    C --> D["4. litsea train --pos"]
    D --> E["5. litsea segment --pos"]
```

1. Obtain a Universal Dependencies treebank in CoNLL-U format
2. Convert to Litsea POS corpus: `litsea convert-conllu --pos ud_data.conllu corpus_pos.txt`
3. Extract POS features: `litsea extract --pos -l japanese corpus_pos.txt features_pos.txt`
4. Train a POS model: `litsea train --pos -e 10 features_pos.txt model_pos.model`
5. Segment with POS tags: `echo "text" | litsea segment --pos -l japanese model_pos.model`
