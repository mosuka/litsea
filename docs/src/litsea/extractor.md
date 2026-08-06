# Extractor

The `Extractor` struct extracts features from a corpus file for model training.

## Definition

```rust
pub struct Extractor {
    segmenter: Segmenter,
}
```

## Constructor

### `Extractor::new`

```rust
pub fn new(language: Language) -> Self
```

Creates a new extractor for the specified language. Internally creates a `Segmenter` without a pre-trained model.

```rust
use litsea::extractor::Extractor;
use litsea::language::Language;

let mut extractor = Extractor::new(Language::Japanese);
```

## Methods

### `extract`

```rust
pub fn extract(
    &self,
    corpus_path: &Path,
    features_path: &Path,
) -> litsea::Result<()>
```

Reads a corpus file (space-separated words, one sentence per line) and writes the extracted features to the output file.

```rust
use std::path::Path;

extractor.extract(
    Path::new("./corpus.txt"),
    Path::new("./features.txt"),
)?;
```

### Pipeline

```mermaid
flowchart LR
    A["corpus.txt<br/>(space-separated words)"] --> B["Extractor::extract()"]
    B --> C["features.txt<br/>(label + features per position)"]
```

The extractor:

1. Reads each line from the corpus file
2. Calls `Segmenter::add_corpus_with_writer()` to process each line
3. Writes the label and feature set for each character position to the output file

### `extract_with_pos`

```rust
pub fn extract_with_pos(
    &self,
    corpus_path: &Path,
    features_path: &Path,
) -> litsea::Result<()>
```

Reads a POS-tagged corpus (`word/POS word/POS ...`, one sentence per line,
POS tags from the UPOS tagset) and writes POS training features. Each output
line is `label\tfeature1\tfeature2\t...` where the label is a `SegmentLabel`
string (`B-<POS>` for a word-initial character, `O` for a continuation).
Unlike the boundary pipeline, the first character position of each sentence
is also emitted, because `segment_with_pos` predicts there to derive the
first word's POS.

```rust
use std::path::Path;

extractor.extract_with_pos(
    Path::new("./pos_corpus.txt"),
    Path::new("./features_pos.txt"),
)?;
```
