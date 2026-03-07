# CoNLL-U Converter

The `conllu` module provides functionality to convert [CoNLL-U](https://universaldependencies.org/format.html) format files (used by Universal Dependencies) to Litsea corpus format for both word segmentation and POS tagging.

## Overview

Universal Dependencies treebanks are distributed in CoNLL-U format. The `conllu` module parses this format and produces Litsea-compatible corpus files in two modes:

- **Word segmentation mode** (`with_pos: false`): space-separated words, one sentence per line
- **POS tagging mode** (`with_pos: true`): `word/POS word/POS ...` format, one sentence per line

## Functions

### `convert_conllu`

```rust
pub fn convert_conllu(
    input_path: &Path,
    output_path: &Path,
    with_pos: bool,
) -> Result<usize, Box<dyn Error>>
```

Reads a CoNLL-U file and writes a Litsea corpus file. Returns the number of sentences converted.

#### Word segmentation corpus

```rust
use std::path::Path;
use litsea::conllu::convert_conllu;

let input = Path::new("./ja_gsd-ud-train.conllu");
let output = Path::new("./corpus.txt");

// Output: space-separated words
let count = convert_conllu(input, output, false)?;
```

#### POS corpus

```rust
use std::path::Path;
use litsea::conllu::convert_conllu;

let input = Path::new("./ja_gsd-ud-train.conllu");
let output = Path::new("./pos_corpus.txt");

// Output: word/POS format
let count = convert_conllu(input, output, true)?;
```

## Input Format

CoNLL-U is a tab-separated format with 10 columns per token:

```text
1	太郎	太郎	PROPN	_	_	13	nsubj	_	SpaceAfter=No
2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
```

The converter uses columns 1 (ID), 2 (FORM), and 4 (UPOS).

## Output Format

### Word segmentation mode (`with_pos: false`)

```text
太郎 は 花子 が 読ん で いる 本 を 渡し た 。
```

### POS tagging mode (`with_pos: true`)

```text
太郎/PROPN は/ADP 花子/PROPN が/ADP 読ん/VERB で/SCONJ いる/AUX 本/NOUN を/ADP 渡し/VERB た/AUX 。/PUNCT
```

## Handling Special Cases

| Case | ID Pattern | Behavior |
|------|-----------|----------|
| Regular token | `1`, `2`, ... | Included in output |
| Multiword token | `1-2`, `3-4`, ... | Skipped (component tokens are used) |
| Empty node | `1.1`, `2.1`, ... | Skipped |
| Missing UPOS | `_` in column 4 | Skipped |
| Comment line | `# ...` | Skipped |
| Blank line | (empty) | Marks end of sentence |
