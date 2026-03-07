# CoNLL-U Converter

The `conllu` module provides functionality to convert [CoNLL-U](https://universaldependencies.org/format.html) format files (used by Universal Dependencies) to Litsea's POS corpus format.

## Overview

Universal Dependencies treebanks are distributed in CoNLL-U format. The `conllu` module parses this format and produces Litsea-compatible POS corpus files where each sentence is represented as `word/POS word/POS ...` on a single line.

## Functions

### `convert_conllu`

```rust
pub fn convert_conllu(input: &Path, output: &Path) -> io::Result<()>
```

Reads a CoNLL-U file and writes a Litsea POS corpus file.

```rust
use std::path::Path;
use litsea::conllu::convert_conllu;

convert_conllu(
    Path::new("./ja_gsd-ud-train.conllu"),
    Path::new("./corpus_pos.txt"),
)?;
```

## Input Format

CoNLL-U is a tab-separated format with 10 columns per token:

```text
1	太郎	太郎	PROPN	_	_	13	nsubj	_	SpaceAfter=No
2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
```

The converter uses columns 1 (ID), 2 (FORM), and 4 (UPOS).

## Output Format

```text
太郎/PROPN は/ADP 花子/PROPN が/ADP 読ん/VERB で/SCONJ いる/AUX 本/NOUN を/ADP 渡し/VERB た/AUX 。/PUNCT
```

## Handling Special Cases

| Case | ID Pattern | Behavior |
|------|-----------|----------|
| Regular token | `1`, `2`, ... | Included in output |
| Multiword token | `1-2`, `3-4`, ... | Skipped (component tokens are used) |
| Empty node | `1.1`, `2.1`, ... | Skipped |
| Missing UPOS | `_` in column 4 | Assigned the `X` tag |
| Comment line | `# ...` | Skipped |
| Blank line | (empty) | Marks end of sentence |
