# split-sentences

Split text into sentences using Unicode UAX #29 rules.

## Usage

```sh
echo "text" | litsea split-sentences
```

## Arguments

None.

## Options

None (besides `--help` and `--version`).

## Input / Output

- **Input**: Reads from stdin, one paragraph per line. Empty lines are skipped.
- **Output**: Writes to stdout, one sentence per line.

## How It Works

This command uses ICU4X's `SentenceSegmenter` which implements the Unicode Standard Annex #29 (UAX #29) sentence break rules. It is **language-independent** -- no `--language` flag is needed.

## Example

```sh
echo "これはテストです。次の文です。" | litsea split-sentences
```

Output:

```text
これはテストです。
次の文です。
```

Multi-line input:

```sh
echo -e "First sentence. Second sentence.\nThird sentence! Fourth." \
  | litsea split-sentences
```

Output:

```text
First sentence.
Second sentence.
Third sentence!
Fourth.
```

## Use Cases

- Pre-processing text before word segmentation (one sentence per line)
- Splitting large documents into individual sentences for analysis
- Preparing training corpora from raw text
