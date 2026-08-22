# litsea-python

Python binding for [Litsea](https://github.com/mosuka/litsea), a compact word segmentation and POS (Part-of-Speech) tagging library for Japanese, Chinese, Korean, and English.

[日本語のREADME](README_ja.md)

## Installation

```sh
pip install litsea
```

Wheels are built with the stable ABI (abi3) and work on CPython 3.10 and later.

## Models are not bundled

The package ships code only. Download a pre-trained model from the [Litsea repository](https://github.com/mosuka/litsea/tree/main/models) and point the segmenter at it:

| Model | Purpose | Size |
|-------|---------|------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | Segmentation | 84 KB – 2.0 MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | Segmentation + POS | 3.0 – 8.0 MB |

You never have to say which kind you have: the model file identifies itself, and `has_pos` reports what the loaded model can do.

## Usage

### Segmentation

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")

seg.segment("これはテストです。")
# ['これ', 'は', 'テスト', 'です', '。']

seg.segment_batch(["これはテストです。", "東京都から神奈川県へ引っ越した"])
# [['これ', 'は', 'テスト', 'です', '。'],
#  ['東京', '都', 'から', '神奈川', '県', 'へ', '引っ越し', 'た']]
```

For space-delimited languages the whitespace comes back as its own token, so the tokens always reconstruct the input:

```python
Segmenter.open("ko", "models/korean.model").segment("안녕하세요 반갑습니다")
# ['안녕하세요', ' ', '반갑습니다']
```

Language names work anywhere a `Language` does:

```python
Segmenter.open("ja", "models/japanese.model")
Segmenter.open("japanese", "models/japanese.model")
```

### POS tagging

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese_pos.model")
seg.has_pos
# True

for token in seg.segment_with_pos("これはテストです。"):
    print(token.surface, token.pos.name, token.start, token.end)
# これ PRON 0 6
# は ADP 6 9
# テスト NOUN 9 18
# です AUX 18 24
# 。 PUNCT 24 27
```

`start` and `end` are byte offsets into the input, so `text.encode()[token.start:token.end].decode()` gives the surface back. They are exact for both segmentation and POS output, including for space-preserving languages such as Korean and English.

Calling `segment_with_pos` on a segmentation-only model raises `PosUnavailableError`.

### Other model sources

```python
Segmenter.from_bytes(Language.KOREAN, open("korean.model", "rb").read())
Segmenter.from_uri(Language.CHINESE, "https://example.com/chinese.model")
```

### Training

```python
from litsea import CancelToken, Extractor, Language, Trainer

Extractor(Language.JAPANESE).extract("corpus.txt", "features.txt")

metrics = Trainer(0.01, 10_000, "features.txt").train("japanese.model")
print(f"accuracy: {metrics.accuracy:.2f}%")
```

Two-stage (segmentation + POS) training:

```python
from litsea import Extractor, Language, TwoStageTrainer

Extractor(Language.JAPANESE).extract_two_stage("corpus_pos.txt", "features", feature_set="fast")

metrics = TwoStageTrainer(10, "features").train("japanese_pos.model")
print(metrics.stage1.accuracy, metrics.stage2.accuracy)
```

A `TwoStageTrainer` can only be used once — training collapses stage 1 into an AdaBoost model, which consumes the trainer. `available` reports whether it can still run.

### Cancelling a training run

Training releases the GIL, so another thread can stop it:

```python
import threading
from litsea import CancelToken, Trainer

cancel = CancelToken()
trainer = Trainer(0.01, 100_000, "features.txt")

threading.Timer(60.0, cancel.cancel).start()
metrics = trainer.train("japanese.model", cancel=cancel)
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics.

The binding never installs a signal handler, so Ctrl-C handling stays yours.

## Errors

Every exception derives from `LitseaError`, so one `except` clause catches them all.

| Exception | Raised when |
|-----------|-------------|
| `InvalidArgumentError` | Unknown language name, unknown feature set, reused trainer |
| `ModelError` | Download failed, or the file is a legacy joint POS model |
| `IoError` | A file could not be read or written |
| `ParseError` | The model or training data is malformed |
| `UnsupportedError` | The scheme or operation is unavailable in this build |
| `PosUnavailableError` | POS tagging requested from a segmentation-only model |

## Threading

A `Segmenter` is immutable and safe to share between threads. `segment_batch`, `segment_with_pos_batch`, `extract`, and every `train` release the GIL. Single-sentence `segment` and `segment_with_pos` keep it: releasing would require copying the input string, which costs more than segmenting one sentence.

## Development

```sh
make setup-venv            # create the venv and install the dev tools
make test-litsea-python    # cargo test + maturin develop + pytest
make build-litsea-python   # build a release wheel
```

## License

MIT. See [LICENSE](../LICENSE).
