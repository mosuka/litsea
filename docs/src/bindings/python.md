# Python

`litsea-python` exposes Litsea to Python 3.10+ through [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs). It is published to PyPI as `litsea`.

## Installation

```sh
pip install litsea
```

Wheels are built against the stable ABI (`abi3-py310`), so one wheel per platform covers every supported Python version.

## Getting a model

The package contains no models. Download one from the [`models/`](https://github.com/mosuka/litsea/tree/main/models) directory and pass its path — see [Pre-trained Models](../pre-trained-models.md).

There is no flag to say what kind of model you have: the file identifies itself, and `has_pos` reports what the loaded model can do.

## Segmentation

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")

seg.segment("これはテストです。")
# ['これ', 'は', 'テスト', 'です', '。']
```

A language name works anywhere a `Language` does — `Segmenter.open("ja", ...)` and `Segmenter.open("japanese", ...)` are equivalent.

For space-delimited languages the whitespace is returned as its own token, so the tokens always reconstruct the input:

```python
Segmenter.open("ko", "models/korean.model").segment("안녕하세요 반갑습니다")
# ['안녕하세요', ' ', '반갑습니다']
```

## POS tagging

```python
seg = Segmenter.open(Language.JAPANESE, "models/japanese_pos.model")

for token in seg.segment_with_pos("これはテストです。"):
    print(token.surface, token.pos.name, token.start, token.end)
# これ PRON 0 6
# は ADP 6 9
# テスト NOUN 9 18
# です AUX 18 24
# 。 PUNCT 24 27
```

`start` and `end` are byte offsets into the input, so `text.encode()[token.start:token.end].decode()` returns the surface. Calling `segment_with_pos` on a segmentation-only model raises `PosUnavailableError`.

## User dictionary

Every constructor accepts three keyword-only arguments ([#263](https://github.com/mosuka/litsea/issues/263), [#264](https://github.com/mosuka/litsea/issues/264)): `user_words`, a sequence of words to emit as single tokens wherever they occur; `user_words_gate`, a flip ratio in `0.0`–`1.0` that applies a word only where the model's margin against it is small enough; and `user_word_tags`, a dict mapping further words to the UPOS tag that `segment_with_pos` gives them.

```python
seg = Segmenter.open("ja", "models/japanese.model", user_words=["雪だるま", "新幹線"])

seg.segment("雪だるまを作る")
# ['雪だるま', 'を', '作る']   (without the dictionary: ['雪', 'だる', 'ま', 'を', '作る'])
```

With a POS model, `user_word_tags` fixes the tag of a word the tagger gets wrong:

```python
seg = Segmenter.open("ja", "models/japanese_pos.model", user_word_tags={"チャットGPT": "PROPN"})

[f"{t.surface}/{t.pos.name}" for t in seg.segment_with_pos("チャットGPTに聞く")]
# ['チャットGPT/PROPN', 'に/ADP', '聞く/VERB']   (in user_words instead: 'チャットGPT/NOUN')
```

A registered word gets a boundary at each edge and none inside, overlapping occurrences are resolved leftmost-longest, and an untagged word of one character is ignored. With a POS model a word in `user_words` is tagged like any other word (`雪だるま/NOUN` with `japanese_pos.model`), while a token that is a word in `user_word_tags` gets its tag, ahead of the model's lexicon and its tagger; `segment` ignores the tags. A word of one character in `user_word_tags` changes no boundary and retags only the tokens that are exactly that character (issue #276). Matching is context-free, so a short fragment registered as a word breaks every other word it occurs in; `user_words_gate=0.0` applies a word only where the model already agrees with it. A gate outside `0.0`–`1.0`, or a tag that is not one of the 17 UPOS tags (upper case), raises `InvalidArgumentError`. See [Segmenter](../litsea/segmenter.md#user-dictionary) for the full semantics.

## API

| Call | Returns |
|------|---------|
| `Segmenter.open(language, path, *, user_words=None, user_words_gate=None, user_word_tags=None)` | A segmenter loaded from a file |
| `Segmenter.from_bytes(language, data, *, user_words=None, user_words_gate=None, user_word_tags=None)` | A segmenter loaded from bytes |
| `Segmenter.from_uri(language, uri, *, user_words=None, user_words_gate=None, user_word_tags=None)` | A segmenter loaded from a path, `file://`, or `http(s)://` URL |
| `segment(text)` | `list[str]` |
| `segment_batch(texts)` | `list[list[str]]` |
| `segment_tokens(text)` | `list[Token]` with byte offsets |
| `segment_with_pos(text)` | `list[Token]` with tags and offsets |
| `segment_with_pos_batch(texts)` | `list[list[Token]]` |
| `language`, `has_pos` (properties) | The `Language` the segmenter was built for; whether the model supports POS tagging |
| `Extractor(language).extract(corpus_path, features_path, *, tsv=False, tag_free=False)` | Writes a features file |
| `Extractor(language).extract_two_stage(corpus_path, output_prefix, *, feature_set="fast", tsv=False)` | Writes `.stage1` / `.stage2` / `.lexicon` |
| `Trainer(threshold, iterations, features).train(model, *, cancel=None)` | `BinaryMetrics` |
| `PerceptronTrainer(epochs, features).train(model, *, cancel=None)` | `MulticlassMetrics` |
| `TwoStageTrainer(epochs, prefix, *, dominance=0.99).train(model, *, cancel=None)` | `TwoStageMetrics` |
| `Trainer(...).load_model(uri)`, `PerceptronTrainer(...).load_model(uri)` | `None`; loads an existing model (path, `file://`, or `http(s)://` URL) to continue training from it |
| `CancelToken()`: `cancel()`, `reset()`, `cancelled` (property) | A flag to pass as `train(..., cancel=...)` |
| `Language.parse(name)` | A `Language`; raises `InvalidArgumentError` for an unknown name |
| `version()`, `__version__` | The `litsea` version string |

`feature_set` is `"full"`, `"balanced"`, or `"fast"`, and `tsv=True` reads the corpus in the space-preserving TSV format.

`Language` and `Upos` are PyO3 classes, not `enum.Enum` subclasses: their members are class attributes, so iterate them with `Language.all()` and `Upos.all()` rather than `for x in Language`. Their `name` property gives the canonical name (`"japanese"`, `"NOUN"`).

## Training

```python
from litsea import Extractor, Language, Trainer

Extractor(Language.JAPANESE).extract("corpus.txt", "features.txt")
metrics = Trainer(0.01, 10_000, "features.txt").train("japanese.model")
print(f"accuracy: {metrics.accuracy:.2f}%")
```

A `TwoStageTrainer` can only run once — training collapses stage 1 into an AdaBoost model, which consumes the trainer. `available` reports whether it can still be used, and a second `train()` raises `InvalidArgumentError`.

### Cancelling

Training releases the GIL, so another thread can stop it:

```python
import threading
from litsea import CancelToken, Trainer

cancel = CancelToken()
threading.Timer(60.0, cancel.cancel).start()
metrics = Trainer(0.01, 100_000, "features.txt").train("japanese.model", cancel=cancel)
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and returns its metrics. The binding never installs a signal handler, so Ctrl-C handling remains the application's.

## Errors

Every exception derives from `LitseaError`.

| Exception | Raised when |
|-----------|-------------|
| `InvalidArgumentError` | Unknown language name, unknown feature set, reused trainer, `user_words_gate` outside `0.0`–`1.0`, a `user_word_tags` value that is not a UPOS tag, `dominance` outside `(0.5, 1.0]`, unknown model URI scheme |
| `ModelError` | Download failed, or the file is a legacy joint POS model (no longer supported; retrain with `litsea train --pos`) |
| `IoError` | A file could not be read or written |
| `ParseError` | The model or training data is malformed |
| `UnsupportedError` | The scheme or operation is unavailable in this build |
| `PosUnavailableError` | POS tagging requested from a segmentation-only model |
| `LitseaError` | Any other failure: the base class itself is raised for errors that fit no subclass |

## Threading and the GIL

A `Segmenter` is immutable and safe to share between threads. `Segmenter.open`, `Segmenter.from_uri`, `segment_batch`, `segment_with_pos_batch`, `extract`, `extract_two_stage`, the trainer constructors, `load_model`, and every `train` release the GIL.

Single-sentence `segment` and `segment_with_pos` keep it. Releasing the GIL requires owning the input string (PyO3's `Ungil` bound forbids touching Python-owned memory with the GIL released), and that copy costs more than segmenting one sentence. Use the batch methods for bulk work.

## Development

```sh
make setup-venv            # create the venv and install the dev tools
make test-litsea-python    # cargo test + maturin develop + pytest
make lint-litsea-python    # clippy + ruff
make build-litsea-python   # build a release wheel into litsea-python/dist
```

The parity tests build the `litsea` CLI and compare the binding's output against it, so the reference implementation — not a hardcoded expectation — decides what is correct.
