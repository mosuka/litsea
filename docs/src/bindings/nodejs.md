# Node.js

`litsea-nodejs` exposes Litsea to Node.js 20+ through [napi-rs](https://napi.rs). It is published to npm as `litsea`, with prebuilt native binaries for Linux, macOS, and Windows on x64 and arm64.

For browsers, use [`litsea-wasm`](../bindings.md) instead.

## Installation

```sh
npm install litsea
```

## Getting a model

The package contains no models. Download one from the [`models/`](https://github.com/mosuka/litsea/tree/main/models) directory and pass its path — see [Pre-trained Models](../pre-trained-models.md). The model file identifies its own kind, so `hasPos` reports what the loaded model can do and no flag is needed.

## Segmentation

```js
import { Segmenter } from 'litsea'

const seg = Segmenter.open('japanese', 'models/japanese.model')

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]
```

The language name and its ISO 639-1 code are interchangeable (`'ja'`, `'japanese'`).

For space-delimited languages the whitespace is returned as its own token, so the tokens always reconstruct the input:

```js
Segmenter.open('ko', 'models/korean.model').segment('안녕하세요 반갑습니다')
// [ '안녕하세요', ' ', '반갑습니다' ]
```

## POS tagging

```js
const seg = Segmenter.open('japanese', 'models/japanese_pos.model')

seg.segmentWithPos('これはテストです。')
// [ { surface: 'これ', start: 0, end: 6, pos: 'PRON' },
//   { surface: 'は', start: 6, end: 9, pos: 'ADP' },
//   { surface: 'テスト', start: 9, end: 18, pos: 'NOUN' },
//   { surface: 'です', start: 18, end: 24, pos: 'AUX' },
//   { surface: '。', start: 24, end: 27, pos: 'PUNCT' } ]
```

`start` and `end` are **byte** offsets. JavaScript string indices are UTF-16 code units, so slice with a `Buffer`:

```js
Buffer.from(text).subarray(token.start, token.end).toString()   // === token.surface
```

`pos` is `undefined` on tokens from `segmentTokens`, which does no tagging.

## User dictionary

Every constructor takes three optional trailing arguments ([#263](https://github.com/mosuka/litsea/issues/263), [#264](https://github.com/mosuka/litsea/issues/264)): `userWords`, an array of words to emit as single tokens wherever they occur; `userWordsGate`, a flip ratio in `0`–`1` that applies a word only where the model's margin against it is small enough; and `userWordTags`, an object mapping further words to the UPOS tag that `segmentWithPos` gives them.

```js
const seg = Segmenter.open('japanese', 'models/japanese.model', ['雪だるま', '新幹線'])

seg.segment('雪だるまを作る')
// [ '雪だるま', 'を', '作る' ]   (without the dictionary: [ '雪', 'だる', 'ま', 'を', '作る' ])
```

With a POS model, `userWordTags` fixes the tag of a word the tagger gets wrong:

```js
const tagged = Segmenter.open('japanese', 'models/japanese_pos.model', undefined, undefined, { チャットGPT: 'PROPN' })

tagged.segmentWithPos('チャットGPTに聞く').map((t) => `${t.surface}/${t.pos}`)
// [ 'チャットGPT/PROPN', 'に/ADP', '聞く/VERB' ]   (in userWords instead: 'チャットGPT/NOUN')
```

A registered word gets a boundary at each edge and none inside, overlapping occurrences are resolved leftmost-longest, and an untagged word of one character is ignored. With a POS model a word in `userWords` is tagged like any other word (`雪だるま/NOUN` with `japanese_pos.model`), while a token that is a word in `userWordTags` gets its tag, ahead of the model's lexicon and its tagger; `segment` ignores the tags. A word of one character in `userWordTags` changes no boundary and retags only the tokens that are exactly that character (issue #276). Matching is context-free, so a short fragment registered as a word breaks every other word it occurs in; a `userWordsGate` of `0` applies a word only where the model already agrees with it. A gate outside `0`–`1`, or a tag that is not one of the 17 UPOS tags (upper case), throws with the `invalid_argument` code — synchronously, also from `fromUri`. See [Segmenter](../litsea/segmenter.md#user-dictionary) for the full semantics.

## API

| Call | Returns |
|------|---------|
| `Segmenter.open(language, path, userWords?, userWordsGate?, userWordTags?)` | A segmenter (synchronous) |
| `Segmenter.fromBytes(language, buffer, userWords?, userWordsGate?, userWordTags?)` | A segmenter (synchronous) |
| `Segmenter.fromUri(language, uri, userWords?, userWordsGate?, userWordTags?)` | `Promise<Segmenter>` — downloads off the event loop |
| `segment(text)` | `string[]` |
| `segmentBatch(texts)` | `string[][]` |
| `segmentTokens(text)` | `Token[]` with byte offsets |
| `segmentWithPos(text)` | `Token[]` with tags and offsets |
| `segmentWithPosBatch(texts)` | `Token[][]` |
| `language`, `hasPos` (getters) | The language name the segmenter was built for; whether the model supports POS tagging |
| `new Extractor(language).extract(corpusPath, featuresPath, tsv?, tagFree?)` | `Promise<void>` |
| `new Extractor(language).extractTwoStage(corpusPath, outputPrefix, featureSet?, tsv?)` | `Promise<void>`; an unknown `featureSet` throws synchronously |
| `new Trainer(threshold, iterations, features).train(model, cancel?)` | `Promise<JsBinaryMetrics>` |
| `new PerceptronTrainer(epochs, features).train(model, cancel?)` | `Promise<JsMulticlassMetrics>` |
| `new TwoStageTrainer(epochs, prefix, dominance?).train(model, cancel?)` | `Promise<JsTwoStageMetrics>` |
| `new CancelToken()`: `cancel()`, `reset()`, `cancelled` (getter) | A flag to pass as `train(model, cancel)` |
| `supportedLanguages()` | `string[]`, the canonical language names |
| `version()` | `string`, the `litsea` version |

`featureSet` is `'full'`, `'balanced'`, or `'fast'` (the default), and `tsv` set to `true` reads the corpus in the space-preserving TSV format.

Type definitions are generated by napi-rs and shipped as `index.d.ts`; the metric objects are declared there as `JsBinaryMetrics`, `JsMulticlassMetrics`, and `JsTwoStageMetrics`.

## Asynchronous by design

Downloading a model, extracting features, and training all return promises and run on libuv's threadpool, so the event loop keeps turning. That is what makes cancellation useful:

```js
import { CancelToken, Trainer } from 'litsea'

const cancel = new CancelToken()
setTimeout(() => cancel.cancel(), 60_000)

const metrics = await new Trainer(0.01, 100_000, 'features.txt').train('japanese.model', cancel)
```

Cancelling is **not** an error: training stops at its next check point, still writes the partially trained model, and resolves with its metrics. The binding never installs a signal handler.

Segmentation itself is synchronous: it is fast enough that a promise would cost more than the work.

A `TwoStageTrainer` can only be used once — training collapses stage 1 into an AdaBoost model, which consumes it. `available` reports the state, and a second `train()` rejects.

## Errors

Every error carries a `code`, matching the error kinds the other bindings expose. Rejected promises carry the same codes as thrown errors.

| `err.code` | Raised when |
|-----------|-------------|
| `invalid_argument` | Unknown language name, unknown feature set, reused trainer, `userWordsGate` outside `0`–`1`, a `userWordTags` value that is not a UPOS tag, `dominance` outside `(0.5, 1.0]`, unknown model URI scheme |
| `model` | Download failed, or the file is a legacy joint POS model (no longer supported; retrain with `litsea train --pos`) |
| `io` | A file could not be read or written |
| `parse` | The model or training data is malformed |
| `unsupported` | The scheme or operation is unavailable in this build |
| `pos_unavailable` | POS tagging requested from a segmentation-only model |
| `runtime` | Any other failure |

Because `napi::Status` is a closed enum, the code reaches JavaScript two ways: synchronous calls use napi's string-status error, and asynchronous ones rebuild the JavaScript `Error` object in `Task::reject` so the property survives the rejection.

## Development

```sh
make test-litsea-nodejs    # cargo test + napi build + node --test
make lint-litsea-nodejs    # clippy
make build-litsea-nodejs   # release build
```

`index.js` and `index.d.ts` are generated by `napi build` and committed; CI rebuilds them and fails if the committed copies are stale. Their content comes from the templates of `@napi-rs/cli`, so `package.json` pins the CLI to an exact version: with a floating range, an upstream CLI release alone would make the committed copies stale, as 3.9.1 and 3.10.0 did in September 2026. Bumping the CLI therefore means regenerating both files in the same PR. The parity tests build the `litsea` CLI and compare the binding's output against it.
