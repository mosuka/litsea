# WebAssembly

`litsea-wasm` runs Litsea in browsers, Deno, and bundlers through [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/). It is published to npm as `litsea-wasm`.

For Node.js, use the native binding [`litsea`](nodejs.md) instead: it is faster and can train models.

## Installation

```sh
npm install litsea-wasm
```

The module is **178 KB (82 KB gzipped)**, measured on a release `wasm-pack build`. Models are downloaded separately.

## Usage

```js
import init, { Segmenter } from 'litsea-wasm'

await init()

const bytes = new Uint8Array(await (await fetch('/models/japanese.model')).arrayBuffer())
const seg = Segmenter.fromBytes('japanese', bytes)

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]

seg.free()
```

The language name and its ISO 639-1 code are interchangeable. The model file identifies its own kind, so `hasPos` reports what was loaded.

## POS tagging

```js
const posBytes = new Uint8Array(await (await fetch('/models/japanese_pos.model')).arrayBuffer())
const seg = Segmenter.fromBytes('japanese', posBytes)

seg.segmentWithPos('これはテストです。').map(String)
// [ 'これ/PRON [0..6]', ... ]
```

`start` and `end` are **byte** offsets into the UTF-8 encoding, and JavaScript string indices are UTF-16 code units, so slice with `TextEncoder` / `TextDecoder`:

```js
const bytes = new TextEncoder().encode(text)
new TextDecoder().decode(bytes.subarray(token.start, token.end))   // === token.surface
```

## User dictionary

`fromBytes` takes three optional trailing arguments ([#263](https://github.com/mosuka/litsea/issues/263), [#264](https://github.com/mosuka/litsea/issues/264)), named as in the Node.js binding: `userWords`, an array of words to emit as single tokens wherever they occur; `userWordsGate`, a flip ratio in `0`–`1` that applies a word only where the model's margin against it is small enough; and `userWordTags`, an object mapping further words to the UPOS tag that `segmentWithPos` gives them.

```js
const seg = Segmenter.fromBytes('japanese', bytes, ['雪だるま', '新幹線'])

seg.segment('雪だるまを作る')
// [ '雪だるま', 'を', '作る' ]   (without the dictionary: [ '雪', 'だる', 'ま', 'を', '作る' ])
```

With a POS model, `userWordTags` fixes the tag of a word the tagger gets wrong:

```js
const posBytes = new Uint8Array(await (await fetch('/models/japanese_pos.model')).arrayBuffer())
const tagged = Segmenter.fromBytes('japanese', posBytes, undefined, undefined, { チャットGPT: 'PROPN' })

tagged.segmentWithPos('チャットGPTに聞く').map((t) => `${t.surface}/${t.pos}`)
// [ 'チャットGPT/PROPN', 'に/ADP', '聞く/VERB' ]   (in userWords instead: 'チャットGPT/NOUN')
```

A registered word gets a boundary at each edge and none inside, overlapping occurrences are resolved leftmost-longest, and an untagged word of one character is ignored. With a POS model a word in `userWords` is tagged like any other word, while a token that is a word in `userWordTags` gets its tag, ahead of the model's lexicon and its tagger; `segment` ignores the tags. A word of one character in `userWordTags` changes no boundary and retags only the tokens that are exactly that character (issue #276). Matching is context-free, so a short fragment registered as a word breaks every other word it occurs in; a `userWordsGate` of `0` applies a word only where the model already agrees with it. A gate outside `0`–`1`, or a tag that is not one of the 17 UPOS tags (upper case), throws with the `invalid_argument` code. See [Segmenter](../litsea/segmenter.md#user-dictionary) for the full semantics.

## API

| Call | Returns |
|------|---------|
| `Segmenter.fromBytes(language, data, userWords?, userWordsGate?, userWordTags?)` | A segmenter |
| `language` (getter) | `string`, the canonical language name |
| `hasPos` (getter) | `boolean` |
| `segment(text)` | `string[]` |
| `segmentBatch(texts)` | `string[][]` |
| `segmentTokens(text)` | `Token[]` with byte offsets (`pos` is `undefined`) |
| `segmentWithPos(text)` | `Token[]` with tags and offsets |
| `segmentWithPosBatch(texts)` | `Token[][]` |
| `free()` | Releases the segmenter (see Memory below) |
| `version()` | `string`, the `litsea` version |
| `supportedLanguages()` | `string[]` |

A `Token` exposes `surface`, `pos`, `start` and `end` as getters, and its `toString()` returns `surface/POS [start..end]`, for example `テスト/NOUN [9..18]`.

## What the host removes

This is the most constrained of the five bindings, and each gap was measured rather than assumed.

| Missing | Why |
|---------|-----|
| `fromUri` | `cargo check --target wasm32-unknown-unknown --features remote_model` fails: reqwest's wasm backend has no `connect_timeout`, which `litsea::model_io` sets. The page fetches the model instead — which also keeps caching, CORS, and progress under its control. |
| Training | A deliberate scope decision, not a technical limit — see below. |
| `CancelToken` | With no training there is nothing to cancel. |

### Why there is no training

`litsea` gained filesystem-free extract/train APIs in [#218](https://github.com/mosuka/litsea/issues/218), and they compile for `wasm32-unknown-unknown`, so this binding *could* expose training. It does not, for two reasons ([#221](https://github.com/mosuka/litsea/issues/221)):

- **A browser is not where training belongs.** A tab would hold the corpus, the features extracted from it (much larger than the corpus), and the model at once. Deciding the API shape would have required measuring that first, and the measurement could well have concluded it is impractical at any useful corpus size.
- **The reference implementation does not either.** `lindera-python`, `lindera-nodejs`, `lindera-php`, and `lindera-ruby` all ship a trainer; `lindera-wasm` ships none. Litsea's bindings match that shape.

Train with the CLI or one of the native bindings, and load the resulting model here.

## Memory

`Segmenter` holds the compiled model, which is several megabytes for a POS model, and WebAssembly objects are **not** garbage collected. Call `free()` when one is no longer needed.

## Caching models

Models are 86 KB – 8.3 MB and cross the network once per visitor — the one cost this binding has that the native ones do not. An optional helper lives in the repository as `litsea-wasm/js/cache.js` (with `cache.d.ts`). The npm package does not include it: `make build-litsea-wasm` copies the two files into `pkg/`, but the `files` list of the `package.json` that wasm-pack generates leaves them out. Copy them into your project and import them from there:

```js
import { fetchModel, clearModelCache } from './cache.js'

const bytes = await fetchModel('/models/japanese.model')
```

It stores fetched models in Cache Storage keyed by URL, and falls back to a plain fetch when Cache Storage is unavailable (an insecure context), so callers do not branch. It is plain JavaScript outside the wasm module, so a page that does not use it pays nothing.

## Errors

Every error carries a `code`, matching the Node.js binding so the two JavaScript bindings agree.

| `err.code` | Raised when |
|-----------|-------------|
| `invalid_argument` | Unknown language name, `userWordsGate` outside `0`–`1`, a `userWordTags` value that is not a UPOS tag (or not a string) |
| `model` | The file is a legacy joint POS model |
| `parse` | The model is malformed or not UTF-8 |
| `pos_unavailable` | POS tagging requested from a segmentation-only model |

## Development

```sh
make test-litsea-wasm    # cargo test + headless browser tests
make lint-litsea-wasm    # clippy on wasm32
make build-litsea-wasm   # wasm-pack build --target web
```

The browser tests cannot spawn a process, so `tests/generate_fixtures.sh` runs the `litsea` CLI first and writes its output next to the test; the test asserts equality against it. The reference implementation still decides what is correct, as in every other binding.

Override the browser with `make test-litsea-wasm WASM_BROWSER=chrome`. If every test passes and the run then fails with `PermissionDenied`, the geckodriver on `PATH` is snap-confined — the tests themselves ran.
