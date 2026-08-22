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
seg.segmentWithPos('これはテストです。')
// [ Token { surface: 'これ', pos: 'PRON', start: 0, end: 6 }, ... ]
```

`start` and `end` are **byte** offsets into the UTF-8 encoding, and JavaScript string indices are UTF-16 code units, so slice with `TextEncoder` / `TextDecoder`:

```js
const bytes = new TextEncoder().encode(text)
new TextDecoder().decode(bytes.subarray(token.start, token.end))   // === token.surface
```

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

Models are 84 KB – 8 MB and cross the network once per visitor — the one cost this binding has that the native ones do not. The package ships an optional helper:

```js
import { fetchModel, clearModelCache } from 'litsea-wasm/js/cache.js'

const bytes = await fetchModel('/models/japanese.model')
```

It stores fetched models in Cache Storage keyed by URL, and falls back to a plain fetch when Cache Storage is unavailable (an insecure context), so callers do not branch. It is plain JavaScript outside the wasm module, so a page that does not use it pays nothing.

## Errors

Every error carries a `code`, matching the Node.js binding so the two JavaScript bindings agree.

| `err.code` | Raised when |
|-----------|-------------|
| `invalid_argument` | Unknown language name |
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
