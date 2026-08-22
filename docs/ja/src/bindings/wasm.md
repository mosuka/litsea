# WebAssembly

`litsea-wasm` は [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) を用いて、ブラウザ・Deno・各種バンドラで Litsea を動かすバインディングです。npm では `litsea-wasm` として配布します。

Node.js ではネイティブバインディングの [`litsea`](nodejs.md) を使ってください。速度が上で、モデルの学習にも対応しています。

## インストール

```sh
npm install litsea-wasm
```

モジュールサイズは **178KB（gzip 82KB）**（リリースビルドでの実測値）です。モデルは別途ダウンロードします。

## 使い方

```js
import init, { Segmenter } from 'litsea-wasm'

await init()

const bytes = new Uint8Array(await (await fetch('/models/japanese.model')).arrayBuffer())
const seg = Segmenter.fromBytes('japanese', bytes)

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]

seg.free()
```

言語名と ISO 639-1 コードのどちらも使えます。モデルファイル自身が種別を持つため、読み込んだモデルで何ができるかは `hasPos` が示します。

## POS タグ付け

```js
seg.segmentWithPos('これはテストです。')
// [ Token { surface: 'これ', pos: 'PRON', start: 0, end: 6 }, ... ]
```

`start` と `end` は UTF-8 での**バイト**オフセットです。JavaScript の文字列インデックスは UTF-16 コードユニットなので、切り出しには `TextEncoder` / `TextDecoder` を使ってください。

```js
const bytes = new TextEncoder().encode(text)
new TextDecoder().decode(bytes.subarray(token.start, token.end))   // === token.surface
```

## ホストの制約で提供しない機能

5 つのバインディングの中で最も制約が強く、いずれも推測ではなく実測に基づいて判断しています。

| 無い機能 | 理由 |
|---------|------|
| `fromUri` | `cargo check --target wasm32-unknown-unknown --features remote_model` が失敗する。reqwest の wasm バックエンドには `connect_timeout` が無く、`litsea::model_io` はそれを設定しているため。代わりにページ側で fetch する（キャッシュ・CORS・進捗の制御もページ側に残る） |
| 学習 | `Extractor` とトレーナはパス前提で、wasm32 にファイルシステムが無い。`litsea-binding-core` も wasm32 では trainer モジュールをコンパイル対象外にしている |
| `CancelToken` | 学習が無い以上、キャンセル対象が存在しない |

ブラウザでの学習には `litsea` 本体に in-memory な extract/train API が必要です。`Segmenter::add_corpus_with_writer` は既にファイルシステム非依存で特徴量を書き出せるため実現可能ですが、コアライブラリ側の作業として [#218](https://github.com/mosuka/litsea/issues/218) で追跡します。

## メモリ

`Segmenter` はコンパイル済みモデル（POS モデルなら数 MB）を保持し、WebAssembly のオブジェクトは **GC されません**。不要になったら `free()` を呼んでください。

## モデルのキャッシュ

モデルは 84KB〜8MB あり、訪問者ごとにネットワークを通ります。これはネイティブ版には無い、このバインディング固有のコストです。補助モジュールを同梱しています。

```js
import { fetchModel, clearModelCache } from 'litsea-wasm/js/cache.js'

const bytes = await fetchModel('/models/japanese.model')
```

URL をキーに Cache Storage へ保存し、利用できない環境（非セキュアコンテキストなど）では通常の fetch にフォールバックするため、呼び出し側での分岐は不要です。wasm モジュール外の素の JavaScript なので、使わないページには一切コストがかかりません。

## エラー

すべてのエラーは Node.js バインディングと同じ `code` を持ち、2 つの JavaScript バインディングで扱いが揃います。

| `err.code` | 発生条件 |
|-----------|---------|
| `invalid_argument` | 未知の言語名 |
| `model` | 旧 joint POS モデル |
| `parse` | モデルの形式不正、または UTF-8 でない |
| `pos_unavailable` | 分割専用モデルに対する POS タグ付けの要求 |

## 開発

```sh
make test-litsea-wasm    # cargo test + ヘッドレスブラウザテスト
make lint-litsea-wasm    # wasm32 での clippy
make build-litsea-wasm   # wasm-pack build --target web
```

ブラウザテストはプロセスを起動できないため、`tests/generate_fixtures.sh` が先に `litsea` CLI を実行して出力を書き出し、テストはそれとの一致を検証します。他のバインディングと同じく、正解は常に参照実装が決めます。

ブラウザは `make test-litsea-wasm WASM_BROWSER=chrome` で切り替えられます。全テストが成功した直後に `PermissionDenied` で失敗する場合、`PATH` 上の geckodriver が snap 制約下にあります（テスト自体は実行済みです）。
