# litsea-wasm

[Litsea](https://github.com/mosuka/litsea) の WebAssembly バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。ブラウザ・Deno・各種バンドラで動作します。

[English README](README.md)

Node.js では [`litsea`](../litsea-nodejs/README_ja.md)（ネイティブバインディング）を使ってください。速度が上で、モデルの学習にも対応しています。

## インストール

```sh
npm install litsea-wasm
```

モジュールサイズは 178KB（gzip 82KB）です。モデルは別途ダウンロードします。

## 使い方

```js
import init, { Segmenter } from 'litsea-wasm'

await init()

// モデルの取得はページ側で行い、バイト列を渡します。
const bytes = new Uint8Array(await (await fetch('/models/japanese.model')).arrayBuffer())
const seg = Segmenter.fromBytes('japanese', bytes)

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]

seg.free()   // WebAssembly のオブジェクトは GC されません
```

**`fromUri` はありません。** reqwest の wasm バックエンドは Litsea が設定するタイムアウトを扱えずビルドできないためです。またページ側で取得する方が設計として素直です（キャッシュ・CORS・進捗・リトライをアプリケーションが制御できます）。

言語名と ISO 639-1 コードのどちらも使えます（`'ja'` / `'japanese'`）。

### POS タグ付け

```js
const posBytes = new Uint8Array(await (await fetch('/models/japanese_pos.model')).arrayBuffer())
const seg = Segmenter.fromBytes('japanese', posBytes)

seg.segmentWithPos('これはテストです。')
// [ Token { surface: 'これ', pos: 'PRON', start: 0, end: 6 }, ... ]
```

`start` と `end` は UTF-8 での**バイト**オフセットです。JavaScript の文字列インデックスは UTF-16 コードユニットなので、切り出しには `TextEncoder` / `TextDecoder` を使ってください。

```js
const bytes = new TextEncoder().encode(text)
new TextDecoder().decode(bytes.subarray(token.start, token.end))   // === token.surface
```

分割専用モデルに対して `segmentWithPos` を呼ぶと `code === 'pos_unavailable'` のエラーが送出されます。モデルファイル自身が種別を持つため、読み込んだモデルで何ができるかは `hasPos` が示します。

### モデルのキャッシュ

モデルは 84KB〜8MB あり、訪問者ごとにネットワークを通ります。Cache Storage に保持する補助モジュールを同梱しています。

```js
import { fetchModel, clearModelCache } from 'litsea-wasm/js/cache.js'

const bytes = await fetchModel('/models/japanese.model')   // 2 回目以降はキャッシュから
```

Cache Storage が使えない環境（非セキュアコンテキストなど）では通常の fetch にフォールバックするため、呼び出し側で分岐する必要はありません。

## メモリ

`Segmenter` はコンパイル済みモデル（POS モデルなら数 MB）を保持し、WebAssembly のオブジェクトは **GC されません**。不要になったら `free()` を呼んでください。

## このバインディングが提供しないもの

| | 理由 |
|---|---|
| `fromUri` なし | reqwest の wasm バックエンドが Litsea のタイムアウト設定でビルドできない。JS 側で fetch する |
| 学習なし | 技術的な制約ではなく方針判断（[#221](https://github.com/mosuka/litsea/issues/221)）。[#218](https://github.com/mosuka/litsea/issues/218) 以降 `litsea` にはファイルシステム非依存の extract/train API があるが、ブラウザのタブがコーパス・特徴量・モデルを同時に抱えることになり、参照実装の `lindera-wasm` も学習を持たない |
| `CancelToken` なし | 学習が無い以上、キャンセル対象が存在しない |

## エラー

すべてのエラーは Node.js バインディングと同じ `code` を持ちます。

| `err.code` | 発生条件 |
|-----------|---------|
| `invalid_argument` | 未知の言語名 |
| `model` | 旧 joint POS モデル |
| `parse` | モデルの形式不正、または UTF-8 でない |
| `pos_unavailable` | 分割専用モデルに対する POS タグ付けの要求 |

## 開発

```sh
make test-litsea-wasm    # cargo test + wasm-pack test --headless
make build-litsea-wasm   # wasm-pack build --target web
```

ブラウザテストは `litsea` CLI が生成したフィクスチャ（`tests/generate_fixtures.sh`）と比較するため、正解は常に参照実装が決めます。

`wasm-pack test` が全テスト成功のあとに `PermissionDenied` で失敗する場合、`PATH` 上の geckodriver が snap 制約下にあります（テスト自体は実行済みです）。snap 以外の geckodriver を入れるか、`--chrome` で実行してください。

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
