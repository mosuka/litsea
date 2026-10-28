# Node.js

`litsea-nodejs` は [napi-rs](https://napi.rs) を用いて Litsea を Node.js 20 以降へ公開するバインディングです。npm では `litsea` として配布し、Linux・macOS・Windows の x64 / arm64 向けにビルド済みバイナリを提供します。

ブラウザ向けには [`litsea-wasm`](../bindings.md) を使ってください。

## インストール

```sh
npm install litsea
```

## モデルの入手

パッケージにモデルは含まれません。[`models/`](https://github.com/mosuka/litsea/tree/main/models) から取得してパスを渡してください（[事前学習済みモデル](../pre-trained-models.md)を参照）。モデルファイル自身が種別を持つため、フラグの指定は不要で、読み込んだモデルで何ができるかは `hasPos` が示します。

## 分割

```js
import { Segmenter } from 'litsea'

const seg = Segmenter.open('japanese', 'models/japanese.model')

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]
```

言語名と ISO 639-1 コードのどちらも使えます（`'ja'` / `'japanese'`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```js
Segmenter.open('ko', 'models/korean.model').segment('안녕하세요 반갑습니다')
// [ '안녕하세요', ' ', '반갑습니다' ]
```

## POS タグ付け

```js
const seg = Segmenter.open('japanese', 'models/japanese_pos.model')

seg.segmentWithPos('これはテストです。')
// [ { surface: 'これ', start: 0, end: 6, pos: 'PRON' },
//   { surface: 'は', start: 6, end: 9, pos: 'ADP' },
//   { surface: 'テスト', start: 9, end: 18, pos: 'NOUN' },
//   { surface: 'です', start: 18, end: 24, pos: 'AUX' },
//   { surface: '。', start: 24, end: 27, pos: 'PUNCT' } ]
```

`start` と `end` は**バイト**オフセットです。JavaScript の文字列インデックスは UTF-16 コードユニットなので、切り出しには `Buffer` を使ってください。

```js
Buffer.from(text).subarray(token.start, token.end).toString()   // === token.surface
```

タグ付けを行わない `segmentTokens` のトークンでは `pos` は `undefined` になります。

## ユーザー辞書

すべてのコンストラクタは、省略可能な末尾の引数を 3 つ受け取ります（[#263](https://github.com/mosuka/litsea/issues/263)、[#264](https://github.com/mosuka/litsea/issues/264)）。`userWords` は、現れた場所で必ず 1 トークンとして出力する語の配列、`userWordsGate` は、モデルの margin が十分小さい場所にだけ語を適用する flip ratio（`0`〜`1`）、`userWordTags` は、さらに登録する語と、`segmentWithPos` がその語に付ける UPOS タグのオブジェクトです。

```js
const seg = Segmenter.open('japanese', 'models/japanese.model', ['雪だるま', '新幹線'])

seg.segment('雪だるまを作る')
// [ '雪だるま', 'を', '作る' ]   （辞書なしでは [ '雪', 'だる', 'ま', 'を', '作る' ]）
```

POS モデルでは、`userWordTags` でタガーが誤る語のタグを直せます:

```js
const tagged = Segmenter.open('japanese', 'models/japanese_pos.model', undefined, undefined, { チャットGPT: 'PROPN' })

tagged.segmentWithPos('チャットGPTに聞く').map((t) => `${t.surface}/${t.pos}`)
// [ 'チャットGPT/PROPN', 'に/ADP', '聞く/VERB' ]   （userWords に入れた場合は 'チャットGPT/NOUN'）
```

登録語には両端に境界が置かれ、内側には置かれません。重なった出現箇所は leftmost-longest で解決し、タグのない 1 文字の語は無視します。POS モデルでは、`userWords` の語は他の語と同じようにタグ付けされます（`japanese_pos.model` では `雪だるま/NOUN`）。`userWordTags` の語と一致するトークンには、モデルの語彙表とタガーより優先して、そのタグが付きます。`segment` はタグを無視します。`userWordTags` の 1 文字の語は境界を変えず、その 1 文字ちょうどのトークンのタグだけを置き換えます（issue #276）。照合は文脈を見ないため、短い断片を語として登録すると、それを含む他の語をすべて壊します。`userWordsGate` を `0` にすると、モデルがすでに同意している場所だけに語を適用します。`0`〜`1` の外の gate と、17 個の UPOS タグ（大文字）以外のタグは、`invalid_argument` コードのエラーを同期的に投げます（`fromUri` でも同様です）。意味論の全体は [Segmenter](../litsea/segmenter.md#ユーザー辞書) を参照してください。

## API

| 呼び出し | 戻り値 |
|---------|-------|
| `Segmenter.open(language, path, userWords?, userWordsGate?, userWordTags?)` | セグメンタ（同期） |
| `Segmenter.fromBytes(language, buffer, userWords?, userWordsGate?, userWordTags?)` | セグメンタ（同期） |
| `Segmenter.fromUri(language, uri, userWords?, userWordsGate?, userWordTags?)` | `Promise<Segmenter>`（イベントループ外でダウンロード） |
| `segment(text)` | `string[]` |
| `segmentBatch(texts)` | `string[][]` |
| `segmentTokens(text)` | バイトオフセット付き `Token[]` |
| `segmentWithPos(text)` | タグとオフセット付き `Token[]` |
| `segmentWithPosBatch(texts)` | `Token[][]` |
| `new Extractor(language).extract(...)` | `Promise<void>` |
| `new Extractor(language).extractTwoStage(...)` | `Promise<void>` |
| `new Trainer(threshold, iterations, features).train(model, cancel?)` | `Promise<BinaryMetrics>` |
| `new PerceptronTrainer(epochs, features).train(model, cancel?)` | `Promise<MulticlassMetrics>` |
| `new TwoStageTrainer(epochs, prefix, dominance?).train(model, cancel?)` | `Promise<TwoStageMetrics>` |

型定義は napi-rs が生成し、`index.d.ts` として同梱されます。

## 非同期設計

モデルのダウンロード・特徴量抽出・学習はいずれも Promise を返し、libuv のスレッドプールで実行されるため、イベントループは止まりません。これがキャンセルを有効にしている理由でもあります。

```js
import { CancelToken, Trainer } from 'litsea'

const cancel = new CancelToken()
setTimeout(() => cancel.cancel(), 60_000)

const metrics = await new Trainer(0.01, 100_000, 'features.txt').train('japanese.model', cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスで resolve します。バインディングはシグナルハンドラを登録しません。

分割自体は同期処理です。Promise のコストの方が処理そのものより大きいためです。

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。状態は `available` が示し、2 回目の `train()` は reject されます。

## エラー

すべてのエラーは、他のバインディングと同じ種別を表す `code` を持ちます。reject された Promise も throw されたエラーと同じ `code` を持ちます。

| `err.code` | 発生条件 |
|-----------|---------|
| `invalid_argument` | 未知の言語名、未知の feature set、使用済みトレーナ、`0`〜`1` の外の `userWordsGate`、UPOS タグでない `userWordTags` の値 |
| `model` | ダウンロード失敗、または旧 joint POS モデル |
| `io` | ファイルの読み書き失敗 |
| `parse` | モデルまたは学習データの形式不正 |
| `unsupported` | このビルドでは利用できないスキームや操作 |
| `pos_unavailable` | 分割専用モデルに対する POS タグ付けの要求 |

`napi::Status` が閉じた enum のため、code は 2 つの経路で JavaScript に渡ります。同期呼び出しは napi の文字列ステータス付きエラーを使い、非同期呼び出しは `Task::reject` で JavaScript の `Error` オブジェクトを再構築してプロパティを reject 後も保持させています。

## 開発

```sh
make test-litsea-nodejs    # cargo test + napi build + node --test
make lint-litsea-nodejs    # clippy
make build-litsea-nodejs   # リリースビルド
```

`index.js` と `index.d.ts` は `napi build` が生成したものをコミットしており、CI で再生成して内容が古い場合は失敗させます。両ファイルの内容は `@napi-rs/cli` のテンプレートで決まるため、`package.json` では CLI を正確なバージョンに固定しています。浮動レンジのままだと、上流の CLI リリースだけでコミット済みのファイルが古くなります（2026 年 9 月の 3.9.1 と 3.10.0 で実際に起きました）。そのため CLI を更新するときは、同じ PR で両ファイルを再生成します。パリティテストは `litsea` CLI をビルドし、その出力とバインディングの出力を突き合わせます。
