# litsea-nodejs

[Litsea](https://github.com/mosuka/litsea) の Node.js バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。

[English README](README.md)

## インストール

```sh
npm install litsea
```

Linux・macOS・Windows の x64 / arm64 向けにビルド済みバイナリを配布します。Node.js 20 以降が必要です。

## モデルは同梱されません

このパッケージにはコードのみが含まれます。事前学習済みモデルは [Litsea リポジトリ](https://github.com/mosuka/litsea/tree/main/models)から取得し、パスを指定して読み込んでください。

| モデル | 用途 | サイズ |
|-------|------|-------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | 分割 | 84KB〜2.0MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | 分割 + POS | 3.0〜8.0MB |

どちらの種別かを指定する必要はありません。モデルファイル自身が種別を持っており、読み込んだモデルで何ができるかは `hasPos` が示します。

## 使い方

### 分割

```js
import { Segmenter } from 'litsea'

const seg = Segmenter.open('japanese', 'models/japanese.model')

seg.segment('これはテストです。')
// [ 'これ', 'は', 'テスト', 'です', '。' ]

seg.segmentBatch(['これはテストです。', '東京都から神奈川県へ引っ越した'])
// [ [ 'これ', 'は', 'テスト', 'です', '。' ],
//   [ '東京', '都', 'から', '神奈川', '県', 'へ', '引っ越し', 'た' ] ]
```

言語名と ISO 639-1 コードのどちらも使えます（`Segmenter.open('ja', …)`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```js
Segmenter.open('ko', 'models/korean.model').segment('안녕하세요 반갑습니다')
// [ '안녕하세요', ' ', '반갑습니다' ]
```

### POS タグ付け

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

分割専用モデルに対して `segmentWithPos` を呼ぶと、`code === 'pos_unavailable'` のエラーが送出されます。

### その他のモデル読み込み方法

```js
Segmenter.fromBytes('korean', readFileSync('korean.model'))
await Segmenter.fromUri('chinese', 'https://example.com/chinese.model')   // イベントループを塞がずにダウンロード
```

### 学習

特徴量抽出と学習は Promise を返し、ワーカースレッドで実行されるため、イベントループは止まりません。

```js
import { Extractor, Trainer } from 'litsea'

await new Extractor('japanese').extract('corpus.txt', 'features.txt')

const metrics = await new Trainer(0.01, 10_000, 'features.txt').train('japanese.model')
console.log(`accuracy: ${metrics.accuracy.toFixed(2)}%`)
```

二段構成（分割 + POS）の学習:

```js
import { Extractor, TwoStageTrainer } from 'litsea'

await new Extractor('japanese').extractTwoStage('corpus_pos.txt', 'features', 'fast')

const metrics = await new TwoStageTrainer(10, 'features').train('japanese_pos.model')
console.log(metrics.stage1.accuracy, metrics.stage2.accuracy)
```

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。再利用可能かどうかは `available` が示し、2 回目の `train()` は `code === 'invalid_argument'` で reject されます。

### 学習のキャンセル

学習はイベントループ外で走るため、JavaScript 側から実行中に停止できます。

```js
import { CancelToken, Trainer } from 'litsea'

const cancel = new CancelToken()
setTimeout(() => cancel.cancel(), 60_000)

const metrics = await new Trainer(0.01, 100_000, 'features.txt').train('japanese.model', cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスで resolve します。バインディングはシグナルハンドラを登録しません。

## エラー

すべてのエラーは `code` を持ちます。

| `err.code` | 発生条件 |
|-----------|---------|
| `invalid_argument` | 未知の言語名、未知の feature set、使用済みトレーナ |
| `model` | ダウンロード失敗、または旧 joint POS モデル |
| `io` | ファイルの読み書き失敗 |
| `parse` | モデルまたは学習データの形式不正 |
| `unsupported` | このビルドでは利用できないスキームや操作 |
| `pos_unavailable` | 分割専用モデルに対する POS タグ付けの要求 |

reject された Promise も、throw されたエラーと同じ `code` を持ちます。

## TypeScript

型定義は napi-rs が生成し、`index.d.ts` として同梱されます。`@types` パッケージは不要です。

## 開発

```sh
make test-litsea-nodejs    # cargo test + napi build + node --test
make build-litsea-nodejs   # リリースビルド
```

`index.js` と `index.d.ts` は `napi build` が生成したものをコミットしています。CI で再生成し、内容が最新であることを検証します。

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
