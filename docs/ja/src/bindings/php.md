# PHP

`litsea-php` は [ext-php-rs](https://github.com/davidcole1340/ext-php-rs) を用いて Litsea を PHP 8.1 以降へ公開するバインディングです。Packagist では `litsea/litsea` として配布します。

## インストール

PHP 拡張は特定の PHP ABI 向けにビルドされた共有オブジェクトです。PyPI や npm と異なりビルド済みパッケージの配布はなく、自分でビルドして有効化します。

```sh
cargo build --release -p litsea-php
php -d extension=/path/to/target/release/liblitsea_php.so your-script.php
```

常時読み込む場合は `php.ini` に `extension=/path/to/liblitsea_php.so` を追記します。ビルドには Rust ツールチェーンと libclang が必要です。

## モデルの入手

拡張にモデルは含まれません。[`models/`](https://github.com/mosuka/litsea/tree/main/models) から取得してパスを渡してください（[事前学習済みモデル](../pre-trained-models.md)を参照）。モデル自身が種別を持つため、フラグの指定は不要で、読み込んだモデルで何ができるかは `hasPos()` が示します。

## 分割

```php
use Litsea\Segmenter;

$seg = Segmenter::open('japanese', 'models/japanese.model');

$seg->segment('これはテストです。');
// ['これ', 'は', 'テスト', 'です', '。']
```

言語名と ISO 639-1 コードのどちらも使えます（`'ja'` / `'japanese'`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```php
Segmenter::open('ko', 'models/korean.model')->segment('안녕하세요 반갑습니다');
// ['안녕하세요', ' ', '반갑습니다']
```

## POS タグ付け

```php
$seg = Segmenter::open('japanese', 'models/japanese_pos.model');

foreach ($seg->segmentWithPos('これはテストです。') as $token) {
    printf("%s\t%s\t[%d:%d]\n", $token->surface, $token->pos, $token->start, $token->end);
}
// これ    PRON    [0:6]
// は      ADP     [6:9]
// テスト  NOUN    [9:18]
// です    AUX     [18:24]
// 。      PUNCT   [24:27]
```

`start` と `end` はバイトオフセットで、PHP の文字列はバイト列です。そのため `substr($text, $token->start, $token->end - $token->start)` がそのまま表層形を返します（JavaScript のようなエンコーディングを意識した切り出しは不要です）。

## API

| 呼び出し | 戻り値 |
|---------|-------|
| `Segmenter::open($language, $path)` | セグメンタ |
| `Segmenter::fromBytes($language, $contents)` | セグメンタ |
| `Segmenter::fromUri($language, $uri)` | セグメンタ（ダウンロードはブロッキング） |
| `segment($text)` | `string[]` |
| `segmentBatch($texts)` | `string[][]` |
| `segmentTokens($text)` | バイトオフセット付き `Token[]` |
| `segmentWithPos($text)` | タグとオフセット付き `Token[]` |
| `segmentWithPosBatch($texts)` | `Token[][]` |
| `(new Extractor($language))->extract(...)` | `void` |
| `(new Extractor($language))->extractTwoStage(...)` | `void` |
| `(new Trainer($threshold, $iterations, $features))->train($model, $cancel?)` | `BinaryMetrics` |
| `(new PerceptronTrainer($epochs, $features))->train($model, $cancel?)` | `MulticlassMetrics` |
| `(new TwoStageTrainer($epochs, $prefix, $dominance?))->train($model, $cancel?)` | `TwoStageMetrics` |

ext-php-rs はメソッドとプロパティを camelCase に変換するため、PHP 側では `segmentWithPos()`・`hasPos()`・`$metrics->numInstances` となります。

## キャンセルは呼び出し前のみ有効

ここが他のバインディングと唯一異なる点です。これはバインディング側の不足ではなく、ホスト言語の性質です。

Python バインディングは GIL を解放し、Node.js バインディングは学習をワーカースレッドで実行するため、どちらも実行中の学習を停止できます。PHP のリクエストはシングルスレッドであり、`pcntl` のシグナルハンドラはブロッキング中のネイティブ呼び出しを中断できないため、**`train()` の実行中に PHP のコードは一切動きません**。したがって `CancelToken` は、呼び出し前にキャンセルした場合のみ効果があります。

```php
$cancel = new Litsea\CancelToken();
$cancel->cancel();

$metrics = (new Litsea\Trainer(0.01, 100000, 'features.txt'))->train('japanese.model', $cancel);
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。

すべての処理がブロッキングであるため、学習は Web リクエストではなく CLI SAPI から実行してください。

## エラー

すべての例外は `Litsea\LitseaException` を継承するため、1 つの `catch` で捕捉できます（Python バインディングと同じ階層です）。

| 例外 | 発生条件 |
|------|---------|
| `Litsea\InvalidArgumentException` | 未知の言語名、未知の feature set、使用済みトレーナ |
| `Litsea\ModelException` | ダウンロード失敗、または旧 joint POS モデル |
| `Litsea\IoException` | ファイルの読み書き失敗 |
| `Litsea\ParseException` | モデルまたは学習データの形式不正 |
| `Litsea\UnsupportedException` | このビルドでは利用できないスキームや操作 |
| `Litsea\PosUnavailableException` | 分割専用モデルに対する POS タグ付けの要求 |

## 開発

```sh
make test-litsea-php    # cargo test + 拡張のビルド + PHPUnit
make lint-litsea-php    # clippy
make build-litsea-php   # リリースビルド
```

パリティテストは `litsea` CLI をビルドし、その出力とバインディングの出力を突き合わせます。
