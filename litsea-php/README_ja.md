# litsea-php

[Litsea](https://github.com/mosuka/litsea) の PHP バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。

[English README](README.md)

## インストール

PHP 拡張は特定の PHP ABI 向けにビルドされた共有オブジェクトであるため、インストール済みバイナリの配布はありません。自分でビルドして `php.ini` で有効化します。

```sh
git clone https://github.com/mosuka/litsea.git
cd litsea
cargo build --release -p litsea-php
```

ビルドしたライブラリを読み込みます。`php.ini` に追記するか:

```ini
extension=/path/to/litsea/target/release/liblitsea_php.so
```

実行ごとに指定します。

```sh
php -d extension=/path/to/liblitsea_php.so your-script.php
```

PHP 8.1 以降と Rust ツールチェーン、およびビルド用の libclang（Debian/Ubuntu では `libclang-dev`）が必要です。

## モデルは同梱されません

事前学習済みモデルは [Litsea リポジトリ](https://github.com/mosuka/litsea/tree/main/models)から取得し、パスを指定して読み込んでください。

| モデル | 用途 | サイズ |
|-------|------|-------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | 分割 | 84KB〜2.0MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | 分割 + POS | 3.0〜8.0MB |

どちらの種別かを指定する必要はありません。モデルファイル自身が種別を持っており、読み込んだモデルで何ができるかは `hasPos()` が示します。

## 使い方

### 分割

```php
use Litsea\Segmenter;

$seg = Segmenter::open('japanese', 'models/japanese.model');

$seg->segment('これはテストです。');
// ['これ', 'は', 'テスト', 'です', '。']

$seg->segmentBatch(['これはテストです。', '東京都から神奈川県へ引っ越した']);
```

言語名と ISO 639-1 コードのどちらも使えます（`'ja'` / `'japanese'`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```php
Segmenter::open('ko', 'models/korean.model')->segment('안녕하세요 반갑습니다');
// ['안녕하세요', ' ', '반갑습니다']
```

### POS タグ付け

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

`start` と `end` はバイトオフセットです。PHP の文字列はバイト列なので、`substr()` にそのまま使えます。

```php
substr($text, $token->start, $token->end - $token->start);   // === $token->surface
```

分割専用モデルに対して `segmentWithPos()` を呼ぶと `Litsea\PosUnavailableException` が送出されます。

### その他のモデル読み込み方法

```php
Segmenter::fromBytes('korean', file_get_contents('korean.model'));
Segmenter::fromUri('chinese', 'https://example.com/chinese.model');
```

### 学習

```php
use Litsea\Extractor;
use Litsea\Trainer;

(new Extractor('japanese'))->extract('corpus.txt', 'features.txt');

$metrics = (new Trainer(0.01, 10000, 'features.txt'))->train('japanese.model');
printf("accuracy: %.2f%%\n", $metrics->accuracy);
```

二段構成（分割 + POS）の学習:

```php
use Litsea\Extractor;
use Litsea\TwoStageTrainer;

(new Extractor('japanese'))->extractTwoStage('corpus_pos.txt', 'features', 'fast');

$metrics = (new TwoStageTrainer(10, 'features'))->train('japanese_pos.model');
printf("%.2f%% / %.2f%%\n", $metrics->stage1Accuracy, $metrics->stage2Accuracy);
```

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。再利用可能かどうかは `isAvailable()` が示し、2 回目の `train()` は例外を送出します。

**学習は CLI SAPI から実行してください。** 学習中はプロセスがブロックされるため、Web リクエストで行うべき処理ではありません。

## PHP ではキャンセルは呼び出し前のみ有効

Python バインディングは GIL を解放し、Node.js バインディングは学習をワーカースレッドへ移すため、どちらも実行中の学習を停止できます。**PHP はどちらもできません。** リクエストはシングルスレッドであり、`pcntl` のシグナルハンドラはブロッキング中のネイティブ呼び出しを中断できないためです。

したがって `Litsea\CancelToken` は、`train()` を呼ぶ**前**にキャンセルした場合のみ効果があります。

```php
use Litsea\CancelToken;
use Litsea\Trainer;

$cancel = new CancelToken();
$cancel->cancel();

// 最初のチェックポイントで停止し、部分学習済みモデルは保存されます。
$metrics = (new Trainer(0.01, 100000, 'features.txt'))->train('japanese.model', $cancel);
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。

## エラー

すべての例外は `Litsea\LitseaException` を継承するため、1 つの `catch` で捕捉できます。

| 例外 | 発生条件 |
|------|---------|
| `Litsea\InvalidArgumentException` | 未知の言語名、未知の feature set、使用済みトレーナ |
| `Litsea\ModelException` | ダウンロード失敗、または旧 joint POS モデル |
| `Litsea\IoException` | ファイルの読み書き失敗 |
| `Litsea\ParseException` | モデルまたは学習データの形式不正 |
| `Litsea\UnsupportedException` | このビルドでは利用できないスキームや操作 |
| `Litsea\PosUnavailableException` | 分割専用モデルに対する POS タグ付けの要求 |

## 命名規則

ext-php-rs はメソッドとプロパティを camelCase に変換するため、PHP 側の API は `segmentWithPos()`・`hasPos()`・`$metrics->numInstances` のようになります。

## 開発

```sh
make test-litsea-php    # cargo test + 拡張のビルド + PHPUnit
make build-litsea-php   # リリースビルド
```

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
