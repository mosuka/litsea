# litsea-php

[Litsea](https://github.com/mosuka/litsea) の PHP バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。

[English README](README.md)

## インストール

PHP 拡張は特定の PHP ABI 向けにビルドされた共有オブジェクトであるため、ダウンロードして使えるビルド済みパッケージはありません。[PIE](https://github.com/php/pie) か自分の手で、使っている PHP に合わせてビルドします。PHP 8.1 以降が必要です。

### PIE でインストール（推奨）

パッケージは Packagist に `mosuka/litsea` として公開され、PHP 拡張のインストーラである PIE で入れます。Composer 自体は `php-ext` 型のパッケージをインストールしないため、`composer require mosuka/litsea` では入りません。

```sh
pie install mosuka/litsea
```

PIE はソースを取得し、`pie` を実行している PHP に合わせてビルドし（別の PHP を対象にするには `--with-php-config=/path/to/php-config` を指定）、`litsea.so` を拡張ディレクトリへ配置して有効化します。ビルドには Rust ツールチェーン（<https://rustup.rs/>）と libclang（Debian/Ubuntu では `libclang-dev`、macOS では Xcode コマンドラインツールか `brew install llvm`）が必要です。Windows は未対応です。

### 手動ビルド

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

どちらの方法でも拡張は `litsea` という名前で登録されます。`php -m` には `litsea` と表示され、確認には `extension_loaded('litsea')` を使います。

### 静的解析向けのスタブ

[`stubs/litsea.stubs.php`](stubs/litsea.stubs.php) は、この拡張のすべてのクラスと関数を IDE・PHPStan・Psalm 向けに宣言したファイルです。ビルド済みの拡張から生成し、古くなるとテストが失敗するため、Rust のソースと食い違うことはありません。解析ツールからこのファイルのコピーを指してください。実行時に読み込んではいけません（拡張がすでにクラスを宣言しているため二重定義になります）。

```neon
# phpstan.neon
parameters:
    stubFiles:
        - path/to/litsea/litsea-php/stubs/litsea.stubs.php
```

```xml
<!-- psalm.xml -->
<stubs>
    <file name="path/to/litsea/litsea-php/stubs/litsea.stubs.php"/>
</stubs>
```

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
make test-litsea-php      # cargo test + 拡張のビルド + PHPUnit
make test-litsea-php-pie  # PIE が実行する phpize / configure / make の経路
make stubs-litsea-php     # API 変更後に stubs/litsea.stubs.php を再生成
make build-litsea-php     # リリースビルド
```

`composer.json` はこのディレクトリではなくリポジトリのルートにあります。Packagist がそこから読むためで、その `php-ext.build-path` が PIE を `litsea-php/` へ戻し、`config.m4` と `Makefile.frag` が `cargo build` を実行します。

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
