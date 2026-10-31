# PHP

`litsea-php` は [ext-php-rs](https://github.com/davidcole1340/ext-php-rs) を用いて Litsea を PHP 8.1 以降へ公開するバインディングです。Packagist に `mosuka/litsea` として公開され、[PIE](https://github.com/php/pie) でインストールします。拡張は `litsea` という名前で登録されます。

## インストール

PHP 拡張は特定の PHP ABI 向けにビルドされた共有オブジェクトなので、ビルド済みバイナリは 1 つの PHP バージョン・スレッドセーフ方式・プラットフォームにしか合いません。PIE は、合うものが GitHub Release にあればそれをダウンロードし、無ければ使っている PHP に合わせて拡張をビルドします。自分の手でビルドすることもできます。

```sh
pie install mosuka/litsea
```

PIE 1.4 以降は、PHP 8.1〜8.5（NTS / ZTS）× Linux x86_64 / arm64（glibc 2.35 以上）または Apple シリコンの macOS 向けのビルド済み `litsea.so` を GitHub Release からダウンロードし、拡張ディレクトリへ配置して有効化します。Rust ツールチェーンは不要です（`--with-php-config=...` で別の PHP を選べます）。それ以外の組み合わせ（Alpine などの musl、古い glibc、Intel Mac、PIE 1.4 未満）ではソースビルドにフォールバックし、Rust ツールチェーン、libclang、PIE のビルドツール（`autoconf`、`libtool`、`make`）が必要になります。Windows は未対応です。Composer 自体は `php-ext` 型のパッケージを無視するため、`composer require mosuka/litsea` ではインストールできません。

PIE を使わない場合は自分でビルドして読み込みます。

```sh
cargo build --release -p litsea-php
php -d extension=/path/to/target/release/liblitsea_php.so your-script.php
```

常時読み込む場合は `php.ini` に `extension=/path/to/liblitsea_php.so` を追記します。macOS では cargo が `liblitsea_php.dylib` を書き出すので、そのパスを読み込んでください。どちらの方法でも `php -m` には `litsea` と表示されます。

## 静的解析向けのスタブ

`litsea-php/stubs/litsea.stubs.php` は、すべてのクラスと関数を IDE・PHPStan（`stubFiles`）・Psalm（`stubs`）向けに宣言したファイルです。ビルド済みの拡張から生成し、テストスイートが検証するため、Rust のソースと食い違うことはありません。解析ツールからこのファイルのコピーを指してください。実行時に読み込んではいけません（拡張がすでにクラスを宣言しています）。

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

## ユーザー辞書

すべてのコンストラクタは、省略可能な末尾の引数を 3 つ受け取ります（[#263](https://github.com/mosuka/litsea/issues/263)、[#264](https://github.com/mosuka/litsea/issues/264)）。`$user_words` は、現れた場所で必ず 1 トークンとして出力する語の配列、`$user_words_gate` は、モデルの margin が十分小さい場所にだけ語を適用する flip ratio（`0.0`〜`1.0`）、`$user_word_tags` は、さらに登録する語と、`segmentWithPos()` がその語に付ける UPOS タグの配列です。3 つとも既定値は `null` で、名前付き引数でも渡せます。

```php
$seg = Segmenter::open('japanese', 'models/japanese.model', ['雪だるま', '新幹線']);

$seg->segment('雪だるまを作る');
// ['雪だるま', 'を', '作る']   （辞書なしでは ['雪', 'だる', 'ま', 'を', '作る']）

Segmenter::open('japanese', 'models/japanese.model', user_words: ['雪だるま'], user_words_gate: 0.5);
```

POS モデルでは、`$user_word_tags` でタガーが誤る語のタグを直せます:

```php
$tagged = Segmenter::open('japanese', 'models/japanese_pos.model', user_word_tags: ['チャットGPT' => 'PROPN']);

array_map(fn ($t) => "{$t->surface}/{$t->pos}", $tagged->segmentWithPos('チャットGPTに聞く'));
// ['チャットGPT/PROPN', 'に/ADP', '聞く/VERB']   （$user_words に入れた場合は 'チャットGPT/NOUN'）
```

登録語には両端に境界が置かれ、内側には置かれません。重なった出現箇所は leftmost-longest で解決し、タグのない 1 文字の語は無視します。POS モデルでは、`$user_words` の語は他の語と同じようにタグ付けされます（`japanese_pos.model` では `雪だるま/NOUN`）。`$user_word_tags` の語と一致するトークンには、モデルの語彙表とタガーより優先して、そのタグが付きます。`segment()` はタグを無視します。`$user_word_tags` の 1 文字の語は境界を変えず、その 1 文字ちょうどのトークンのタグだけを置き換えます（issue #276）。照合は文脈を見ないため、短い断片を語として登録すると、それを含む他の語をすべて壊します。gate を `0.0` にすると、モデルがすでに同意している場所だけに語を適用します。`0.0`〜`1.0` の外の gate と、17 個の UPOS タグ（大文字）以外のタグは、`Litsea\InvalidArgumentException` を投げます。意味論の全体は [Segmenter](../litsea/segmenter.md#ユーザー辞書) を参照してください。

## API

| 呼び出し | 戻り値 |
|---------|-------|
| `Segmenter::open($language, $path, $user_words?, $user_words_gate?, $user_word_tags?)` | セグメンタ |
| `Segmenter::fromBytes($language, $data, $user_words?, $user_words_gate?, $user_word_tags?)` | セグメンタ |
| `Segmenter::fromUri($language, $uri, $user_words?, $user_words_gate?, $user_word_tags?)` | セグメンタ（ダウンロードはブロッキング） |
| `language()` | `string`（正規の言語名） |
| `hasPos()` | `bool` |
| `segment($text)` | `string[]` |
| `segmentBatch($texts)` | `string[][]` |
| `segmentTokens($text)` | バイトオフセット付き `Token[]` |
| `segmentWithPos($text)` | タグとオフセット付き `Token[]` |
| `segmentWithPosBatch($texts)` | `Token[][]` |
| `(new Extractor($language))->extract($corpus_path, $features_path, $tsv = false, $tag_free = false)` | `void`（`$tsv` は TSV 形式のコーパスを読み、`$tag_free` はタグに依存する特徴テンプレートを省きます） |
| `(new Extractor($language))->extractTwoStage($corpus_path, $output_prefix, $feature_set = null, $tsv = false)` | `void`（`{$output_prefix}.stage1`・`.stage2`・`.lexicon` を書き出します。`$feature_set` は `'full'`・`'balanced'`・`'fast'` のいずれかで、`null` は `'fast'` を意味します） |
| `(new Trainer($threshold, $num_iterations, $features_path))->train($model_path, $cancel?)` | `BinaryMetrics` |
| `(new Trainer(...))->loadModel($model_uri)` | `void`（既存のモデルから学習を続けます。パス・`file://`・`http(s)://` を受け付けます） |
| `(new PerceptronTrainer($num_epochs, $features_path))->train($model_path, $cancel?)` | `MulticlassMetrics` |
| `(new TwoStageTrainer($num_epochs, $features_prefix, $dominance = 0.99))->train($model_path, $cancel?)` | `TwoStageMetrics` |
| `(new TwoStageTrainer(...))->isAvailable()` | `bool`（`train()` の実行後は `false`） |
| `(new CancelToken())->cancel()` / `reset()` / `isCancelled()` | `void` / `void` / `bool` |
| `Litsea\version()` | `string`（`litsea` のバージョン） |
| `Litsea\supported_languages()` | `string[]` |

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

すべての例外は `Litsea\LitseaException` を継承するため、1 つの `catch` で捕捉できます（Python バインディングと同じ階層です）。下の表のどの例外にも当たらない失敗は、`Litsea\LitseaException` そのものとして投げられます。

| 例外 | 発生条件 |
|------|---------|
| `Litsea\InvalidArgumentException` | 未知の言語名、未知の feature set、使用済みトレーナ、`0.0`〜`1.0` の外の `$user_words_gate`、UPOS タグでない `$user_word_tags` の値、(0.5, 1.0] の外の `$dominance`、未知の URI スキーム |
| `Litsea\ModelException` | ダウンロード失敗、または旧 joint POS モデル |
| `Litsea\IoException` | ファイルの読み書き失敗 |
| `Litsea\ParseException` | モデルまたは学習データの形式不正 |
| `Litsea\UnsupportedException` | このビルドでは利用できないスキームや操作 |
| `Litsea\PosUnavailableException` | 分割専用モデルに対する POS タグ付けの要求 |

## 開発

```sh
make test-litsea-php      # cargo test + 拡張のビルド + PHPUnit
make test-litsea-php-pie  # PIE が実行する phpize / configure / make の経路
make stubs-litsea-php     # litsea-php/stubs/litsea.stubs.php を再生成
make lint-litsea-php      # clippy
make build-litsea-php     # リリースビルド
```

パリティテストは `litsea` CLI をビルドし、その出力とバインディングの出力を突き合わせます。

`composer.json` はリポジトリのルートにあります。Packagist がルートのマニフェストしか読まないためで、その `php-ext.build-path` が PIE を `litsea-php/` へ戻し、`config.m4` と `Makefile.frag` が PIE の `phpize` / `configure` / `make` の流れを `cargo build --release -p litsea-php` に変換します。
