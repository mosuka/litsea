# litsea-ruby

[Litsea](https://github.com/mosuka/litsea) の Ruby バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。

[English README](README.md)

## インストール

```sh
gem install litsea
```

インストール時にネイティブ拡張をコンパイルするため、Rust ツールチェーンが必要です。Ruby 3.1 以降に対応しています。

## モデルは同梱されません

gem にはコードのみが含まれます。事前学習済みモデルは [Litsea リポジトリ](https://github.com/mosuka/litsea/tree/main/models)から取得し、パスを指定して読み込んでください。

| モデル | 用途 | サイズ |
|-------|------|-------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | 分割 | 84KB〜2.0MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | 分割 + POS | 3.0〜8.0MB |

どちらの種別かを指定する必要はありません。モデルファイル自身が種別を持っており、読み込んだモデルで何ができるかは `has_pos?` が示します。

## 使い方

### 分割

```ruby
require "litsea"

seg = Litsea::Segmenter.open(:japanese, "models/japanese.model")

seg.segment("これはテストです。")
# => ["これ", "は", "テスト", "です", "。"]

seg.segment_batch(["これはテストです。", "東京都から神奈川県へ引っ越した"])
```

言語は Symbol でも String でも指定でき、ISO 639-1 コードも使えます（`:ja` / `"japanese"`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```ruby
Litsea::Segmenter.open(:korean, "models/korean.model").segment("안녕하세요 반갑습니다")
# => ["안녕하세요", " ", "반갑습니다"]
```

### POS タグ付け

```ruby
seg = Litsea::Segmenter.open(:japanese, "models/japanese_pos.model")

seg.segment_with_pos("これはテストです。").each do |token|
  puts "#{token.surface}\t#{token.pos}\t[#{token.start}..#{token.end}]"
end
# これ    PRON    [0..6]
# は      ADP     [6..9]
# テスト  NOUN    [9..18]
# です    AUX     [18..24]
# 。      PUNCT   [24..27]
```

`start` と `end` は**バイト**オフセットです。Ruby の `String#[]` は文字単位なので、切り出しには `byteslice` を使ってください。

```ruby
text.byteslice(token.start, token.end - token.start)   # == token.surface
```

分割専用モデルに対して `segment_with_pos` を呼ぶと `Litsea::PosUnavailableError` が発生します。

### その他のモデル読み込み方法

```ruby
Litsea::Segmenter.from_bytes(:korean, File.binread("korean.model"))
Litsea::Segmenter.from_uri(:chinese, "https://example.com/chinese.model")
```

### 学習

```ruby
Litsea::Extractor.new(:japanese).extract("corpus.txt", "features.txt")

metrics = Litsea::Trainer.new(0.01, 10_000, "features.txt").train("japanese.model")
puts format("accuracy: %.2f%%", metrics.accuracy)
```

二段構成（分割 + POS）の学習:

```ruby
Litsea::Extractor.new(:japanese).extract_two_stage("corpus_pos.txt", "features", feature_set: "fast")

metrics = Litsea::TwoStageTrainer.new(10, "features").train("japanese_pos.model")
puts metrics.stage1.accuracy, metrics.stage2.accuracy
```

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。再利用可能かどうかは `available?` が示し、2 回目の `train` は例外を発生させます。

### 学習のキャンセル

学習は GVL を解放するため、他の Ruby スレッドが動き続けます。つまり、実行中の学習を別スレッドから停止できます。

```ruby
cancel = Litsea::CancelToken.new
Thread.new { sleep 60; cancel.cancel }

metrics = Litsea::Trainer.new(0.01, 100_000, "features.txt").train("japanese.model", cancel: cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。バインディングはシグナルハンドラを登録しません。

## エラー

すべてのエラーは `Litsea::Error` を継承するため、1 つの `rescue` で捕捉できます。

| エラー | 発生条件 |
|-------|---------|
| `Litsea::InvalidArgumentError` | 未知の言語名、未知の feature set、使用済みトレーナ |
| `Litsea::ModelError` | ダウンロード失敗、または旧 joint POS モデル |
| `Litsea::IoError` | ファイルの読み書き失敗 |
| `Litsea::ParseError` | モデルまたは学習データの形式不正 |
| `Litsea::UnsupportedError` | このビルドでは利用できないスキームや操作 |
| `Litsea::PosUnavailableError` | 分割専用モデルに対する POS タグ付けの要求 |

## 開発

```sh
make test-litsea-ruby    # cargo test + rake compile + rake test
make build-litsea-ruby   # リリースビルド
```

パリティテストは `litsea` CLI をビルドし、その出力とバインディングの出力を突き合わせます。なお、有効な Ruby に `bundle` が入っている必要があります（rbenv なら `rbenv local 3.4.9` などで 3.1 以降を選択してください）。

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
