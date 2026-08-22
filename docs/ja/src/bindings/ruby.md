# Ruby

`litsea-ruby` は [magnus](https://github.com/matsadler/magnus) と rb-sys を用いて Litsea を Ruby 3.1 以降へ公開するバインディングです。RubyGems では `litsea` として配布します。

## インストール

```sh
gem install litsea
```

gem はソース配布で、インストール時に拡張をコンパイルするため Rust ツールチェーンが必要です。

## モデルの入手

gem にモデルは含まれません。[`models/`](https://github.com/mosuka/litsea/tree/main/models) から取得してパスを渡してください（[事前学習済みモデル](../pre-trained-models.md)を参照）。モデル自身が種別を持つため、フラグの指定は不要で、読み込んだモデルで何ができるかは `has_pos?` が示します。

## 分割

```ruby
require "litsea"

seg = Litsea::Segmenter.open(:japanese, "models/japanese.model")

seg.segment("これはテストです。")
# => ["これ", "は", "テスト", "です", "。"]
```

言語は Symbol でも String でも指定でき、ISO 639-1 コードも使えます（`:ja` / `"japanese"`）。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```ruby
Litsea::Segmenter.open(:korean, "models/korean.model").segment("안녕하세요 반갑습니다")
# => ["안녕하세요", " ", "반갑습니다"]
```

## POS タグ付け

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

## API

| 呼び出し | 戻り値 |
|---------|-------|
| `Litsea::Segmenter.open(language, path)` | セグメンタ |
| `Litsea::Segmenter.from_bytes(language, data)` | セグメンタ（バイナリ String も可） |
| `Litsea::Segmenter.from_uri(language, uri)` | セグメンタ |
| `#segment(text)` | `Array<String>` |
| `#segment_batch(texts)` | `Array<Array<String>>` |
| `#segment_tokens(text)` | バイトオフセット付き `Array<Litsea::Token>` |
| `#segment_with_pos(text)` | タグとオフセット付き `Array<Litsea::Token>` |
| `#segment_with_pos_batch(texts)` | `Array<Array<Litsea::Token>>` |
| `Litsea::Extractor.new(language)#extract(...)` | `nil` |
| `Litsea::Extractor.new(language)#extract_two_stage(...)` | `nil` |
| `Litsea::Trainer.new(threshold, iterations, features)#train(model, cancel:)` | `BinaryMetrics` |
| `Litsea::PerceptronTrainer.new(epochs, features)#train(model, cancel:)` | `MulticlassMetrics` |
| `Litsea::TwoStageTrainer.new(epochs, prefix, dominance:)#train(model, cancel:)` | `TwoStageMetrics` |

## GVL の解放

モデルの読み込み・特徴量抽出・学習といった時間のかかる処理は、GVL（Global VM Lock）を解放した状態で実行されます。そのため他の Ruby スレッドが動き続け、キャンセルが意味を持ちます。

```ruby
cancel = Litsea::CancelToken.new
Thread.new { sleep 60; cancel.cancel }

metrics = Litsea::Trainer.new(0.01, 100_000, "features.txt").train("japanese.model", cancel: cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。バインディングはシグナルハンドラを登録しません。

`rb_thread_call_without_gvl` は magnus も rb-sys もラップしていません（magnus は未バインドの C 関数一覧に挙げており、この関数は rb-sys が生成するバインディングの対象外ヘッダで宣言されています）。そのため本バインディングは `src/gvl.rs` で自ら宣言し、パニックが C フレームを越えて巻き戻らないよう `extern "C"` のトランポリンで捕捉しています。この主張は 2 つのテストで担保しています。1 つは学習中に別の Ruby スレッドが動き続けること、もう 1 つは別スレッドからのキャンセルが学習ウィンドウ内に収まることです。GVL 解放を外すと両方とも失敗します。

1 文の分割は GVL を保持したままです。解放のコストの方が処理そのものより大きいためです。

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。状態は `available?` が示し、2 回目の `train` は例外を発生させます。

## エラー

すべてのエラーは `Litsea::Error` を継承するため、1 つの `rescue` で捕捉できます（Python・PHP バインディングと同じ階層です）。

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
make lint-litsea-ruby    # clippy + rubocop
make build-litsea-ruby   # リリースビルド
```

有効な Ruby で `bundle` が使える必要があります。バージョン管理ツールの shim が存在していても、選択中のインタプリタに bundler が無い場合があるため、Makefile がその旨を検出して案内します。パリティテストは `litsea` CLI をビルドし、その出力とバインディングの出力を突き合わせます。
