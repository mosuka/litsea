# litsea-python

[Litsea](https://github.com/mosuka/litsea) の Python バインディングです。Litsea は日本語・中国語・韓国語・英語に対応した、コンパクトな単語分割と品詞（POS）タグ付けのライブラリです。

[English README](README.md)

## インストール

```sh
pip install litsea
```

wheel は安定 ABI（abi3）でビルドされており、CPython 3.10 以降で動作します。

## モデルは同梱されません

このパッケージにはコードのみが含まれます。事前学習済みモデルは [Litsea リポジトリ](https://github.com/mosuka/litsea/tree/main/models)から取得し、パスを指定して読み込んでください。

| モデル | 用途 | サイズ |
|-------|------|-------|
| `japanese.model`, `chinese.model`, `korean.model`, `english.model` | 分割 | 84KB〜2.0MB |
| `japanese_pos.model`, `chinese_pos.model`, `korean_pos.model`, `english_pos.model` | 分割 + POS | 3.0〜8.0MB |

どちらの種別かを指定する必要はありません。モデルファイル自身が種別を持っており、読み込んだモデルで何ができるかは `has_pos` が示します。

## 使い方

### 分割

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")

seg.segment("これはテストです。")
# ['これ', 'は', 'テスト', 'です', '。']

seg.segment_batch(["これはテストです。", "東京都から神奈川県へ引っ越した"])
# [['これ', 'は', 'テスト', 'です', '。'],
#  ['東京', '都', 'から', '神奈川', '県', 'へ', '引っ越し', 'た']]
```

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```python
Segmenter.open("ko", "models/korean.model").segment("안녕하세요 반갑습니다")
# ['안녕하세요', ' ', '반갑습니다']
```

`Language` を受け取る箇所では言語名の文字列も使えます。

```python
Segmenter.open("ja", "models/japanese.model")
Segmenter.open("japanese", "models/japanese.model")
```

### POS タグ付け

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese_pos.model")
seg.has_pos
# True

for token in seg.segment_with_pos("これはテストです。"):
    print(token.surface, token.pos.name, token.start, token.end)
# これ PRON 0 6
# は ADP 6 9
# テスト NOUN 9 18
# です AUX 18 24
# 。 PUNCT 24 27
```

`start` と `end` は入力文字列に対するバイトオフセットです。`text.encode()[token.start:token.end].decode()` で表層形が復元できます。分割・POS のどちらの出力でも正確で、空白を保持する韓国語・英語でも同様です。

分割専用モデルに対して `segment_with_pos` を呼ぶと `PosUnavailableError` が送出されます。

### その他のモデル読み込み方法

```python
Segmenter.from_bytes(Language.KOREAN, open("korean.model", "rb").read())
Segmenter.from_uri(Language.CHINESE, "https://example.com/chinese.model")
```

### 学習

```python
from litsea import Extractor, Language, Trainer

Extractor(Language.JAPANESE).extract("corpus.txt", "features.txt")

metrics = Trainer(0.01, 10_000, "features.txt").train("japanese.model")
print(f"accuracy: {metrics.accuracy:.2f}%")
```

二段構成（分割 + POS）の学習:

```python
from litsea import Extractor, Language, TwoStageTrainer

Extractor(Language.JAPANESE).extract_two_stage("corpus_pos.txt", "features", feature_set="fast")

metrics = TwoStageTrainer(10, "features").train("japanese_pos.model")
print(metrics.stage1.accuracy, metrics.stage2.accuracy)
```

`TwoStageTrainer` は 1 度しか使えません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。再利用可能かどうかは `available` で確認できます。

### 学習のキャンセル

学習中は GIL が解放されるため、別スレッドから停止できます。

```python
import threading
from litsea import CancelToken, Trainer

cancel = CancelToken()
trainer = Trainer(0.01, 100_000, "features.txt")

threading.Timer(60.0, cancel.cancel).start()
metrics = trainer.train("japanese.model", cancel=cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。

バインディング側でシグナルハンドラを登録することはないため、Ctrl-C の扱いはアプリケーション側の裁量のままです。

## エラー

すべての例外は `LitseaError` を継承するため、1 つの `except` で捕捉できます。

| 例外 | 発生条件 |
|------|---------|
| `InvalidArgumentError` | 未知の言語名、未知の feature set、使用済みトレーナ |
| `ModelError` | ダウンロード失敗、または旧 joint POS モデル |
| `IoError` | ファイルの読み書き失敗 |
| `ParseError` | モデルまたは学習データの形式不正 |
| `UnsupportedError` | このビルドでは利用できないスキームや操作 |
| `PosUnavailableError` | 分割専用モデルに対する POS タグ付けの要求 |

## スレッド

`Segmenter` はイミュータブルで、スレッド間で共有できます。`segment_batch`・`segment_with_pos_batch`・`extract`・各 `train` は GIL を解放します。単文の `segment` / `segment_with_pos` は GIL を保持します。解放するには入力文字列のコピーが必要で、そのコストが 1 文の分割コストを上回るためです。

## 開発

```sh
make setup-venv            # venv 作成と開発ツールのインストール
make test-litsea-python    # cargo test + maturin develop + pytest
make build-litsea-python   # リリース wheel のビルド
```

## ライセンス

MIT。[LICENSE](../LICENSE) を参照してください。
