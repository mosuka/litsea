# Python

`litsea-python` は [PyO3](https://pyo3.rs) と [maturin](https://www.maturin.rs) を用いて Litsea を Python 3.10 以降へ公開するバインディングです。PyPI では `litsea` という名前で配布します。

## インストール

```sh
pip install litsea
```

wheel は安定 ABI（`abi3-py310`）でビルドされるため、プラットフォームごとに 1 つの wheel がサポート対象の全 Python バージョンをカバーします。

## モデルの入手

パッケージにモデルは含まれません。[`models/`](https://github.com/mosuka/litsea/tree/main/models) ディレクトリから取得してパスを渡してください（[事前学習済みモデル](../pre-trained-models.md)を参照）。

モデルの種別を指定するフラグはありません。ファイル自身が種別を持っており、読み込んだモデルで何ができるかは `has_pos` が示します。

## 分割

```python
from litsea import Language, Segmenter

seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")

seg.segment("これはテストです。")
# ['これ', 'は', 'テスト', 'です', '。']
```

`Language` を受け取る箇所では言語名も使えます。`Segmenter.open("ja", ...)` と `Segmenter.open("japanese", ...)` は等価です。

空白区切りの言語では空白自体が 1 トークンとして返るため、トークンを連結すると常に入力が復元されます。

```python
Segmenter.open("ko", "models/korean.model").segment("안녕하세요 반갑습니다")
# ['안녕하세요', ' ', '반갑습니다']
```

## POS タグ付け

```python
seg = Segmenter.open(Language.JAPANESE, "models/japanese_pos.model")

for token in seg.segment_with_pos("これはテストです。"):
    print(token.surface, token.pos.name, token.start, token.end)
# これ PRON 0 6
# は ADP 6 9
# テスト NOUN 9 18
# です AUX 18 24
# 。 PUNCT 24 27
```

`start` と `end` は入力に対するバイトオフセットです。`text.encode()[token.start:token.end].decode()` で表層形が得られます。分割専用モデルに対して `segment_with_pos` を呼ぶと `PosUnavailableError` が送出されます。

## ユーザー辞書

すべてのコンストラクタは、キーワード専用の引数を 3 つ受け取ります（[#263](https://github.com/mosuka/litsea/issues/263)、[#264](https://github.com/mosuka/litsea/issues/264)）。`user_words` は、現れた場所で必ず 1 トークンとして出力する語のシーケンス、`user_words_gate` は、モデルの margin が十分小さい場所にだけ語を適用する flip ratio（`0.0`〜`1.0`）、`user_word_tags` は、さらに登録する語と、`segment_with_pos` がその語に付ける UPOS タグの dict です。

```python
seg = Segmenter.open("ja", "models/japanese.model", user_words=["雪だるま", "新幹線"])

seg.segment("雪だるまを作る")
# ['雪だるま', 'を', '作る']   （辞書なしでは ['雪', 'だる', 'ま', 'を', '作る']）
```

POS モデルでは、`user_word_tags` でタガーが誤る語のタグを直せます:

```python
seg = Segmenter.open("ja", "models/japanese_pos.model", user_word_tags={"チャットGPT": "PROPN"})

[f"{t.surface}/{t.pos.name}" for t in seg.segment_with_pos("チャットGPTに聞く")]
# ['チャットGPT/PROPN', 'に/ADP', '聞く/VERB']   （user_words に入れた場合は 'チャットGPT/NOUN'）
```

登録語には両端に境界が置かれ、内側には置かれません。重なった出現箇所は leftmost-longest で解決し、1 文字の語は無視します。POS モデルでは、`user_words` の語は他の語と同じようにタグ付けされます（`japanese_pos.model` では `雪だるま/NOUN`）。`user_word_tags` の語と一致するトークンには、モデルの語彙表とタガーより優先して、そのタグが付きます。`segment` はタグを無視します。照合は文脈を見ないため、短い断片を語として登録すると、それを含む他の語をすべて壊します。`user_words_gate=0.0` は、モデルがすでに同意している場所だけに語を適用します。`0.0`〜`1.0` の外の gate と、17 個の UPOS タグ（大文字）以外のタグは、`InvalidArgumentError` を送出します。意味論の全体は [Segmenter](../litsea/segmenter.md#ユーザー辞書) を参照してください。

## API

| 呼び出し | 戻り値 |
|---------|-------|
| `Segmenter.open(language, path, *, user_words=None, user_words_gate=None, user_word_tags=None)` | ファイルから読み込んだセグメンタ |
| `Segmenter.from_bytes(language, data, *, user_words=None, user_words_gate=None, user_word_tags=None)` | バイト列から読み込んだセグメンタ |
| `Segmenter.from_uri(language, uri, *, user_words=None, user_words_gate=None, user_word_tags=None)` | パス・`file://`・`http(s)://` から読み込んだセグメンタ |
| `segment(text)` | `list[str]` |
| `segment_batch(texts)` | `list[list[str]]` |
| `segment_tokens(text)` | バイトオフセット付き `list[Token]` |
| `segment_with_pos(text)` | タグとオフセット付き `list[Token]` |
| `segment_with_pos_batch(texts)` | `list[list[Token]]` |
| `Extractor(language).extract(...)` | 特徴量ファイルを書き出す |
| `Extractor(language).extract_two_stage(...)` | `.stage1` / `.stage2` / `.lexicon` を書き出す |
| `Trainer(threshold, iterations, features).train(model, cancel=None)` | `BinaryMetrics` |
| `PerceptronTrainer(epochs, features).train(model, cancel=None)` | `MulticlassMetrics` |
| `TwoStageTrainer(epochs, prefix, dominance=0.99).train(model, cancel=None)` | `TwoStageMetrics` |

`Language` と `Upos` は `enum.Enum` のサブクラスではなく PyO3 のクラスです。メンバーはクラス属性なので、列挙には `for x in Language` ではなく `Language.all()` / `Upos.all()` を使ってください。

## 学習

```python
from litsea import Extractor, Language, Trainer

Extractor(Language.JAPANESE).extract("corpus.txt", "features.txt")
metrics = Trainer(0.01, 10_000, "features.txt").train("japanese.model")
print(f"accuracy: {metrics.accuracy:.2f}%")
```

`TwoStageTrainer` は 1 度しか実行できません。学習時に stage 1 が AdaBoost モデルへ collapse され、トレーナが消費されるためです。再利用可能かどうかは `available` が示し、2 回目の `train()` は `InvalidArgumentError` を送出します。

### キャンセル

学習中は GIL が解放されるため、別スレッドから停止できます。

```python
import threading
from litsea import CancelToken, Trainer

cancel = CancelToken()
threading.Timer(60.0, cancel.cancel).start()
metrics = Trainer(0.01, 100_000, "features.txt").train("japanese.model", cancel=cancel)
```

キャンセルは**エラーではありません**。次のチェックポイントで停止し、部分的に学習されたモデルを保存してメトリクスを返します。バインディングはシグナルハンドラを登録しないため、Ctrl-C の扱いはアプリケーション側のままです。

## エラー

すべての例外は `LitseaError` を継承します。

| 例外 | 発生条件 |
|------|---------|
| `InvalidArgumentError` | 未知の言語名、未知の feature set、使用済みトレーナ、`0.0`〜`1.0` の外の `user_words_gate`、UPOS タグでない `user_word_tags` の値 |
| `ModelError` | ダウンロード失敗、または旧 joint POS モデル |
| `IoError` | ファイルの読み書き失敗 |
| `ParseError` | モデルまたは学習データの形式不正 |
| `UnsupportedError` | このビルドでは利用できないスキームや操作 |
| `PosUnavailableError` | 分割専用モデルに対する POS タグ付けの要求 |

## スレッドと GIL

`Segmenter` はイミュータブルで、スレッド間で共有できます。`segment_batch`・`segment_with_pos_batch`・`extract`・各 `train` は GIL を解放します。

単文の `segment` / `segment_with_pos` は GIL を保持します。GIL を解放するには入力文字列を所有する必要があり（PyO3 の `Ungil` 境界により、GIL 解放中に Python 所有のメモリへ触れられないため）、そのコピーのコストが 1 文の分割コストを上回るからです。大量処理にはバッチ版を使ってください。

## 開発

```sh
make setup-venv            # venv 作成と開発ツールのインストール
make test-litsea-python    # cargo test + maturin develop + pytest
make lint-litsea-python    # clippy + ruff
make build-litsea-python   # リリース wheel を litsea-python/dist へ出力
```

パリティテストは `litsea` CLI をビルドして、その出力とバインディングの出力を突き合わせます。何が正しいかを決めるのはハードコードした期待値ではなく参照実装です。
