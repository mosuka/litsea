# segment

学習済みモデルを使用してテキストを単語に分割します。

## 使い方

```sh
echo "text" | litsea segment [OPTIONS] <MODEL_URI>
```

## 引数

| Argument | Description |
|----------|------------|
| `MODEL_URI` | 学習済みモデルファイルのパスまたはURL。サポート形式: ローカルファイルパス, `file://`, `http://`, `https://` |

## オプション

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | 文字タイプ分類に使用する言語。指定可能な値: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko`, `english` / `en` |
| `--pos` | off | 品詞推定付き分割を有効にします。[二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)モデル（`train --pos`）が必要です |
| `--threads <N>` | `1` | バッチ分割のワーカースレッド数（issue #185）。既定値では従来どおりのシングルスレッド動作。`N > 1` では入力行を並列に分割しつつ**入力順で**出力するため、出力はどちらでもバイト単位で同一です（`--pos` の有無を問わず使用可）。大きな入力の実時間はコア数に応じて短縮されますが、1 行あたりのレイテンシは変わりません |
| `--user-words <FILE>` | なし | ユーザー辞書（issue #263、#264）。1 行 1 語の UTF-8 ファイルで、語の後ろに TAB と UPOS タグを続けることもできます。登録語の出現箇所はすべて 1 トークンとして出力されます — 両端に境界を置き、内側には置きません。重なった出現箇所は leftmost-longest で解決します。`--pos` では、タグ付きの語のトークンに登録されたタグを付けます。1 文字の語と空行は無視されます |
| `--user-words-gate <RATIO>` | なし | `--user-words` の flip ratio による gate（`0.0..=1.0`）。語が覆すモデルの margin の割合が `RATIO` 以下のときだけ出現箇所を適用します（`0` = モデルがすでに同意している場所だけ、`1` = 常に）。`--user-words` が必要です |

## 入力 / 出力

- **入力**: stdinから読み取り、1行に1文。空行はスキップされます。
- **出力**: stdoutに書き込み、スペース区切りのトークン、入力行ごとに1行。
- **パイプライン**: 後段の処理がパイプを早期に閉じた場合（例:
  `litsea segment model | head -1`）、コマンドは正常終了します（終了コード0）。
  そのため `segment` はシェルのパイプライン内で問題なく連携できます。

## 使用例

**日本語:**

```sh
echo "LitseaはTinySegmenterを参考に開発された。" \
  | litsea segment -l japanese ./models/japanese.model
```

```text
Litsea は Tiny Segmenter を 参考 に 開発 さ れ た 。
```

**中国語:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./models/chinese.model
```

**韓国語:**

```sh
echo "한국어 단어 분할 테스트입니다." \
  | litsea segment -l korean ./models/korean.model
```

**英語:**

```sh
echo "I don't know." | litsea segment -l english ./models/english.model
```

**ファイルの処理:**

```sh
cat input.txt | litsea segment -l japanese ./models/japanese.model > output.txt
```

**URLからモデルを読み込み:**

```sh
echo "テスト文です。" \
  | litsea segment -l japanese https://example.com/models/japanese.model
```

## 品詞推定付き分割（`--pos`）

`--pos` フラグを指定すると、`train --pos` で作成した
[二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)モデルを
使って単語分割と品詞推定を一緒に行います。`--pos` に他の種類のモデルを
渡すと正確なエラーで失敗します（スタンドアロンの Averaged Perceptron
ファイル — 削除された joint POS モデル形式 — は、`train --pos` での
再学習を促すヒント付きで拒否されます）。

### 使い方

```sh
echo "text" | litsea segment --pos [OPTIONS] <MODEL_URI>
```

### 出力形式

各単語が `単語/品詞` の形式で出力されます。品詞は UPOS タグセットに準拠します。

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./models/japanese_pos.model
```

```text
今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

### ファイルの処理

```sh
cat input.txt | litsea segment --pos -l japanese ./models/japanese_pos.model > output.txt
```

## バッチ分割の並列化（`--threads`）

文どうしは独立なので、エンジンを変更することなくバッチスループットを
コア数でスケールできます: 入力行をチャンク単位で読み込み、各チャンクを
ワーカーへ分配し（各ワーカーは自分専用の再利用バッファを保持）、出力は
厳密に入力順で書き出します。

```sh
litsea segment --threads 8 -l japanese ./models/japanese.model < corpus.txt > segmented.txt
```

`--threads 1`（既定値）は従来の逐次ループそのものを使います。
なお `cargo bench -- external_corpus` はシングルスレッドの**エンジン**
計測のままです — CLI レベルのスレッドスケーリングとエンジンスループットは
別の数値であり、直接比較しないでください
（[ベンチマーク](../advanced/benchmarking.md)を参照）。

## ユーザー辞書（`--user-words`）

ユーザー辞書は、再学習なしで登録語を 1 トークンとして出力させます（issue #263）。
ファイルは 1 行 1 語の UTF-8 で、行末の改行だけを取り除くので語は書かれたとおりに
照合され、空行は読み飛ばし、コメント構文はありません。1 文字の語は無視されます。
語の後ろに TAB と `--pos` 用の UPOS タグを続けることもできます
（[タグ付きの語](#タグ付きの語)を参照）。

```sh
printf '雪だるま\n新幹線\n' > words.txt
printf '雪だるまを作る\n新幹線で帰る\n' \
  | litsea segment --user-words words.txt -l japanese ./models/japanese.model
```

```text
雪だるま を 作る
新幹線 で 帰る
```

辞書なしでは `japanese.model` は `雪 だる ま を 作る` と `新 幹線 で 帰る` を
出力します。辞書は `--pos` でも使え（タグなしの語は他の語と同じようにタグ付け
されます。この例では `雪だるま/NOUN を/ADP 作る/VERB`）、`--threads` でも
出力は逐次実行とバイト単位で同一です。

照合は文脈を見ません。登録語はその文字列が現れる場所すべてに適用されるため、
`って` のような 2 文字の断片を語として登録すると、それを含む他の語をすべて
壊します。短い語や曖昧な語を含む辞書には `--user-words-gate RATIO` を使うと、
語が覆すモデルの margin の割合が `RATIO` 以下のときだけ出現箇所を適用します。
`--user-words-gate 0` では上の例はモデル自身の `雪 だる ま を 作る` になります。
そこではモデルがこの語に同意していないためです。意味論の全体は
[Segmenter](../litsea/segmenter.md)（「ユーザー辞書」）を参照してください。

### タグ付きの語

行には、TAB の後ろに UPOS タグを書けます（`語<TAB>UPOS`、issue #264）。
`--pos` では、タグ付きの語のトークンすべてに、タガーのタグの代わりに登録された
タグを付けます。モデルの語彙表がタグを固定していても、タガーが決めるはずでも
同じです。それ以外のトークンはタガーのタグのままです。`--pos` なしでは、タグは
何も変えません。

```sh
printf 'チャットGPT\tPROPN\nポケモン\tPROPN\n雪だるま\n' > words.txt
printf 'チャットGPTに聞く\nポケモンを集める\n雪だるまを作る\n' \
  | litsea segment --pos --user-words words.txt -l japanese ./models/japanese_pos.model
```

```text
チャットGPT/PROPN に/ADP 聞く/VERB
ポケモン/PROPN を/ADP 集める/VERB
雪だるま/NOUN を/ADP 作る/VERB
```

タグなしで登録すると、`チャットGPT` と `ポケモン` は `NOUN` になります。
`--user-words-gate` を使う場合、gate が却下した出現箇所でも、モデルが結局登録
どおりに切り出せばタグが付きます。

行は最初の TAB で分けます。タグは前後の空白を取り除いてから（末尾の CR や空白は
受け付けます）解釈し、大文字で書いた 17 個の UPOS タグのどれかでなければなりません。
語は今までどおり書かれたとおりに照合します。同じ語を複数の行に書けますが、異なる
タグを付けるとエラーになり、タグなしで繰り返した行はタグを保ちます。ファイルは
`--pos` の有無にかかわらず検査し、エラーには行番号が付きます:

| 行 | エラー |
| --- | --- |
| `<TAB>NOUN` | TAB の前の語が空 |
| `word<TAB>NOUN<TAB>X` | 列が多すぎる |
| `word<TAB>` | UPOS タグが空 |
| `word<TAB>Noun` | 未知の UPOS タグ（それ以外は無視される 1 文字の語でも検査します） |
| `word<TAB>PROPN` の後の `word<TAB>NOUN` | 同じ語に 2 つの行で異なるタグ |

```text
Error: user dictionary words.txt, line 2: unknown UPOS tag "Noun" (expected one of ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X)
```

## 注意事項

- `--language` フラグは、モデルが学習された言語と一致する必要があります
- CLIは非同期のURI APIを通じてモデルを読み込み、TLS（rustls）を使用したHTTP/HTTPSをサポートしています。ライブラリには同期的なローカル読み込み（`load_model_from_path`）も用意されています
- モデルURIはファイルパスに限定されません -- 有効なURLであれば使用可能です
- `--pos` を使用する場合、モデルは `train --pos` で学習した二段構成モデルである必要があります
