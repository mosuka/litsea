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
| `-l`, `--language <LANGUAGE>` | `japanese` | 文字タイプ分類に使用する言語。指定可能な値: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |

## 入力 / 出力

- **入力**: stdinから読み取り、1行に1文。空行はスキップされます。
- **出力**: stdoutに書き込み、スペース区切りのトークン、入力行ごとに1行。

## 使用例

**日本語:**

```sh
echo "LitseaはTinySegmenterを参考に開発された。" \
  | litsea segment -l japanese ./resources/japanese.model
```

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
```

**中国語:**

```sh
echo "中文分词测试。" | litsea segment -l chinese ./resources/chinese.model
```

**韓国語:**

```sh
echo "한국어 단어 분할 테스트입니다." \
  | litsea segment -l korean ./resources/korean.model
```

**ファイルの処理:**

```sh
cat input.txt | litsea segment -l japanese ./resources/japanese.model > output.txt
```

**URLからモデルを読み込み:**

```sh
echo "テスト文です。" \
  | litsea segment -l japanese https://example.com/models/japanese.model
```

## 品詞推定付き分割（`--pos`）

`--pos` フラグを指定すると、Averaged Perceptron モデルを使用して単語分割と品詞推定を同時に行います。

### 使い方

```sh
echo "text" | litsea segment --pos [OPTIONS] <MODEL_URI>
```

### 出力形式

各単語が `単語/品詞` の形式で出力されます。品詞は UPOS タグセットに準拠します。

```sh
echo "今日はいい天気ですね。" \
  | litsea segment --pos -l japanese ./resources/japanese_pos.model
```

```text
今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

### ファイルの処理

```sh
cat input.txt | litsea segment --pos -l japanese ./resources/japanese_pos.model > output.txt
```

## 注意事項

- `--language` フラグは、モデルが学習された言語と一致する必要があります
- モデルの読み込みは非同期で行われ、TLS（rustls）を使用したHTTP/HTTPSをサポートしています
- モデルURIはファイルパスに限定されません -- 有効なURLであれば使用可能です
