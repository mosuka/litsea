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
| `--pos` | off | 品詞推定付き分割を有効にします。joint モデル（`train --pos`）または[二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)モデル（`train --two-stage`）のいずれも使用でき、ファイルから自動判別されます |
| `--threads <N>` | `1` | バッチ分割のワーカースレッド数（issue #185）。既定値では従来どおりのシングルスレッド動作。`N > 1` では入力行を並列に分割しつつ**入力順で**出力するため、出力はどちらでもバイト単位で同一です（`--pos` の有無を問わず使用可）。大きな入力の実時間はコア数に応じて短縮されますが、1 行あたりのレイテンシは変わりません |

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
  | litsea segment -l japanese ./models/RWCP.model
```

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
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

`--pos` フラグを指定すると、単語分割と品詞推定を同時に行います。モデルの種類
（joint な Averaged Perceptron モデルか、
[二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)モデルか）
はファイルヘッダから自動判別されるため、`train --two-stage` で学習したモデルでも
同じコマンドがそのまま使えます。

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

## 注意事項

- `--language` フラグは、モデルが学習された言語と一致する必要があります
- CLIは非同期のURI APIを通じてモデルを読み込み、TLS（rustls）を使用したHTTP/HTTPSをサポートしています。ライブラリには同期的なローカル読み込み（`load_model_from_path`）も用意されています
- モデルURIはファイルパスに限定されません -- 有効なURLであれば使用可能です
- `--pos` を使用する場合、モデルは `train --pos` または `train --two-stage` で学習した品詞推定対応モデルである必要があります
