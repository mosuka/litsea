# split-sentences

Unicode UAX #29のルールを使用してテキストを文に分割します。

## 使い方

```sh
echo "text" | litsea split-sentences
```

## 引数

なし。

## オプション

なし（`--help` と `--version` を除く）。

## 入力 / 出力

- **入力**: stdinから読み取り、1行に1段落。空行はスキップされます。
- **出力**: stdoutに書き込み、1行に1文。

## 動作の仕組み

このコマンドは、ICU4Xの `SentenceSegmenter` を使用しており、Unicode Standard Annex #29（UAX #29）の文区切りルールを実装しています。**言語非依存**のため、`--language` フラグは不要です。

## 使用例

```sh
echo "これはテストです。次の文です。" | litsea split-sentences
```

出力:

```text
これはテストです。
次の文です。
```

複数行の入力:

```sh
echo -e "First sentence. Second sentence.\nThird sentence! Fourth." \
  | litsea split-sentences
```

出力:

```text
First sentence.
Second sentence.
Third sentence!
Fourth.
```

## ユースケース

- 単語分割の前処理として、テキストを1行1文に変換
- 大規模なドキュメントを分析用に個別の文に分割
- 生テキストから学習コーパスを準備
