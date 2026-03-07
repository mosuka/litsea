# convert-conllu

CoNLL-U（Universal Dependencies）形式のファイルを Litsea のコーパス形式に変換します。

## 目的

[Universal Dependencies](https://universaldependencies.org/) プロジェクトが提供する Treebank（CoNLL-U 形式）を、Litsea の単語分割モデルおよび品詞推定モデルの学習に必要なコーパス形式に変換します。

## 使い方

```sh
litsea convert-conllu [OPTIONS] <INPUT_FILE> <OUTPUT_FILE>
```

## 引数

| Argument | Description |
|----------|------------|
| `INPUT_FILE` | 入力 CoNLL-U ファイルのパス |
| `OUTPUT_FILE` | 出力 Litsea コーパスファイルのパス |

## オプション

| Option | Description |
|--------|------------|
| `--pos` | 出力に品詞タグを含める（`単語/品詞` 形式）。このフラグがない場合、スペース区切りの単語のみを出力 |

## 入力形式（CoNLL-U）

CoNLL-U はタブ区切りの10カラム形式です。本コマンドでは ID、FORM（表層形）、UPOS（品詞）の3カラムを使用します。

```text
# text = 太郎は花子が読んでいる本を次郎に渡した。
1	太郎	太郎	PROPN	_	_	12	nsubj	_	SpaceAfter=No
2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
3	花子	花子	PROPN	_	_	5	nsubj	_	SpaceAfter=No
4	が	が	ADP	_	_	3	case	_	SpaceAfter=No
5	読ん	読む	VERB	_	_	9	acl	_	SpaceAfter=No
...
```

## 出力形式

### 単語分割モード（デフォルト）

スペース区切りの単語、1行1文:

```text
太郎 は 花子 が 読ん で いる 本 を 次郎 に 渡し た 。
```

### 品詞推定モード（`--pos`）

`単語/品詞` をスペースで区切り、1行に1文を出力します:

```text
太郎/PROPN は/ADP 花子/PROPN が/ADP 読ん/VERB で/SCONJ いる/AUX 本/NOUN を/ADP 次郎/PROPN に/ADP 渡し/VERB た/AUX 。/PUNCT
```

## スキップされるデータ

変換時に以下の行はスキップされます:

- **コメント行**: `#` で始まる行
- **マルチワードトークン**: ID が `1-2` のようにハイフンを含む行
- **空ノード**: ID が `1.1` のようにピリオドを含む行
- **UPOS 未注釈**: UPOS カラムが `_` のトークン

## 使用例

### 単語分割用コーパスの作成

```sh
litsea convert-conllu ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus.txt
```

### 品詞付きコーパスの作成

```sh
litsea convert-conllu --pos ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus_pos.txt
```

### 単語分割の完全なワークフロー

```sh
# 1. CoNLL-U を単語分割用コーパスに変換
litsea convert-conllu ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus.txt

# 2. 特徴量を抽出
litsea extract -l japanese ./corpus.txt ./features.txt

# 3. モデルを学習
litsea train -t 0.005 -i 1000 ./features.txt ./resources/japanese.model
```

### 品詞推定の完全なワークフロー

```sh
# 1. CoNLL-U を品詞付きコーパスに変換
litsea convert-conllu --pos ./UD_Japanese-GSD/ja_gsd-ud-train.conllu ./corpus_pos.txt

# 2. 品詞付き特徴量を抽出
litsea extract --pos -l japanese ./corpus_pos.txt ./features_pos.txt

# 3. POS モデルを学習
litsea train --pos --num-epochs 10 ./features_pos.txt ./resources/japanese_pos.model
```

成功時のstderr出力:

```text
Converted 7050 sentences.
```
