# convert-conllu

CoNLL-U（Universal Dependencies）形式のファイルを Litsea の品詞付きコーパス形式に変換します。

## 目的

[Universal Dependencies](https://universaldependencies.org/) プロジェクトが提供する Treebank（CoNLL-U 形式）を、Litsea の品詞推定モデル学習に必要なコーパス形式に変換します。

## 使い方

```sh
litsea convert-conllu <INPUT_FILE> <OUTPUT_FILE>
```

## 引数

| Argument | Description |
|----------|------------|
| `INPUT_FILE` | 入力 CoNLL-U ファイルのパス |
| `OUTPUT_FILE` | 出力 Litsea コーパスファイルのパス |

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

## 出力形式（Litsea コーパス）

`単語/品詞` をスペースで区切り、1行に1文を出力します。

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

### UD Japanese-GSD での使用

```sh
# UD Japanese-GSD リポジトリをクローン
git clone https://github.com/UniversalDependencies/UD_Japanese-GSD.git

# CoNLL-U を Litsea コーパス形式に変換
litsea convert-conllu \
    ./UD_Japanese-GSD/ja_gsd-ud-train.conllu \
    ./corpus_pos.txt

# 変換後のコーパスで品詞モデルを学習
litsea extract --pos -l japanese ./corpus_pos.txt ./features_pos.txt
litsea train --pos --num-epochs 10 ./features_pos.txt ./resources/japanese_pos.model
```

成功時のstderr出力:

```text
Converted 7050 sentences.
```
