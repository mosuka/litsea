# CoNLL-U コンバーター

`conllu` モジュールは、[CoNLL-U](https://universaldependencies.org/format.html)（Universal Dependencies）形式のファイルを、単語分割用と品詞推定用の両方の Litsea コーパス形式に変換する機能を提供します。

## 概要

Universal Dependencies の Treebanks は CoNLL-U 形式で配布されています。`conllu` モジュールはこの形式を解析し、2つのモードで Litsea 互換のコーパスファイルを生成します:

- **単語分割モード** (`with_pos: false`): スペース区切りの単語、1行1文
- **品詞推定モード** (`with_pos: true`): `単語/品詞 単語/品詞 ...` 形式、1行1文

## 関数

### `convert_conllu`

```rust
pub fn convert_conllu(
    input_path: &Path,
    output_path: &Path,
    with_pos: bool,
) -> Result<usize, Box<dyn Error>>
```

CoNLL-U ファイルを読み込み、Litsea コーパス形式に変換して出力ファイルに書き込みます。変換された文数を返します。

#### 引数

- `input_path` - CoNLL-U ファイルのパス
- `output_path` - 出力ファイルのパス
- `with_pos` - `true` の場合は `単語/品詞` 形式で出力、`false` の場合はスペース区切りの単語のみ

#### 戻り値

変換に成功した文の数。

#### 変換ルール

- コメント行（`#` で始まる行）はスキップ
- マルチワードトークン（ID にハイフンを含む行、例: `1-2`）はスキップ
- 空ノード（ID にピリオドを含む行、例: `1.1`）はスキップ
- UPOS が `_` のトークンはスキップ
- 空行は文の区切りとして扱う

#### 使用例（単語分割用）

```rust
use std::path::Path;
use litsea::conllu::convert_conllu;

let input = Path::new("./UD_Japanese-GSD/ja_gsd-ud-train.conllu");
let output = Path::new("./corpus.txt");

// スペース区切りの単語を出力
let sentence_count = convert_conllu(input, output, false).unwrap();
println!("Converted {} sentences.", sentence_count);
```

#### 使用例（品詞推定用）

```rust
use std::path::Path;
use litsea::conllu::convert_conllu;

let input = Path::new("./UD_Japanese-GSD/ja_gsd-ud-train.conllu");
let output = Path::new("./pos_corpus.txt");

// 単語/品詞 形式で出力
let sentence_count = convert_conllu(input, output, true).unwrap();
println!("Converted {} sentences.", sentence_count);
```

#### 入力例（CoNLL-U）

```text
# text = 太郎は走った。
1	太郎	太郎	PROPN	_	_	3	nsubj	_	SpaceAfter=No
2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
3	走っ	走る	VERB	_	_	0	root	_	SpaceAfter=No
4	た	た	AUX	_	_	3	aux	_	SpaceAfter=No
5	。	。	PUNCT	_	_	3	punct	_	SpaceAfter=No

```

#### 出力例（単語分割モード）

```text
太郎 は 走っ た 。
```

#### 出力例（品詞推定モード）

```text
太郎/PROPN は/ADP 走っ/VERB た/AUX 。/PUNCT
```
