# UPOS

`upos` モジュールは、品詞タグ付けに使用する Universal POS (UPOS) タグセットと分割ラベル型を定義します。

## Upos

### 定義

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Upos {
    ADJ,    // 形容詞 (Adjective)
    ADP,    // 接置詞 (Adposition)
    ADV,    // 副詞 (Adverb)
    AUX,    // 助動詞 (Auxiliary)
    CCONJ,  // 等位接続詞 (Coordinating conjunction)
    DET,    // 限定詞 (Determiner)
    INTJ,   // 間投詞 (Interjection)
    NOUN,   // 名詞 (Noun)
    NUM,    // 数詞 (Numeral)
    PART,   // 助詞・小辞 (Particle)
    PRON,   // 代名詞 (Pronoun)
    PROPN,  // 固有名詞 (Proper noun)
    PUNCT,  // 句読点 (Punctuation)
    SCONJ,  // 従属接続詞 (Subordinating conjunction)
    SYM,    // 記号 (Symbol)
    VERB,   // 動詞 (Verb)
    X,      // その他 (Other)
}
```

Litsea は [Universal Dependencies](https://universaldependencies.org/u/pos/) プロジェクトの全 17 UPOS タグをサポートしています:

| タグ | 説明 | 例（日本語） |
|-----|------|-------------|
| `ADJ` | 形容詞 | いい, 大きい |
| `ADP` | 接置詞 | は, が, を, に |
| `ADV` | 副詞 | とても, まだ |
| `AUX` | 助動詞 | です, ます, た |
| `CCONJ` | 等位接続詞 | と, や |
| `DET` | 限定詞 | この, その |
| `INTJ` | 間投詞 | ああ, はい |
| `NOUN` | 名詞 | 天気, 本 |
| `NUM` | 数詞 | 一, 二, 100 |
| `PART` | 助詞・小辞 | ね, よ |
| `PRON` | 代名詞 | これ, それ |
| `PROPN` | 固有名詞 | 東京, 太郎 |
| `PUNCT` | 句読点 | 。, 、 |
| `SCONJ` | 従属接続詞 | ので, から |
| `SYM` | 記号 | %, $ |
| `VERB` | 動詞 | 読む, 書く |
| `X` | その他 | （未分類トークン） |

### 定数

#### `Upos::ALL`

```rust
pub const ALL: [Upos; 17]
```

全 17 品詞の配列を返します。

### トレイト実装

- `Display`: `"NOUN"`, `"VERB"` などの文字列に変換
- `FromStr`: 文字列から `Upos` にパース。不正な文字列にはエラーを返す

```rust
use litsea::upos::Upos;

let pos: Upos = "NOUN".parse().unwrap();
assert_eq!(pos.to_string(), "NOUN");
```

## SegmentLabel

### 定義

`SegmentLabel` 型は単語境界検出と品詞タグ付けを組み合わせます。各文字位置に 18 ラベルのいずれかが割り当てられます:

- **`B(Upos)`**（17 ラベル）: 指定された UPOS タグを持つ単語境界（例: `B-NOUN`, `B-VERB`）
- **`O`**（1 ラベル）: 非境界（現在の単語の継続）

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    B(Upos),  // 単語の先頭文字（境界）。品詞情報を持つ
    O,        // 単語の継続文字（非境界）
}
```

```rust
use litsea::upos::SegmentLabel;

// "今日は" の分割ラベル
// 今 → B-NOUN  （"今日" の先頭、NOUN としてタグ付け）
// 日 → O       （"今日" の継続）
// は → B-ADP   （"は" の先頭、ADP としてタグ付け）
```

### メソッド

#### `all_labels`

```rust
pub fn all_labels() -> Vec<SegmentLabel>
```

全 18 ラベル（B-ADJ, B-ADP, ..., B-X, O）の一覧を返します。

#### `is_boundary`

```rust
pub fn is_boundary(&self) -> bool
```

境界ラベル（`B-*`）かどうかを返します。

#### `pos`

```rust
pub fn pos(&self) -> Option<Upos>
```

品詞タグを返します。非境界（`O`）の場合は `None`。

### トレイト実装

- `Display`: `"B-NOUN"`, `"O"` などの文字列に変換
- `FromStr`: 文字列から `SegmentLabel` にパース

```rust
use litsea::upos::{SegmentLabel, Upos};

let label: SegmentLabel = "B-NOUN".parse().unwrap();
assert!(label.is_boundary());
assert_eq!(label.pos(), Some(Upos::NOUN));

let label_o: SegmentLabel = "O".parse().unwrap();
assert!(!label_o.is_boundary());
assert_eq!(label_o.pos(), None);
```
