# 特徴量抽出

Litsea は、各単語境界候補の周辺の局所的なコンテキストを捉えるために文字 n-gram 特徴量を使用します。本章ではすべての特徴量タイプをカタログ化します。

## 特徴量カテゴリ

入力の各文字位置 *i* について、セグメンタは文字、その種別コード、および前回の境界判定からなるスライディングウィンドウから特徴量を抽出します。

### 基本特徴量（38 個）

| Category | IDs | 説明 | Window |
|----------|-----|------|--------|
| **UW** (Unary Word) | UW1--UW6 | 位置 i-3 から i+2 の個々の文字 | 6 |
| **BW** (Bigram Word) | BW1--BW3 | 隣接する文字ペア | 3 |
| **UC** (Unary Char-type) | UC1--UC6 | 位置 i-3 から i+2 の文字種コード | 6 |
| **BC** (Bigram Char-type) | BC1--BC3 | 隣接する種別コードペア | 3 |
| **TC** (Trigram Char-type) | TC1--TC4 | 種別コードのトリプル | 4 |
| **UP** (Unary Previous-tag) | UP1--UP3 | 直前 3 つの境界判定 | 3 |
| **BP** (Bigram Previous-tag) | BP1--BP2 | 境界判定のペア | 2 |
| **UQ** (Unary tag+type) | UQ1--UQ3 | 境界判定と種別コードの組み合わせ | 3 |
| **BQ** (Bigram tag+type) | BQ1--BQ4 | 判定と種別コードのバイグラム組み合わせ | 4 |
| **TQ** (Trigram tag+type) | TQ1--TQ4 | 判定と種別コードのトライグラム組み合わせ | 4 |

### 言語固有の特徴量（4 個、日本語と中国語のみ）

| Category | IDs | 説明 | Count |
|----------|-----|------|-------|
| **WC** (Word+Char-type) | WC1--WC4 | 文字と種別コードの混合特徴量 | 4 |

- `WC1`: 位置 i-1 の文字 + 位置 i の種別コード
- `WC2`: 位置 i-1 の種別コード + 位置 i の文字
- `WC3`: 位置 i-1 の文字 + 位置 i-1 の種別コード
- `WC4`: 位置 i の文字 + 位置 i の種別コード

> **韓国語に WC がない理由:** 韓国語のハングル音節は 2 種類（SN と SF）にのみ分類されるため、WC 特徴量は有用な信号ではなくノイズを追加してしまいます。

### 特徴量の総数

| Language | Base | WC | Total |
|----------|------|----|-------|
| Japanese | 38 | 4 | **42** |
| Chinese | 38 | 4 | **42** |
| Korean | 38 | 0 | **38** |

## 特徴量の形式

各特徴量は `PREFIX:VALUE` の形式の文字列として表現されます:

```text
UW4:は        ← The character at position i is "は"
UC4:I         ← The type code at position i is "I" (Hiragana)
BW2:はテ      ← The bigram at position i-1..i is "はテ"
BC2:IK        ← The type bigram is Hiragana + Katakana
UP3:B         ← The previous boundary decision was "B" (boundary)
WC1:はK       ← Character "は" combined with type "K"
```

## スライディングウィンドウの配置

セグメンタは入力をセンチネル文字でパディングします:

```text
Index:   0    1    2    3    4    5    ...  n+2  n+3  n+4  n+5
Chars:   B3   B2   B1   c1   c2   c3  ...  cn   E1   E2   E3
Types:   O    O    O    t1   t2   t3  ...  tn   O    O    O
Tags:    U    U    U    U    ?    ?   ...  ?
```

- **B3, B2, B1** -- 開始センチネル（パディング）
- **E1, E2, E3** -- 終了センチネル（パディング）
- **O** -- パディング位置の「Other」種別
- **U** -- 初期位置の「Unknown」タグ
- **B** -- 「Boundary」タグ（単語の開始）
- **O** -- 「Other」タグ（継続）

特徴量は位置 4 から len-3 まで抽出され、i-3 から i+2 の完全なウィンドウが利用可能です。

## 学習データの形式

`extract` コマンドは以下の形式で特徴量をファイルに書き出します:

```text
1	UW1:B2 UW2:B1 UW3:L UW4:i UW5:t UC1:O UC2:O UC3:A UC4:A ...
-1	UW1:B1 UW2:L UW3:i UW4:t UW5:s UC1:O UC2:A UC3:A UC4:A ...
```

各行には以下が含まれます:
1. ラベル（境界の場合 `1`、非境界の場合 `-1`）
2. タブ区切りの特徴量文字列
