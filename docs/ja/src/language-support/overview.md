# 言語サポート概要

Litseaは、`Language` 列挙型に基づく統一的なフレームワークを通じて、4つの言語の単語分割をサポートしています。

## サポート言語

| Language | Enum Variant | CLI Values | Feature Count | 単語 F1（held-out） |
|----------|-------------|------------|---------------|---------------------|
| 日本語 | `Language::Japanese` | `japanese`, `ja` | 42 | 96.70% |
| 中国語 | `Language::Chinese` | `chinese`, `zh` | 42 | 90.69% |
| 韓国語 | `Language::Korean` | `korean`, `ko` | 38 | 99.91% |
| 英語 | `Language::English` | `english`, `en` | 38 | 98.31% |

## Language 列挙型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    English,
}
```

- **デフォルト**は `Japanese`
- `#[non_exhaustive]` 付き -- 新しい言語を破壊的変更なしに追加できるため、外部の `match` 式にはワイルドカードアームが必要
- `FromStr` を実装 -- 完全な言語名またはISO 639-1コードからパース可能（大文字小文字を区別しない）
- `Display` を実装 -- 小文字の完全な言語名を出力

### パース例

```rust
use litsea::language::Language;

let ja: Language = "japanese".parse().unwrap();
let zh: Language = "zh".parse().unwrap();
let ko: Language = "Korean".parse().unwrap();   // case-insensitive
let err = "french".parse::<Language>();          // Err(...)
```

## 言語間の違い

各言語は独自の**文字タイプパターン**を定義しており、文字をタイプコードに分類します。これらのタイプコードはAdaBoost分類器の特徴量として使用されます。

| Aspect | Japanese | Chinese | Korean | English |
|--------|----------|---------|--------|---------|
| 文字タイプ数 | 8 (M, H, I, K, P, A, N, O) | 9 (F, C, X, R, P, B, A, N, O) | 10 (E, SN, SF, J, G, H, P, A, N, O) | 7 (U, W, Q, P, A, N, O) |
| WC特徴量 | あり（4個追加） | あり（4個追加） | なし | なし |
| 総特徴量数 | 42 | 42 | 38 | 38 |
| マッチング方式 | `match`（文字範囲） | `match`（文字範囲） | `match`（文字範囲）+ コードポイント判定 | `match`（文字範囲） |

### 韓国語・英語の特徴量が少ない理由

韓国語のハングル音節は、**SN**（받침/終声なし）と**SF**（받침あり）の2種類にのみ分類されます。この二値的な区別では、WC特徴量（単語＋文字タイプの組み合わせ）は冗長な情報を生成し、識別力がほとんどありません。これらを除外することで、ノイズを低減し、モデルをコンパクトに保ちます。

英語も関連する理由で同じ結論に至りました。支配的な境界シグナルは空白であり、held-out
の dev split での比較では、WC を含まない38テンプレートの tag-free モデルが Word F1
98.68%、WC（`WC1`〜`WC4`）を含む42テンプレートでは98.65%という結果になりました
-- WC特徴量は「役に立たない」どころか「悪化させる」ことが実測で確認されています。
詳細は[英語](english.md#wc特徴量なし)を参照してください。
