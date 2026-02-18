# 言語サポート概要

Litseaは、`Language` 列挙型に基づく統一的なフレームワークを通じて、3つの言語の単語分割をサポートしています。

## サポート言語

| Language | Enum Variant | CLI Values | Feature Count | Pre-trained Model Accuracy |
|----------|-------------|------------|---------------|---------------------------|
| 日本語 | `Language::Japanese` | `japanese`, `ja` | 42 | 94.15% |
| 中国語 | `Language::Chinese` | `chinese`, `zh` | 42 | 80.72% |
| 韓国語 | `Language::Korean` | `korean`, `ko` | 38 | 85.08% |

## Language 列挙型

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
}
```

- **デフォルト**は `Japanese`
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

| Aspect | Japanese | Chinese | Korean |
|--------|----------|---------|--------|
| 文字タイプ数 | 8 (M, H, I, K, P, A, N, O) | 9 (F, C, X, R, P, B, A, N, O) | 10 (E, SN, SF, J, G, H, P, A, N, O) |
| WC特徴量 | あり（4個追加） | あり（4個追加） | なし |
| 総特徴量数 | 42 | 42 | 38 |
| マッチング方式 | 正規表現のみ | 正規表現のみ | 正規表現 + クロージャ |

### 韓国語の特徴量が少ない理由

韓国語のハングル音節は、**SN**（받침/終声なし）と**SF**（받침あり）の2種類にのみ分類されます。この二値的な区別では、WC特徴量（単語＋文字タイプの組み合わせ）は冗長な情報を生成し、識別力がほとんどありません。これらを除外することで、ノイズを低減し、モデルをコンパクトに保ちます。
