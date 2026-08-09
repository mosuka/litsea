# 新しい言語の追加

Litseaの多言語フレームワークは、容易に拡張できるよう設計されています。本ガイドでは、新しい言語のサポートを追加する方法を説明します。

## 手順の概要

1. `Language` 列挙型にバリアントを追加
2. `Display` および `FromStr` のmatchアームを実装
3. 文字タイプパターン関数を作成
4. パターン関数を登録
5. WC特徴量の有無を決定
6. 学習コーパスを用意してモデルを学習
7. テストを追加

## 手順1: `Language` にバリアントを追加

`litsea/src/language.rs` で、`Language` 列挙型に新しいバリアントを追加します。

```rust
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    Thai,       // ← new language
}
```

## 手順2: Display と FromStr を実装

新しい言語のmatchアームを追加します。

```rust
// In Display impl
Language::Thai => write!(f, "thai"),

// In FromStr impl
"thai" | "th" => Ok(Language::Thai),
```

## 手順3: 文字タイプパターンを作成

新しい言語の文字を**種別 ID**（type id）に分類する関数を定義します。ID は言語の順序付き `type_codes()` テーブル（手順 4）へのインデックスです: 共通クラスは固定インデックス（"O" = 0、"P" = 1、"A" = 2、"N" = 3）を占め、言語固有クラスは 4 から続きます。分類は文字範囲に対する `match` 式で直接行います（正規表現は使いません）。**最初にマッチしたアーム**が種別を決定します。

```rust
fn thai_char_type_id(c: char) -> u8 {
    match c {
        // タイ文字の子音・順行母音 (U+0E01-U+0E3A)
        '\u{0E01}'..='\u{0E3A}' => 4, // "T"
        // タイ文字の母音・声調記号 (U+0E40-U+0E4E)
        '\u{0E40}'..='\u{0E4E}' => 5, // "V"
        // タイ数字 (U+0E50-U+0E59)
        '\u{0E50}'..='\u{0E59}' => DIGIT_TYPE_ID, // "N"
        // 共通クラス: "P"（句読点）、"A"（ラテン文字）、"N"（数字）
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
    }
}
```

### 文字タイプ設計のヒント

- 語境界パターンと相関する**言語学的に異なるカテゴリ**を特定する
- **順序は重要** -- 最初にマッチしたものが優先されるため、より具体的なパターンを汎用的なパターンの前に配置する
- 中国語の「F」のように、**高頻度の機能語**を別のタイプとして検討する
- 単純な範囲比較では対応できないロジックには**matchガード**を使用する（韓国語が받침の有無で音節を分割する際に使用しているように）
- 共通の「P」/「A」/「N」クラスには、共有ヘルパー `punct_latin_digit()` を再利用する
- **コード集合は prefix-free に保つ** -- どのコードも他のコードのプレフィックスであってはならない（韓国語の `SN`/`SF` が成立するのは `S` 単独がコードでないため）。モデルローダは packed 特徴キーへのコンパイル時に連結された種別コードを左から右へデコードするため、prefix-free 性がデコードの一意性を保証する（言語ごとにユニットテストで固定される）

## 手順4: 種別コードテーブルと分類関数を登録

言語の順序付きコードテーブルを `Language::type_codes()` に（インデックス = 種別 ID、共通コードが先頭）、ディスパッチアームを `Language::char_type_id()` に追加します。`char_type()` 自体はこの 2 つから導出されるため、文字列コードと数値 ID が乖離することはありません。

```rust
pub(crate) fn type_codes(self) -> &'static [&'static str] {
    match self {
        // ...
        Language::Thai => &["O", "P", "A", "N", "T", "V"],    // ← new
    }
}

pub(crate) fn char_type_id(self, c: char) -> u8 {
    match self {
        // ...
        Language::Thai => thai_char_type_id(c),    // ← new
    }
}
```

## 手順5: WC特徴量の有無を決定

特徴テンプレートは `packed_model.rs`（`TEMPLATES`）に一度だけ定義されており、`templates_for()` が末尾の `WC1`--`WC4`（文字/種別混合テンプレート）を言語が使用するかどうかを決定します。

```rust
pub(crate) fn templates_for(language: Language) -> &'static [Template] {
    match language {
        Language::Japanese | Language::Chinese => &TEMPLATES[..],
        _ => &TEMPLATES[..BASE_TEMPLATE_COUNT], // 38 個の基本テンプレート
    }
}
```

対象言語の文字タイプに十分な多様性があり、WC特徴量が有益である場合は、最初のmatchアームに追加してください。韓国語のSN/SFのようにタイプ体系が低エントロピーの場合は、WC特徴量を除外する方が適切です。

## 手順6: コーパスを用意してモデルを学習

1. **コーパスを用意**します（単語をスペースで区切った形式）。

   ```text
   word1 word2 word3 word4
   ```

2. **特徴量を抽出**します。

   ```sh
   litsea extract -l thai ./corpus.txt ./features.txt
   ```

3. **モデルを学習**します。

   ```sh
   litsea train -t 0.005 -i 1000 ./features.txt ./models/thai.model
   ```

## 手順7: テストを追加

`language.rs` と `segmenter.rs` の両方にテストを追加します。

```rust
// In language.rs tests
#[test]
fn test_thai_char_types() {
    let lang = Language::Thai;
    assert_eq!(lang.char_type('ก'), "T");   // Thai consonant
    assert_eq!(lang.char_type('A'), "A");   // ASCII
    assert_eq!(lang.char_type('@'), "O");   // Other
}

// In segmenter.rs tests
#[test]
fn test_char_type_thai() {
    let segmenter = Segmenter::new(Language::Thai);
    assert_eq!(segmenter.char_type("ก"), "T");
}
```

全テストを実行して検証します。

```sh
cargo test --workspace
```
