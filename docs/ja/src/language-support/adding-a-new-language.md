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

新しい言語の `CharTypePatterns` を返す関数を定義します。

```rust
fn thai_patterns() -> CharTypePatterns {
    CharTypePatterns::from_matchers(vec![
        // Thai characters (U+0E01-U+0E3A)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E01}-\u{0E3A}]").unwrap()
        ), "T"),
        // Thai vowels (U+0E40-U+0E4E)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E40}-\u{0E4E}]").unwrap()
        ), "V"),
        // Thai digits (U+0E50-U+0E59)
        (CharMatcher::Regex(
            Regex::new(r"[\u{0E50}-\u{0E59}]").unwrap()
        ), "N"),
        // ASCII + Full-width Latin
        (CharMatcher::Regex(
            Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap()
        ), "A"),
        // Digits
        (CharMatcher::Regex(
            Regex::new(r"[0-9０-９]").unwrap()
        ), "N"),
    ])
}
```

### 文字タイプ設計のヒント

- 語境界パターンと相関する**言語学的に異なるカテゴリ**を特定する
- **順序は重要** -- 最初にマッチしたものが優先されるため、より具体的なパターンを汎用的なパターンの前に配置する
- 中国語の「F」のように、**高頻度の機能語**を別のタイプとして検討する
- 単一の正規表現では表現できない複雑なロジックには**クロージャ**を使用する

## 手順4: パターン関数を登録

`Language::char_type_patterns()` にmatchアームを追加します。

```rust
pub fn char_type_patterns(&self) -> CharTypePatterns {
    match self {
        Language::Japanese => japanese_patterns(),
        Language::Chinese => chinese_patterns(),
        Language::Korean => korean_patterns(),
        Language::Thai => thai_patterns(),    // ← new
    }
}
```

## 手順5: WC特徴量の有無を決定

`segmenter.rs` の `get_attributes()` では、WC特徴量を含めるかどうかを言語に基づいて `match` で判定しています。

```rust
match self.language {
    Language::Japanese | Language::Chinese => {
        // Include WC features
        attrs.insert(format!("WC1:{}{}", w3, c4));
        attrs.insert(format!("WC2:{}{}", c3, w4));
        attrs.insert(format!("WC3:{}{}", w3, c3));
        attrs.insert(format!("WC4:{}{}", w4, c4));
    }
    _ => {}
}
```

対象言語の文字タイプに十分な多様性があり、WC特徴量が有益である場合は、matchアームに追加してください。韓国語のSN/SFのようにタイプ体系が低エントロピーの場合は、WC特徴量を除外する方が適切です。

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
   litsea train -t 0.005 -i 1000 ./features.txt ./resources/thai.model
   ```

## 手順7: テストを追加

`language.rs` と `segmenter.rs` の両方にテストを追加します。

```rust
// In language.rs tests
#[test]
fn test_thai_patterns() {
    let p = Language::Thai.char_type_patterns();
    assert_eq!(p.get_type("ก"), "T");   // Thai consonant
    assert_eq!(p.get_type("A"), "A");   // ASCII
    assert_eq!(p.get_type("@"), "O");   // Other
}

// In segmenter.rs tests
#[test]
fn test_get_type_thai() {
    let segmenter = Segmenter::new(Language::Thai, None);
    assert_eq!(segmenter.get_type("ก"), "T");
}
```

全テストを実行して検証します。

```sh
cargo test --workspace
```
