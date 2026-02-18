# はじめに

**Litsea** は、[TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) および [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker) に触発されて開発された、Rust で実装された極めてコンパクトな単語分割ライブラリです。

[MeCab](https://taku910.github.io/mecab/) や [Lindera](https://github.com/lindera/lindera) などの従来の形態素解析器とは異なり、Litsea は大規模な辞書に依存しません。代わりに、**AdaBoost 二値分類**アルゴリズムに基づくコンパクトな学習済みモデルを使用して単語分割を行います。

## 主な特徴

- **高速かつ安全な Rust 実装** -- Rust の安全性保証とパフォーマンスを活用
- **コンパクトな学習済みモデル** -- モデルファイルはわずか数キロバイト
- **辞書不要** -- 統計モデルのみで分割を実行
- **多言語対応** -- 日本語、中国語（簡体字/繁体字）、韓国語
- **モデル学習機能** -- AdaBoost を使用して独自のコーパスからカスタムモデルを学習可能
- **リモートモデル読み込み** -- HTTP/HTTPS URL またはローカルファイルからモデルを読み込み
- **シンプルで拡張性の高い API** -- Rust プロジェクトへのライブラリとしての統合が容易

## 仕組み

Litsea は単語分割を**二値分類問題**として扱います。文中の各文字位置について、モデルがその位置が**単語境界**（+1）か**非境界**（-1）かを予測します。分類器は、各言語固有の文字 n-gram 特徴量と文字種情報を使用します。

```text
Input:  "LitseaはRust製です"
         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
         O O O O B O B O B   ← boundary predictions
Output: ["Litsea", "は", "Rust製", "です"]
```

## 名前の由来

クスノキ科には *Lindera*（クロモジ）と同じ科に属する *Litsea cubeba*（アオモジ）という小さな植物があります。これが **Litsea** という名前の由来です。

## 現在のバージョン

Litsea v0.4.0 -- Rust Edition 2024、最低 Rust バージョン 1.87。

## リンク

- [GitHub リポジトリ](https://github.com/mosuka/litsea)
- [crates.io](https://crates.io/crates/litsea)
- [API ドキュメント (docs.rs)](https://docs.rs/litsea)
- [English Documentation](../../)
