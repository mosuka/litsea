# はじめに

**Litsea** は、[TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) および [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker) に触発されて開発された、Rust で実装された極めてコンパクトな単語分割ライブラリです。

[MeCab](https://taku910.github.io/mecab/) や [Lindera](https://github.com/lindera/lindera) などの従来の形態素解析器とは異なり、Litsea は大規模な辞書に依存しません。代わりに、**AdaBoost 二値分類**アルゴリズムに基づくコンパクトな学習済みモデルを使用して単語分割を行います。また、**Averaged Perceptron** 多クラス分類器と [Universal POS (UPOS)](https://universaldependencies.org/u/pos/) タグセットを用いた**単語分割と品詞推定の同時実行**もサポートしています。

## 主な特徴

- **高速かつ安全な Rust 実装** -- Rust の安全性保証とパフォーマンスを活用
- **コンパクトな学習済みモデル** -- モデルファイルはわずか数キロバイト
- **辞書不要** -- 統計モデルのみで分割を実行
- **品詞推定（POS Tagging）** -- Averaged Perceptron 多クラス分類により、単語分割と同時に UPOS 品詞タグを推定
- **多言語対応** -- 日本語、中国語（簡体字/繁体字）、韓国語
- **モデル学習機能** -- AdaBoost または Averaged Perceptron を使用して独自のコーパスからカスタムモデルを学習可能
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

### 品詞推定（POS Tagging）

Litsea は単語分割に加えて、**品詞推定**（Part-of-Speech Tagging）もサポートしています。**Averaged Perceptron** 多クラス分類器を使用し、単語分割と品詞推定を同時に（Joint方式で）行います。

各文字位置に対して、18 クラスの **SegmentLabel** を予測します:

- `B-NOUN`, `B-VERB`, ..., `B-X`（17 品詞の境界ラベル）
- `O`（非境界 = 単語の継続）

品詞タグには [Universal Dependencies](https://universaldependencies.org/) の **UPOS タグセット**（17 品詞）を採用しています。

```text
Input:  "今日はいい天気ですね。"
Output: 今日/X は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
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
