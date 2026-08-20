# はじめに

**Litsea** は、[TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) および [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker) に触発されて開発された、Rust で実装された極めてコンパクトな単語分割ライブラリです。

[MeCab](https://taku910.github.io/mecab/) や [Lindera](https://github.com/lindera/lindera) などの従来の形態素解析器とは異なり、Litsea は大規模な辞書に依存しません。代わりに、**AdaBoost 二値分類**アルゴリズムに基づくコンパクトな学習済みモデルを使用して単語分割を行います。また、二段構成（two-stage）アーキテクチャにより、[Universal POS (UPOS)](https://universaldependencies.org/u/pos/) タグセットを用いた**単語分割と品詞推定（POS Tagging）**もサポートしています。

## 主な特徴

- **高速かつ安全な Rust 実装** -- Rust の安全性保証とパフォーマンスを活用
- **コンパクトな学習済みモデル** -- レガシーな `RWCP.model` / `JEITA_Genpaku_ChaSen_IPAdic.model` はキロバイト級。品質を最適化した `japanese`/`chinese`/`korean.model` は約 86 KB〜2.0 MB で、アプリケーションへの直接埋め込みや HTTP 経由の配信に十分な小ささ
- **辞書不要** -- 統計モデルのみで分割を実行
- **二段構成の品詞推定** -- 二値境界分類器での分割と、候補タグ語彙表 + 単語単位タガーによる各単語のタグ付けを組み合わせ、通常の単語分割にわずかなコストしか追加しない
- **多言語対応** -- 日本語、中国語（簡体字/繁体字）、韓国語
- **モデル学習機能** -- AdaBoost または Averaged Perceptron を使用して独自のコーパスからカスタムモデルを学習可能
- **リモートモデル読み込み** -- HTTP/HTTPS URL（オプトインの `remote_model` フィーチャー）またはローカルファイルからモデルを読み込み
- **シンプルで拡張性の高い API** -- Rust プロジェクトへのライブラリとしての統合が容易

## 仕組み

Litsea は単語分割を**二値分類問題**として扱います。文中の各文字位置について、モデルがその位置が**単語境界**（+1）か**非境界**（-1）かを予測します。分類器は、各言語固有の文字 n-gram 特徴量と文字種情報を使用します。

```text
Input:  "これはテストです。"
         こ れ は テ ス ト で す 。
         B  O  B  B  O  O  B  O  B   ← word-start predictions (RWCP.model)
Output: ["これ", "は", "テスト", "です", "。"]
```

### 品詞推定（POS Tagging）

Litsea は単語分割に加えて、**品詞推定**（Part-of-Speech Tagging）もサポートしています。二段構成アーキテクチャにより、まず二値境界分類器で文を分割し、次に候補タグ語彙表 + 単語単位タガーで各単語にタグを付与します。

各文字位置に対して、18 クラスの **SegmentLabel** を予測します:

- `B-NOUN`, `B-VERB`, ..., `B-X`（17 品詞の境界ラベル）
- `O`（非境界 = 単語の継続）

品詞タグには [Universal Dependencies](https://universaldependencies.org/) の **UPOS タグセット**（17 品詞）を採用しています。

```text
Input:  "今日はいい天気ですね。"
Output: 今日/NOUN は/ADP いい/ADJ 天気/NOUN です/AUX ね/PART 。/PUNCT
```

## 名前の由来

クスノキ科には *Lindera*（クロモジ）と同じ科に属する *Litsea cubeba*（アオモジ）という小さな植物があります。これが **Litsea** という名前の由来です。

## 現在のバージョン

Litsea v0.12.0 -- Rust Edition 2024、最低 Rust バージョン 1.87。

## リンク

- [GitHub リポジトリ](https://github.com/mosuka/litsea)
- [crates.io](https://crates.io/crates/litsea)
- [API ドキュメント (docs.rs)](https://docs.rs/litsea)
- [English Documentation](../../)
