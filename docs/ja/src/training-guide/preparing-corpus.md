# コーパスの準備

良質な学習コーパスは、モデルの精度にとって不可欠です。このガイドでは、コーパスの準備方法を説明します。

## コーパスの形式

コーパスは以下の条件を満たすプレーンテキストファイルである必要があります:
- **1行1文**
- **単語をスペースで区切る**

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 。
Rust で 実装 さ れ た コンパクト な 単語 分割 ソフトウェア です 。
```

## コーパスの自動作成

Litsea には、Wikipedia からコーパスを構築するためのヘルパースクリプトが `scripts/` ディレクトリに用意されています。

### ステップ 1: Wikipedia テキストのダウンロード

```sh
bash scripts/wikitexts.sh ja   # Japanese
bash scripts/wikitexts.sh ko   # Korean
bash scripts/wikitexts.sh zh   # Chinese
```

このスクリプトは以下の処理を行います:
1. Wikipedia API から記事タイトルをダウンロード
2. 言語固有の基準でフィルタリング
3. 記事テキストを取得
4. `litsea split-sentences` を使用して文に分割

### ステップ 2: Lindera によるトークン化

```sh
bash scripts/corpus.sh ja ./wikitexts_ja.txt ./corpus_ja.txt
bash scripts/corpus.sh ko ./wikitexts_ko.txt ./corpus_ko.txt
bash scripts/corpus.sh zh ./wikitexts_zh.txt ./corpus_zh.txt
```

このスクリプトは [Lindera](https://github.com/lindera/lindera) を言語固有の辞書とともに使用します:

| 言語 | 辞書 | 備考 |
|----------|-----------|-------|
| 日本語 | UniDic | 複合語フィルタ付き |
| 韓国語 | ko-dic | 韓国語辞書 |
| 中国語 | CC-CEDICT | 中英辞書 |

出力は **wakati** 形式（スペース区切りのトークン）で、特徴量抽出にそのまま使用できます。

## コーパスの品質に関するヒント

- **多様性** -- さまざまな分野のテキストを含める（ニュース、文学、ウェブなど）
- **データ量** -- 大きなコーパスほど一般的に良いモデルを生成するが、収穫逓減がある
- **一貫性** -- コーパス全体で一貫したトークン化を確保する
- **重複排除** -- 偏りを避けるために重複文を除去する
- **クリーニング** -- HTML タグ、特殊なフォーマット、非テキストコンテンツを除去する
