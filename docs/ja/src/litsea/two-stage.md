# 二段構成モデル（Two-Stage Model）

`two_stage` モジュールは `litsea-two-stage v1` モデルコンテナ
（`TwoStageLearner`）、stage-2 特徴量セットの選択肢（`TwoStageFeatureSet`）、
モデル種別の判定器（`ModelKind`）を定義します。
アーキテクチャと計測済みの品質・速度の数値は
[二段構成タグ付け](../algorithm/two-stage-tagging.md) を、
モデルの新規学習は
[`TwoStageTrainer`](trainer.md#twostagetrainer) を参照してください。

## `TwoStageLearner`

```rust
pub struct TwoStageLearner {
    // private: stage1: AdaBoost,
    // private: stage2: AveragedPerceptron,
    // private: lexicon: HashMap<String, Vec<(Upos, u32)>>,
    // private: dominance: f64,
}
```

二段構成モデルの 3 パーツを保持します: stage-1 境界分類器
（スカラー重み、AdaBoost 形式）、候補タグ語彙表、stage-2 単語レベル
タガー（`AveragedPerceptron`）。構築・シリアライズの規約は
`AdaBoost` / `AveragedPerceptron` と同じです。

### コンストラクタ

```rust
pub fn new() -> Self
pub fn from_parts(
    stage1: AdaBoost,
    stage2: AveragedPerceptron,
    lexicon: impl IntoIterator<Item = (String, Vec<(Upos, u32)>)>,
    dominance: f64,
) -> Result<Self>
```

`new` は空の learner を作成します（使用前に `load_model*` 系メソッドで
埋める必要があります）。`from_parts` はパーツから learner を構築し、
組み合わせを検証します: `dominance` は `(0.5, 1.0]` の範囲内、
stage-2 の各クラス名は有効な `Upos` タグである必要があり、
各語彙エントリは非空のサーフェス（タブ・改行を含まない）、
正のカウントを持つ非空のタグリスト、重複タグ・重複サーフェスが
無いことが要求されます。語彙エントリは入力順に関わらず正規順序
（カウント降順、同数はタグ名昇順）に正規化されます。

### モデルの入出力

```rust
pub fn save_model(&self, path: &Path) -> Result<()>
pub fn save_model_to_writer<W: Write>(&self, writer: &mut W) -> Result<()>
pub async fn load_model(&mut self, uri: &str) -> Result<()>
pub fn load_model_from_path(&mut self, path: &Path) -> Result<()>
pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()>
```

`AdaBoost`/`AveragedPerceptron` と同じ規約です: `save_model`/`load_model`
はファイルパス、または（`load_model` のみ）`file://`/`http(s)://` の URI
に対応します（後者は `remote_model` フィーチャが必要）。
`*_to_writer`/`*_from_reader` 系は任意の writer/reader に対応します。
空の learner（語彙エントリが無い、またはどちらかの内部 learner が
未学習）の保存は `LitseaError::InvalidInput` を返します。
読み込みはファイル構造全体を検証します（オンディスクのレイアウトは
[モデルファイル形式](../advanced/model-file-format.md) を参照）。
不正な内容は `LitseaError::InvalidData` で拒否され、読み込みエラー時に
learner は変更されません。

```rust
use std::path::Path;

use litsea::two_stage::TwoStageLearner;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_two_stage.model"))?;
```

### アクセサ

```rust
pub fn stage1(&self) -> &AdaBoost
pub fn stage2(&self) -> &AveragedPerceptron
pub fn dominance(&self) -> f64
pub fn lexicon_len(&self) -> usize
pub fn lexicon_entry(&self, surface: &str) -> Option<&[(Upos, u32)]>
```

`dominance` は分類器スキップの閾値です: 推論時、あるサーフェスの
最頻タグが学習時出現の少なくともこの割合を占める場合、stage-2
分類器を一切呼び出さずにタグ付けします。`lexicon_entry` は
学習中に観測されたサーフェスの候補タグを頻度降順で返し、
一度も観測されていないサーフェスには `None` を返します。

実際に推論を実行するには、`TwoStageLearner` を直接呼び出すのではなく
[`Segmenter::with_two_stage_learner`](segmenter.md#with_two_stage_learner)
経由で `Segmenter` にインストールしてください——segmenter がこれを
高速ルックアップ用の packed スコアリングテーブルへコンパイルします。

## `TwoStageFeatureSet`

```rust
#[non_exhaustive]
pub enum TwoStageFeatureSet {
    Full,
    Balanced,
    #[default]
    Fast,
}
```

[`Extractor::extract_two_stage`](extractor.md#extract_two_stage) が書き出す
23 個の単語レベル stage-2 テンプレート（[特徴量抽出](../algorithm/feature-extraction.md)
参照）のうちどれを使うかを選択します。`Fast`（既定値）は計測済みの
最小セット——サーフェス、単語長、先頭/末尾文字タイプ、隣接文脈文字＋
タイプ、2文字プレフィックス/サフィックス。`Balanced` はこれに
先頭/末尾文字そのものと単語タイプコード文字列を追加します。`Full` は
全テンプレートを含みます。分割精度は 3 セットとも同一です（stage-1 が
決定するため）——変わるのはタグ付け精度とスループットのみです。
3 セットの相対的な順序（正確な数値ではありません——この型自身の
rustdoc に記載の数値は、同梱モデルとは異なるエポック数で計測した
初期プロトタイプのものです）についてはこの型自身の rustdoc を、
同梱モデルの現在の実測値は [学習済みモデル](../pre-trained-models.md)
を参照してください。

`FromStr`（大文字小文字を区別しない: `"full"`, `"balanced"`, `"fast"`）と
`Display`（小文字）を実装しています。`#[non_exhaustive]` が付与されて
おり、外部の `match` 式にはワイルドカードアームが必要です。

## `ModelKind`

```rust
pub enum ModelKind {
    AdaBoost,
    AveragedPerceptron,
    TwoStage,
}
```

`ModelKind::detect(content: &str) -> ModelKind` はモデルファイルの
1行目を調べて形式を判別します——これは判別のためのヒューリスティックで
あり完全な検証ではありません。`AdaBoost` はプレーンな分割モデル形式
（畳み込み済みの二段構成 stage 1 の形式でもあります）、
`AveragedPerceptron` はスタンドアロンのパーセプトロンファイル
（`train --perceptron` の出力であり、`[stage2]` セクションのペイロード
形式。これは削除された joint POS モデル形式であり、POS モデルとしては
読み込めません）、`TwoStage` は `litsea-two-stage` コンテナです。

種類違いのファイルには `TwoStageLearner` のローダーが正確なエラーを
返します: スタンドアロンの Averaged Perceptron ファイルを指定すると
"joint POS models are no longer supported — retrain with `litsea train
--two-stage`" で失敗し、それ以外の非二段構成の内容はマジック行の欠落
エラーで失敗します。
