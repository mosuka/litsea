# Litsea コードベース リファクタリング計画

作成日: 2026-06-12
対象バージョン: 0.4.0(workspace: `litsea` / `litsea-cli`、約 4,200 行)

本計画は、コードベース全体の調査に基づき、蓄積した重複・設計上の負債・性能上の無駄を
**6 フェーズ**に分けて解消するためのものです。各フェーズは独立してレビュー・マージ可能な
単位とし、「挙動を変えないフェーズ」と「意図的に API を変えるフェーズ」を明確に分離します。

---

## 1. 現状調査サマリ

### 1.1 確認済みのバグ(最優先)

| # | 内容 | 根拠 |
|---|------|------|
| B1 | ✅ **修正済み(Phase 0 PR に前倒しで含めた)** — **`--no-default-features` でビルド不能**。`perceptron.rs` が `reqwest` を feature gate なしで使用しており(`perceptron.rs:9` の `use reqwest::Client;` と `load_model_from_url`)、`remote_model` を無効にするとコンパイルエラーになる。`adaboost.rs` は正しく `#[cfg(feature = "remote_model")]` で保護されているのに対し、後から追加された perceptron 側が追従していない。 | `cargo check -p litsea --no-default-features` が E0432/E0282 で失敗することを確認済み |
| B2 | ✅ **修正済み(Phase 1)** — `AveragedPerceptron::save_model` の doc コメントが「モデルを**JSON形式**でファイルに保存する」と記述しているが、実際はクラスヘッダ + TSV のテキスト形式(`perceptron.rs:231-242`)。 | コード読解 |
| B3 | ✅ **解消済み(Phase 0)** — CI(regression.yml / periodic.yml)に feature マトリクスがなく、B1 のようなビルド破壊が検出されない。 | ワークフロー確認 |
| B4 | **増分学習時の特徴量インデックス不整合(潜在バグ、未修正)**。`Trainer::new`(`initialize_features` + `initialize_instances`)の後に `load_model` を呼ぶと(CLI の `train -m` パス)、`parse_model_content` が `features`/`feature_index` をモデルファイルの内容(非ゼロ重みの特徴のみ)で再構築するため、`instances_buf` に格納済みの特徴インデックスが古い索引を指したまま `train()` が走る。Phase 3 のエラー型・API 再設計時に修正予定。 | Phase 2 作業中のコード読解で発見 |

### 1.2 重複コード(本計画の主対象)

| # | 内容 | 場所 | 規模 |
|---|------|------|------|
| D1 | **モデル I/O の全面重複**。URI スキーム解決・HTTP ダウンロード・wasm32 向け cfg 分岐・ファイル読み込みのロジックが `AdaBoost` と `AveragedPerceptron` でほぼコピペ。 | `adaboost.rs:334-524` / `perceptron.rs:271-438` | 約 170 行 × 2 |
| D2 | **Segmenter のパイプライン 4 重複**。`B3/B2/B1`・`E1/E2/E3` パディングと chars/types 配列の構築が `process_corpus`、`process_corpus_with_pos`、`segment`、`segment_with_pos` の 4 箇所に展開されている。 | `segmenter.rs:86-128, 134-201, 328-361, 378-432` | 約 30 行 × 4 |
| D3 | **Extractor の 2 メソッド重複**。`extract` と `extract_with_pos` はラベル型(`i8` / `SegmentLabel`)以外ほぼ同一で、`RefCell<Option<io::Error>>` によるエラー伝播ハックも両方に存在。 | `extractor.rs:51-153` | 約 50 行 × 2 |
| D4 | **Metrics 構造体が 2 つ**(`adaboost::Metrics` と `perceptron::Metrics`)。名前が衝突するため `trainer.rs` では `perceptron::Metrics` とフルパス参照している。 | `adaboost.rs:14-32` / `perceptron.rs:34-50` | 小 |
| D5 | **Trainer / PosTrainer の構造重複**。`new → load_model → train(保存込み)` の同型ラッパーが 2 つ。 | `trainer.rs:15-148` | 約 60 行 × 2 |
| D6 | **言語別パターンの共通部分重複**。句読点・Latin・数字の正規表現が日本語/中国語/韓国語の 3 関数にコピペ。`expect("hardcoded regex pattern is valid")` も全行に重複。 | `language.rs:138-302` | 中 |
| D7 | エラー生成の `std::io::Error::new(InvalidData, format!(...))` ボイラープレートが全ファイルに散在(20 箇所以上)。 | 全域 | 中 |

### 1.3 設計・API 上の負債

- **エラー型がない**: ライブラリ全体が `std::io::Error` の流用と `Box<dyn Error>`(`trainer.rs`、`extractor.rs`)の混在。呼び出し側がエラー種別で分岐できない。
- **async の伝播**: `load_model` だけが async なため、CLI 全体が tokio 必須、doc テストが `tokio_test::block_on` 必須、ベンチでも `tokio::runtime::Runtime` を手作り(`bench.rs:14-30`)。実際に非同期が必要なのは `remote_model` の HTTP 取得のみ。
- **同一ファイルの 2 回読み**: `Trainer::new` が `initialize_features` と `initialize_instances` で同じ特徴量ファイルを 2 回パースする(`trainer.rs:46-47`、`adaboost.rs:95-203`)。
- **API の非一貫性**: `AdaBoost::predict(HashSet<String>)` は所有権を取るが `AveragedPerceptron::predict(&HashSet<String>)` は借用。`add_instance` のラベル型も `i8` vs `String`。
- **内部状態の露出**: `Segmenter::learner` / `pos_learner` が pub フィールド。`lib.rs` に re-export がなく、利用者は `litsea::segmenter::Segmenter` のようにモジュールパスを辿る必要がある。
- **`segment_with_pos` の先頭単語処理が不自然**: ループ後に「結果が空の場合だけ」先頭位置を再予測する後付けロジック(`segmenter.rs:420-428`)。先頭単語の品詞はループ内の最初の予測から自然に決まる構造にすべき。
- **`util` モジュールの形骸化**: 中身は `ModelScheme` のみ。モデル I/O 共通化の際に吸収すべき。
- **doc コメントの日英混在**: `adaboost.rs`・`language.rs` は英語、`perceptron.rs`・`upos.rs` は日本語、`segmenter.rs`・`trainer.rs` は混在。crates.io に公開されるライブラリとして不統一。

### 1.4 パフォーマンス上の無駄

- **文字種分類が正規表現の線形走査**: 1 文字ごとに最大 9 個の `Regex::is_match` を順に試す(`language.rs`)。しかも呼び出し側で `char` → `String` 変換してから渡している(`segmenter.rs:108, 341` の `ch.to_string()`)。`char` の数値範囲 `match` にすれば正規表現自体が不要。
- **特徴量生成のアロケーション過多**: `get_attributes` が 1 文字位置あたり約 40 個の `String` を `format!` で生成し `HashSet` に格納(`segmenter.rs:453-532`)。セグメント処理のホットパス。
- **内部表現がすべて `String`**: `chars: Vec<String>`(1 文字ずつ!)、`tags: Vec<String>`("B"/"O"/"U" の 3 値)、`types: Vec<String>`(数種の固定コード)。enum / `char` / `&'static str` で十分。
- **AveragedPerceptron の学習ループ**: epoch ごとに `self.instances` を丸ごと clone し(`perceptron.rs:208`)、インスタンスごとに `Vec<String>` → `HashSet<String>` を再構築(`perceptron.rs:220`)。重み構造も `HashMap<String, HashMap<String, f64>>` の二重ハッシュ。

### 1.5 その他(体裁・周辺)

- `rustfmt.toml` の大半がコメントアウト行で占められている。
- doc コメントの様式が不統一(`# Returns: ...` インライン型と `# Returns` セクション型の混在)。
- CLI の `--language` が生 `String` 受け → 手動 `parse`(clap の `ValueEnum` 未使用)。
- ベンチ・doc テストが tokio に不必要に依存(async 整理で解消)。

---

## 2. リファクタリング方針(全フェーズ共通の原則)

1. **モデルファイルの後方互換を維持する**。`models/` 配下の 8 モデル(AdaBoost テキスト形式・Perceptron TSV 形式)は再学習なしでロードできること。フォーマット変更は本計画のスコープ外。
2. **フェーズ 0 で固定したゴールデンテストを全フェーズの合格基準にする**。挙動変更が必要な場合(例: `segment_with_pos` の先頭単語処理)は、変更内容を PR で明記してゴールデンを更新する。
3. **「挙動不変フェーズ」(0〜2, 4)と「API 変更フェーズ」(3)を分離**し、破壊的変更は v0.5.0 として一括リリースする。
4. 各フェーズの完了条件: `cargo fmt --check` / `cargo clippy --workspace --all-targets -- -D warnings` / `cargo test --workspace` / `cargo check -p litsea --no-default-features`(Phase 1 以降)/ ゴールデンテスト / criterion ベースライン比較(性能フェーズ)。
5. 1 フェーズ = 1 PR を基本とし、Phase 3・4 は内容ごとに複数 PR に分割してよい。

---

## 3. フェーズ計画

### フェーズ 0: 安全網の整備(挙動不変・追加のみ) — ✅ 実施済み

**目的**: 以降のすべてのリファクタリングの回帰を機械的に検出できる状態を作る。

**実施結果**(2026-06-12):
- `litsea/tests/golden.rs` を追加(全 8 モデルのスナップショット 10 テスト、すべて green)。
- round-trip テスト(AdaBoost / Perceptron)を同ファイルに追加。
- regression.yml に `feature-check` ジョブを追加(`--no-default-features` と wasm32)。B1 修正も本 PR に前倒しで含めたため、当初予定と異なり最初から必須ジョブとした。
- criterion ベースライン(`--save-baseline pre-refactor`、短縮計測の参考値、median):
  `segment_short/adaboost` ja 56.8µs / zh 37.1µs / ko 51.3µs、
  `segment_short/averaged_perceptron` ja 293µs / zh 211µs / ko 286µs、
  `segment_long_japanese` adaboost 611ms / perceptron **4.48s**、
  `add_corpus` 161µs、`predict_adaboost` 2.9µs、`get_type_hiragana` 61ns、
  `char_type_patterns_japanese`(パターン構築)205µs。
  ※ コンテナ環境は使い捨てのため、Phase 4 では比較対象の数値を都度取り直すこと。

**作業項目**:
1. **ゴールデンテストの追加**(`litsea/tests/golden.rs`):
   - `models/` の各モデル(ja/zh/ko × segment / segment_with_pos)について、代表文 5〜10 文の分割結果をスナップショットとして固定。
   - `JEITA_Genpaku_ChaSen_IPAdic.model`・`RWCP.model` も最低 1 ケース。
2. **モデルファイル round-trip テスト**: `load_model → save_model → load_model` で予測結果が一致することを検証(フォーマット互換の監視)。
3. **criterion ベースラインの記録**: `cargo bench -- --save-baseline pre-refactor` を実行し、結果を PR に記録(数値はリポジトリにはコミットしない)。
4. **CI 強化**(regression.yml):
   - `cargo check -p litsea --no-default-features` ジョブを追加。**現状は B1 により失敗するため、Phase 1 と同一 PR で導入するか、Phase 1 マージまで `continue-on-error` とする。**
   - wasm32 ターゲットの `cargo check`(`--target wasm32-unknown-unknown --no-default-features`)も追加候補(現状 cfg 分岐があるのに検証されていない)。

**完了条件**: 新規テストが green、CI にフィーチャーチェックが入る。
**リスク**: ほぼなし(追加のみ)。
**規模目安**: 小(テスト +200〜300 行、CI 数十行)。

---

### フェーズ 1: バグ修正とモデル I/O の共通化(挙動不変) — ✅ 実施済み

**目的**: 確認済みバグの解消と、最大の重複(D1)の排除。

**実施結果**(2026-06-12):
- 内部モジュール `litsea/src/model_io.rs` を新設し、URI 解決・`file://`・HTTP(S) ダウンロード・wasm32 分岐を `read_model_bytes()` に一本化。`AdaBoost::load_model` / `AveragedPerceptron::load_model` は「バイト列取得 + `parse_model_content`」の 2 行に縮退(両学習器あわせて約 340 行の重複を削除、net -226 行 + 新モジュール約 130 行)。
- 当初計画からの変更点: `util::ModelScheme` の移動と `util.rs` 廃止は**公開 API の互換維持のため Phase 3 に延期**(`model_io` は `util::ModelScheme` を利用する内部モジュールとした)。エラー生成ヘルパも Phase 3 のエラー型導入に統合する形で見送り。
- 副次効果: wasm32 + `--no-default-features` ビルドで出ていた到達不能コード警告 3 件が解消。
- `model_io` に単体テスト 4 件を追加(プレーンパス / `file://` / 不正スキーム / 存在しないファイル)。

**作業項目**:
1. ✅ **B1 修正**(Phase 0 PR で実施済み): `perceptron.rs` の `reqwest` 利用を `adaboost.rs:358-372` と同じ構造で `#[cfg(feature = "remote_model")]` ゲートに統一。
2. ✅ **モデル I/O 共通モジュール `litsea/src/model_io.rs` の新設**:
   - `util::ModelScheme` を移動し、`util.rs` を廃止。
   - URI 解決 → 読み取りソース取得を一本化する関数を提供:
     ```rust
     // 概形。詳細はフェーズ内で確定する。
     pub(crate) async fn read_model(uri: &str) -> io::Result<Vec<u8>>;
     // 内部に: ファイルパス解決 / file:// / http(s):// (remote_model 時のみ) / wasm32 cfg 分岐
     ```
   - `AdaBoost` / `AveragedPerceptron` は各自の `parse_model_content` のみ保持し、
     `load_model` は `read_model` + `parse_model_content` の 2 行に縮退させる。
   - HTTP クライアント生成(User-Agent 付与)も同モジュールへ移動。
   - これにより `adaboost.rs:334-524` / `perceptron.rs:271-438` の重複約 340 行を 1 実装に集約。
3. ✅ **B2 修正**: `save_model` の doc コメントを実フォーマット(ヘッダ + TSV)の記述に修正。
4. ⏭ **エラー生成ヘルパ**(Phase 3 へ統合): `invalid_data(format!(...))` 的な内部ヘルパを `model_io.rs` に置き、散在する `io::Error::new(InvalidData, ...)` ボイラープレートを削減(完全なエラー型再設計は Phase 3 に送る)。
5. ✅ Phase 0 で `continue-on-error` にした feature チェック CI を必須化(B1 修正と同時に Phase 0 PR で実施済み)。

**完了条件**: 全テスト・ゴールデン green、`--no-default-features` / `remote_model` 双方でビルド成功、重複 I/O コードの消滅。
**リスク**: 低。I/O 経路の挙動はゴールデン + round-trip テストで担保。
**規模目安**: 中(差分 ±400 行程度、純減)。

---

### フェーズ 2: Segmenter / Extractor / Trainer の構造統一 — ✅ 実施済み

**目的**: D2・D3・D5 の重複排除と、コールバック設計の歪み(RefCell ハック)の解消。

**実施結果**(2026-06-12):
- `sentence_context()`(パディング付き chars/types 構築)と `process_tokens()`(コーパス処理の共通パイプライン)を導入し、4 箇所の重複を 1 箇所に集約。`process_corpus` / `process_corpus_with_pos` は数行の薄いラッパーに縮退。
- **意図的な挙動変更**: `segment_with_pos` の先頭単語の品詞を、先頭文字位置の予測ラベルから決定するよう修正(従来は常に X、1 単語文では E1 センチネル上で予測していた)。全テスト文で品詞が改善(これ→PRON、東京→PROPN、字→NOUN(旧 SYM)、好→ADJ(旧 PUNCT)等)。ゴールデンテストの期待値を更新済み。分割位置は不変。
- Extractor: `extract` / `extract_with_pos` を `write_features` + `format_row` の共通実装に統合し、`RefCell<Option<io::Error>>` ハックを撤廃(行を整形してからループ外で `?` 伝播)。
- 当初計画からの変更点:
  - `segment` / `segment_with_pos` の走査ループの完全統合は**見送り**(境界判定の型・所有権の契約が複雑になり可読性が下がるため、共通コンテキスト上の 2 つの薄いループとして保持)。
  - AdaBoost の特徴量ファイル 2 回読みの 1 パス化は**取り止め**。OS のページキャッシュで 2 回目の読み込みは安価であり、全体をメモリに載せる方式は巨大コーパスでメモリを浪費、逐次インターン方式は特徴順序が変わり学習結果の再現性を壊すため、コストに見合わない。
  - Trainer / PosTrainer の統合は公開 API 変更を伴うため Phase 3 へ。
- 副産物: 増分学習パスの潜在バグ B4 を発見(1.1 節参照)。

**作業項目**:
1. **文コンテキスト構築の一本化**(`segmenter.rs`):
   - パディング付き chars/types 配列の構築を担う内部型を導入:
     ```rust
     struct SentenceContext { chars: Vec<String>, types: Vec<String> }
     impl SentenceContext {
         fn from_sentence(seg: &Segmenter, s: &str) -> Self;       // segment 系
         fn from_tokens(seg: &Segmenter, tokens: ...) -> Self;     // corpus 系
         fn positions(&self) -> Range<usize>;                      // 4..len-3
     }
     ```
   - 現在 4 箇所にある `B3/B2/B1` / `E1/E2/E3` パディング構築をこの型に集約。
     (`String` → `char`/enum 化は Phase 4 で行い、ここでは構造だけ直す)
2. **コーパス処理の統合**: `process_corpus` と `process_corpus_with_pos` を、ラベル生成だけが異なる単一実装に統合。`i8` ラベルは `SegmentLabel` から導出(`B(_) → 1`, `O → -1`)する変換を挟み、ラベル付けロジックを 1 箇所にする。
3. **推論の統合**: `segment` と `segment_with_pos` の走査ループを共通化(境界判定を返すクロージャ/内部 trait で差し替え)。あわせて `segment_with_pos` の先頭単語品詞の後付けロジック(`segmenter.rs:420-428`)を「最初の予測位置で先頭単語の品詞を決める」素直な構造に整理する。**ゴールデンテストで出力差を確認し、差が出る場合は PR に明記**。
4. **Extractor の統合と RefCell 排除**(`extractor.rs`):
   - コールバックを `FnMut(...) -> io::Result<()>` に変更し(または特徴量行のイテレータを返す設計にし)、`RefCell<Option<io::Error>>` による帯域外エラー伝播を廃止。
   - `extract` / `extract_with_pos` は「ラベルの文字列化」だけが異なる単一実装に統合。
5. **Trainer / PosTrainer の整理**(`trainer.rs`):
   - `AdaBoost::initialize_features` + `initialize_instances` の 2 回ファイル読みを、1 パスで両方を構築する `AdaBoost::load_training_data(path)`(仮)に置き換え。
   - 2 つの Trainer は共通の流れ(`load → train → save → metrics`)を持つため、薄い共通ヘルパに寄せるか、いっそ CLI 側へ吸収するかを判断(公開 API 変更を伴う場合は Phase 3 で実施)。

**完了条件**: 全テスト・ゴールデン green。`segmenter.rs` のパディング構築が 1 箇所、`extractor.rs` から `RefCell` が消える。
**リスク**: 中。境界条件(`tags[3] = "U"` の扱い、`labels` のインデックスずれ等)が複雑なため、Phase 0 のゴールデンに先頭 1 文字・1 単語文・空白連続などのエッジケースを必ず含めておく。
**規模目安**: 中〜大(`segmenter.rs` を中心に ±600 行程度、純減)。

---

### フェーズ 3: エラー型と公開 API の再設計(破壊的変更 → v0.5.0) — ✅ 実施済み

**目的**: 利用者向け API の一貫性確保。破壊的変更をこのフェーズに集約する。

**実施結果**(2026-06-12):
- `thiserror` ベースの `LitseaError`(`Io` / `InvalidData` / `InvalidInput` / `Unsupported` / `Download`)と `litsea::Result<T>` を導入し、`io::Error` 流用と `Box<dyn Error>` を全廃。
- 同期 API `load_model_from_path` / `load_model_from_reader` を両学習器に追加(案A)。doc テストの `tokio_test`、ベンチの手製ランタイムが不要になり、`tokio-test` dev 依存を削除。
- `lib.rs` に主要型を再エクスポート。`Segmenter` のフィールドを非公開化しアクセサを追加。`get_attributes` を非公開化。
- 命名整理: `get_bias`→`bias`、`get_metrics`→`metrics`、`get_type`→`char_type`。`Metrics` 2 種を `metrics::BinaryMetrics` / `MulticlassMetrics` に改名・集約(D4 解消)。
- **B4 修正**: `AdaBoost::load_model_from_reader` は既存の特徴インデックスがある場合に名前でマージ(未知特徴は末尾に追加)し、構築済みインスタンスのインデックスを保全。`AveragedPerceptron` 側もクラス一覧をマージ。回帰テスト追加。
- `util` モジュール廃止(`ModelScheme` は `model_io` 内部へ)。CLI は `Language` を型付き引数で受理。
- `CHANGELOG.md` を新設し、v0.5.0 の変更点と移行ガイドを記載。バージョンを 0.5.0 に更新。
- 見送り: Trainer / PosTrainer の構造統合(2 つの薄いラッパーのままエラー型のみ刷新。統合の利得が小さい)。

**作業項目**:
1. **エラー型の導入**(`thiserror` 採用、`litsea/src/error.rs`):
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum LitseaError {
       #[error("I/O error: {0}")]            Io(#[from] std::io::Error),
       #[error("invalid model (line {line}): {reason}")] InvalidModel { line: usize, reason: String },
       #[error("unsupported URI scheme: {0}")] UnsupportedScheme(String),
       #[error("unsupported in this build/environment: {0}")] Unsupported(&'static str),
       // ...
   }
   pub type Result<T> = std::result::Result<T, LitseaError>;
   ```
   - `Box<dyn Error>`(trainer / extractor)と `io::Error` 流用をすべて置換。手書き `map_err` 祭り(D7)を `#[from]` と専用バリアントで削減。
2. **async の限定**:
   - ローカル読み込みを sync の基本 API にする: `load_model_from_path(&Path)` / `load_model_from_reader(R: BufRead)`(公開)。
   - URI 対応の `async fn load_model(uri)` は `remote_model` feature 時のみ提供(または常時提供だがリモートのみ非同期)。
   - 効果: CLI 以外から tokio 依存が消え、doc テストの `tokio_test::block_on`、ベンチの手製ランタイム(`bench.rs:14-30`)が不要になる。
3. **公開面の整理**:
   - `lib.rs` で主要型を re-export: `pub use {Segmenter, Language, AdaBoost, AveragedPerceptron, Upos, SegmentLabel, LitseaError};`
   - `Segmenter::learner` / `pos_learner` の pub フィールドを廃止し、コンストラクタ/アクセサ経由に。
   - `parse_model_content` 等の内部メソッドの可視性を `pub(crate)` に統一・点検。
4. **API 一貫性**:
   - `predict` を両学習器とも借用(`&` 受け)に統一。
   - メソッド命名を Rust 慣習に統一(`get_bias` → `bias`、`get_type` → `char_type` 等。`get_metrics` → `metrics`)。
   - `adaboost::Metrics` → `BinaryMetrics`、`perceptron::Metrics` → `MulticlassMetrics` に改名し `litsea/src/metrics.rs` に同居(D4 解消)。
5. **CLI の整備**(`litsea-cli`):
   - `Language` に `clap::ValueEnum` を実装(または CLI 側ラッパー)し、生 `String` + 手動 parse を廃止。
   - `--pos` フラグの分岐を、共通化された Trainer/Segmenter API の上に薄く載せ直す。
6. **CHANGELOG / 移行ガイド**を作成し、v0.5.0 としてのリリース準備(公開は本計画外)。

**完了条件**: 全テスト green、`litsea-cli` と docs のコード例が新 API でコンパイル可、semver 上の変更点が CHANGELOG に列挙されている。
**リスク**: 中。破壊的変更だが利用箇所(CLI・ベンチ・doc テスト・docs)はすべてリポジトリ内なので追従可能。外部利用者向けには移行ガイドで対応。
**規模目安**: 大(全ファイルに波及、ただし機械的な置換が中心)。

---

### フェーズ 4: 内部データ表現とパフォーマンス最適化(挙動不変) — ✅ 実施済み

**目的**: ホットパス(文字種分類・特徴量生成・Perceptron 学習)のアロケーション削減。

**実施結果**(2026-06-12、criterion median、Phase 0 ベースライン比):
- **文字種分類の match 化**: `CharTypePatterns`(正規表現の線形走査)を `char` 範囲 `match` の `Language::char_type()` に置換。**61ns → 9.4ns(-85%)**。3 言語で重複していた P/A/N パターンも 1 関数に集約(D6 解消)。**`regex` 依存を削除**。
- **Perceptron 重みレイアウトの転置**: `HashMap<class, HashMap<feat, w>>` → `HashMap<feat, Vec<w; クラス数>>`。1 文字あたりのハッシュ参照が 特徴数×クラス数(~756 回)→ 特徴数(~42 回)に減少。
- **属性生成のアロケーション削減**: 特徴テンプレートを単一の `write_attributes()`(再利用バッファ + sink)に集約。`segment()` は HashSet を作らず重みを直接合算、`segment_with_pos()` は `Vec<String>` バッファを位置間で再利用。bias の毎文字再計算(全重みの総和!)も文単位の 1 回に。
- **学習ループ**: `train()` のインスタンス全クローンを `mem::take` に置換、エポック毎の HashSet 再構築を撤廃。
- tags/types を `&'static str` 化。
- **ベンチ結果**: `segment_short` adaboost **-66〜70%**(ja 56.8→17.1µs)、perceptron **-88〜90%**(ja 293→30.8µs)。`segment_long_japanese` adaboost **611ms → 215ms(-65%)**、perceptron **4.48s → 396ms(-91%)**。`add_corpus` -37%。
- **挙動不変をゴールデンテスト 10 件の完全一致で確認**(モデルファイル形式も round-trip テストで不変)。
- 未実施(費用対効果で見送り): `chars: Vec<String>` → `Vec<char>` 化(センチネル表現の複雑化に対し残る利得が小さい)。

**作業項目**(独立した PR に分割可、効果の大きい順):
1. **文字種分類の match 化**(`language.rs`):
   - `CharTypePatterns` の正規表現線形走査を、`char` の数値範囲による `match` 実装(`enum CharType` + `fn classify(ch: char) -> CharType`)に置換。韓国語の받침判定クロージャも自然に統合できる。
   - 共通パターン(句読点・Latin・数字)の 3 言語コピペ(D6)もここで解消。
   - 効果次第で `regex` 依存を削除(依存ゼロ化は wasm ビルドにも有利)。
   - 公開 API(`get_type(&str) -> &str`)は型コード文字列を返す互換層を残すか Phase 3 と合わせて整理。
2. **Segmenter 内部表現の脱 String**:
   - `chars: Vec<String>` → `Vec<char>`(`B1/E1` 等のセンチネルは私用領域の `char` 定数または enum)。
   - `tags` → `enum Tag { U, B, O }`、`types` → `CharType` enum。
   - `get_attributes` の `format!` × 40 を、事前確保バッファへの書き込み + 必要箇所のみ `String` 化に変更。`HashSet<String>` は重複がほぼ発生しない構造のため `Vec<String>` 化を検討(モデルの特徴名文字列フォーマットは**変更しない**)。
3. **AveragedPerceptron の学習ループ最適化**:
   - epoch ごとの `instances.clone()`(`perceptron.rs:208`)を撤廃(借用分割 or インデックス走査)。
   - インスタンスごとの `HashSet` 再構築をやめ、`Vec<String>`(または ID 列)のまま予測。
   - 重みを「feature ID × class ID」の密/疎 Vec に変更(文字列キーの二重 HashMap を排除)。保存・読み込み時のみ文字列に変換し、**モデルファイル形式は不変**。
4. **AdaBoost の軽微な最適化**: `predict` の借用化(Phase 3 で実施済みのはず)、`initialize_*` の 1 パス化(Phase 2 で実施済み)に伴う残課題の整理。

**完了条件**: ゴールデンテスト完全一致(出力不変)、criterion で segment 系ベンチの改善を確認(目安: 文字種分類と特徴量生成で数倍規模の改善が見込める)、リグレッションなし。
**リスク**: 中。最適化は 1 PR = 1 テーマで小さく刻み、各 PR でベースライン比較を必須にする。
**規模目安**: 大(ただし分割実施)。

---

### フェーズ 5: ドキュメント・体裁の統一(仕上げ)

**目的**: コードコメント・ドキュメント・周辺ファイルの一貫性確保。

**作業項目**:
1. **doc コメント言語の統一**: crates.io 公開ライブラリのため**英語に統一**(`perceptron.rs`・`upos.rs`・`trainer.rs` 後半・`segmenter.rs` の日本語コメントを翻訳)。日本語ドキュメントは既存の `docs/ja`(mdbook)に集約する。
2. **doc コメント様式の統一**: `# Returns:` インライン型と `# Returns` セクション型の混在を後者に統一。冗長な逐語説明(処理手順の箇条書きをそのまま書いたもの)を要点のみに圧縮。
3. **docs(mdbook)/ README の更新**: Phase 3 の新 API にコード例を追従。EN/JA の同期。
4. **周辺ファイルの掃除**: `rustfmt.toml` のコメントアウト行削除、`Makefile` ターゲットの点検(`lint` に `--no-default-features` チェック追加等)。
5. `cargo doc --no-deps` の警告ゼロ化、`#![warn(missing_docs)]` の導入検討。

**完了条件**: ドキュメント言語・様式が統一され、docs のコード例がすべてコンパイル可能(doc テスト化)。
**リスク**: 低。
**規模目安**: 中(コード挙動への影響なし)。

---

## 4. 実施順序と依存関係

```
Phase 0 (安全網)
  └─> Phase 1 (バグ修正 + モデルI/O共通化)   … 0 とまとめて着手可
        └─> Phase 2 (Segmenter/Extractor/Trainer 統一)
              └─> Phase 3 (エラー型・公開API再設計 = v0.5.0)
                    └─> Phase 4 (性能最適化)  … 3 と並行可能な項目もあるが、API 確定後が手戻りなし
                          └─> Phase 5 (ドキュメント統一・仕上げ)
```

- **Phase 0+1 は最初の 1 PR にまとめてよい**(CI の feature チェックが B1 修正に依存するため)。
- Phase 4 の各最適化は独立 PR に分割し、それぞれゴールデン一致とベンチ比較を添付する。
- リリース戦略: Phase 1〜2 完了時点で 0.4.x パッチ/マイナーを出すことも可能。Phase 3 マージ後に v0.5.0。

## 5. リスク管理

| リスク | 緩和策 |
|--------|--------|
| 分割結果の意図しない変化 | Phase 0 のゴールデンテスト(エッジケース込み)を全フェーズの必須ゲートにする |
| モデルファイル互換の破壊 | round-trip テスト + 既存 8 モデルのロードテストを CI に常設 |
| 性能リグレッション | criterion ベースライン比較を Phase 4 の各 PR で必須化 |
| 破壊的変更の散発 | API 変更を Phase 3 に集約し、CHANGELOG と移行ガイドで一括提示 |
| wasm32 / no-default-features の退行 | Phase 0〜1 で CI にターゲット・フィーチャーマトリクスを追加 |
