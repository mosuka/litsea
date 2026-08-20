# 新しい言語の追加

Litseaの多言語フレームワークは、容易に拡張できるよう設計されています。本ガイドでは、新しい言語のサポートを追加する方法を、英語追加（issue #194）を具体例として一貫して用いながら説明します。

## 手順の概要

1. `Language` 列挙型にバリアントを追加
2. `Display` および `FromStr` のmatchアームを実装
3. 文字タイプ判定関数を作成
4. 分類関数を登録
5. WC特徴量の有無を決定
6. コーパスプロトコルを選択（スペース区切りまたは空白保持 TSV）
7. 同梱分割モデルを学習（binary-perceptron 畳み込み手順）
8. 任意で二段構成 POS モデルを学習
9. held-out 評価用ゴールドファイルを追加
10. テストを追加

## 手順1: `Language` にバリアントを追加

`litsea/src/language.rs` で、`Language` 列挙型に新しいバリアントを追加します。

```rust
#[non_exhaustive]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    English,
    Thai,       // ← new language
}
```

この列挙型に `#[non_exhaustive]` が付いているのは、まさに新しい言語の追加が想定されているためです。したがってバリアントの追加は、下流クレートにとって破壊的変更にはなりません。

## 手順2: Display と FromStr を実装

新しい言語のmatchアームを追加します。

```rust
// In Display impl
Language::Thai => write!(f, "thai"),

// In FromStr impl
"thai" | "th" => Ok(Language::Thai),
```

あわせて `language.rs` の `ParseLanguageError` のメッセージも更新してください。このメッセージはサポート言語を列挙しており（`Supported: japanese (ja), chinese (zh), korean (ko), english (en)`）、ユニットテストで固定されているため、メッセージとテストの両方に新しい言語を含める必要があります。

## 手順3: 文字タイプ判定関数を作成

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

英語の `english_char_type_id` は、同じパターンをラテン文字言語に適用した、リポジトリ内に実在する例です: 「U」（大文字）、「W」（空白）、「Q」（アポストロフィ）を専用クラスとして追加するだけでなく、共通クラス「P」を ASCII 句読点まで**拡張**しています（他の言語では ASCII 句読点は「O」のままです）-- 言語の分類関数は、新しいクラスを `punct_latin_digit()` の後ろに追加するだけでなく、その前段に追加のロジックを自由に重ねることができます。

### 文字タイプ設計のヒント

- 語境界パターンと相関する**言語学的に異なるカテゴリ**を特定する
- **順序は重要** -- 最初にマッチしたものが優先されるため、より具体的なパターンを汎用的なパターンの前に配置する
- **高頻度の機能語**を別のタイプとして検討する（中国語の「F」のように）。あるいは、スペース区切りの言語であれば、実際に境界と相関する句読点・大文字小文字・発音区別符号の違いを検討する（英語の「U」/「W」/「Q」のように）
- 単純な範囲だけでは足りない場合は**アーム本体内の追加ロジック**を使用する（韓国語が받침の有無で音節を分割するためにコードポイント判定を使用しているように）
- 共通の「P」/「A」/「N」クラスには、共有ヘルパー `punct_latin_digit()` を再利用する
- **コード集合は prefix-free に保つ** -- どのコードも他のコードのプレフィックスであってはならない（韓国語の `SN`/`SF` が成立するのは `S` 単独がコードでないためであり、そのため裸の `"S"` は韓国語に限らず**すべての**言語についてユニットテストで拒否されます）。モデルローダは packed 特徴キーへのコンパイル時に連結された種別コードを左から右へデコードするため、prefix-free 性がデコードの一意性を保証します
- **種別テーブルは最低 7 個のコードが必要です。** 共有テストコンテキスト（`packed_model.rs` の `ctx_for`）が `codes.len() >= 7` をアサートしており、英語の 7 コードのテーブルが現状の最小値です。上限は固定されていませんが、密な特徴量テーブル（`BC`/`UC`/`TC`/`BQ`/`TQ`）はおおよそ `type_count^2` から `type_count^3` に比例して大きくなるため、テーブルを大きくしすぎると分類の粒度と引き換えにモデルサイズとロード時間が増加します

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
        Language::Korean | Language::English => &TEMPLATES[..BASE_TEMPLATE_COUNT], // 38 個の基本テンプレート
    }
}
```

この match 式は意図的に**ワイルドカードアームを持たない網羅的（exhaustive）な match**にしてあります -- `Thai` を追加してもどちらのアームにも加えなければコンパイルエラーになり、暗黙のデフォルトにはなりません。これは意図的な設計です: 以前のバージョンのこの match には `_ => &TEMPLATES[..BASE_TEMPLATE_COUNT]` というフォールバックがあり、新しい言語は誰も意図的に決めないまま 38 テンプレート（WCなし）構成になってしまっていました。WC の判断は明示的に行い、推測ではなく計測で裏付けてください: held-out の dev split で tag-free モデルを WC ありとなしの両方で学習し、単語 F1 を比較します（英語での比較は[英語](english.md#wc特徴量なし)に記載されており、WC は「役に立たない」どころか実測で「悪化させる」ことが確認されました -- WCなし 98.68% に対し WCあり 98.65%）。出発点となるヒューリスティックとして: 対象言語の文字タイプに十分な多様性があり WC 特徴量が有益になりそうであれば含める、韓国語や英語の「SN」/「A」や空白が支配的なような低エントロピーなタイプ体系であれば除外する -- ただし、どちらを選ぶ場合も同梱モデルにコミットする前に数値で検証してください。

## 手順6: コーパスプロトコルを選択

Litseaは2種類のコーパスプロトコルをサポートしており、どちらを選ぶかは対象言語がスペース区切りで表記されるかどうかで決まります:

- **スペース区切り**（デフォルト）: 単語を1個のスペースでつなぎ、1行1文とする形式。スペースなしで表記される言語（日本語、中国語）や、スペースが境界シグナルを持たない場合に使用します。
- **空白保持 TSV**（`--format tsv`、issue #152）: トークンをタブで区切り、トークン自体がリテラルなスペース `" "` になり得る形式で、元の空白がそのまま学習用の一級特徴量として残ります。スペース自体が最も強い境界シグナルとなる言語（韓国語、英語）に使用します。

対象言語が単語間にスペースを入れて表記される場合（英語のように。韓国語の語節間の慣習とは異なりますが、根底にある理由は同じです）は、空白保持プロトコルを使用します:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l en -o /tmp)
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l english --format tsv --tag-free corpus.tsv features.txt
```

**元のツリーバンクに複数語トークン（短縮形、接語）が含まれる場合は、コーパスを信頼する前に `corpus_udtreebank.sh -s` がそれらを正しく扱えているか確認してください。** UD CoNLL-U では、英語の `don't` のような短縮形を、間にスペースを持たない2つの単語行（`do`、`n't`）をカバーする*範囲*行（例: `3-4  don't`）として表現します。`corpus_udtreebank.sh -s` は範囲行を特別扱いします: それ自体としてはトークンを出力せず、範囲を構成する単語間へのスペース挿入を抑制し、範囲自身の `SpaceAfter` アノテーションは最後の構成単語の後にのみ適用します。新しいツリーバンクについて確認すべき不変条件は次のとおりです: **範囲を構成する単語形を連結すると、範囲自身の表層形が再現されなければならない。** これは UD English-EWT のすべての複数語トークンで成立していました（学習前にコーパス全体を比較するスクリプトで検証済み）。対象のツリーバンクで成立しない場合、安全なフォールバックは、構成単語を展開する代わりに範囲自身の表層形を単一トークンとして出力することです。もう1つの独立したチェックとして、各文の TSV トークン（スペーストークンを含む）を連結して文を再構成し、その結果を CoNLL-U ファイルのその文に対応する `# text =` メタデータ行と diff してください -- これにより、構成単語の連結チェックだけでは見逃すスペースのバグを検出できます。`corpus_udtreebank.sh` 自体を変更した場合は、既存の空白保持ゴールドファイル（例: `resources/eval/korean_gsd_test.tsv`）も再生成し、コミット済みのバージョンと diff してください -- 変更が加法的であれば、バイト単位で同一になるはずです。

## 手順7: 同梱分割モデルを学習

同梱の分割モデルは、プレーンな `litsea train -t/-i`（AdaBoost ブースティング）では学習して**いません**。2クラスの Averaged Perceptron として学習し、無損失で AdaBoost モデル形式へ畳み込んでいます -- 導出の全体と正確な5ステップの手順（抽出 → 境界ラベル `1`/`-1` を `B`/`O` にリマップ → `train --perceptron` → `scripts/collapse_binary_perceptron.py` → 任意で剪定）は[事前学習済みモデル: 学習手順](../pre-trained-models.md#学習手順)を参照してください。空白保持プロトコルの言語の場合:

```sh
litsea extract -l english --format tsv --tag-free corpus.tsv features.txt
sed -i 's/^1\t/B\t/; s/^-1\t/O\t/' features.txt
litsea train --perceptron --num-epochs 20 features.txt perceptron.model
scripts/collapse_binary_perceptron.py perceptron.model models/english.model
```

エポック数や tag-free/WC の判断を推測で決めないでください -- held-out の**dev** split（最後に一度だけ触れるべき test split には触れない）でいくつかのエポック数のスイープを行い、それぞれの最良エポック数で tag-free とタグありの比較、および（手順5に従って）WC ありとなしの比較を行ってください。例えば英語のスイープでは、品質は20エポックでピークに達し、それ以降はわずかに悪化することがわかりました -- 一発だけの低エポック実行ではモデルの実際の品質を過小評価してしまい、一発だけの高エポック実行では、すでに通り過ぎた収束点ではなく過学習が天井であるかのように見えてしまいます。

## 手順8: 任意で二段構成 POS モデルを学習

言語に UPOS タグ付きデータが利用可能な場合は、追加で[二段構成](../algorithm/two-stage-tagging.md)モデルを学習できます:

```sh
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
litsea extract --pos -l english --stage2-features full pos_corpus.txt pos_features
litsea train --pos --num-epochs 50 pos_features models/english_pos.model
```

上記のエポックスイープと同様に、dev split で `--stage2-features`（`fast`/`balanced`/`full`）をスイープしてください。このパスを使う同梱3言語は、それぞれ異なる勝者を選んでいます（[stage-2 特徴量セットの選び方](../algorithm/two-stage-tagging.md#stage-2-特徴量セットの選び方)を参照）。

**空白保持プロトコルの言語における既知の制限**: 二段構成 extractor のコーパス形式には空白保持バリアントが存在しないため（`-p` と `-s` は互いに排他的）、`english_pos.model` は `english.model` 自体とは異なり、*空白なし*の `word/POS` コーパスで学習（および評価）されます。これが実際にどの程度問題になるかは言語によって異なります: 韓国語は膠着語形態論によりスペースを取り除いても強い境界の手がかりが残るため、空白保持の数値に近いまま留まります（空白なし 99.90% 対 空白あり 99.91%）。一方、英語の正書法にはそうしたシグナルがほとんど存在しないため、ギャップは大きくなります（空白なし 70.33% 対 空白あり 98.31%）。対象言語について、このギャップが小さいだろうと決め付けずに実測してください。もし小さくない場合は、目立つ形でドキュメント化してください -- このような大きなギャップをユーザーにどう説明するかの完全な実例については[英語](english.md#english_posmodel)を参照してください。

## 手順9: held-out 評価用ゴールドファイルを追加

ツリーバンクの**test** split（dev split のスイープが完了した後にのみ触れる）から held-out ゴールドデータを生成し、既存の命名規則（分割用は `<language>_<treebank>_test.{txt,tsv}`、POS 用は `<language>_<treebank>_test_pos.txt`）に従って `resources/eval/` 配下に追加します:

```sh
bash scripts/corpus_udtreebank.sh -s "$conllu_file_test" resources/eval/english_ewt_test.tsv
bash scripts/corpus_udtreebank.sh -p "$conllu_file_test" resources/eval/english_ewt_test_pos.txt
litsea evaluate -l english --format tsv models/english.model resources/eval/english_ewt_test.tsv
litsea evaluate -l english --pos models/english_pos.model resources/eval/english_ewt_test_pos.txt
```

`resources/eval/README.md` を新しいファイルとその出所/ライセンス（UD ツリーバンクは通常 CC BY-SA 4.0 であり、リポジトリの他の部分の MIT/Apache-2.0 ライセンスとは異なります）で更新してください。モデルのドキュメント（新しい数値が必要なドキュメントページの一覧は手順10を参照）には、`litsea train` の in-sample の出力ではなく、`litsea evaluate` から得た held-out の数値を記録してください。

## 手順10: テストとドキュメントを追加

この手順はスコープを狭く見積もりがちです: 新しい言語は `language.rs`/`segmenter.rs` だけでなく、もっと多くのテストファイルに影響します。以下のチェックリストに沿って確認してください。

**コードテスト:**

- `litsea/src/language.rs`: `ALL_LANGUAGES`（配列サイズを増やす）、`test_language_from_str`、`test_parse_language_error_message`（新しい完全なエラー文字列）、`test_language_display`、および全ての種別コードと共通クラス・「O」のケースをいくつかカバーする新しい `test_<language>_char_types`
- `litsea/src/packed_model.rs`: `test_templates_for_language_gating`（新しい言語のテンプレート数をアサート）、および `test_pack_parse_roundtrip_unique_and_injective` と `test_dense_index_consistent_with_key_decode` にあるハードコードされた2つの言語列挙配列
- `litsea/src/segmenter.rs`: 新しい `test_char_type_<language>`（既存の言語ごとのテストを踏襲）
- `litsea/src/word_features.rs`: ラウンドトリップのサンプルリストにケースを追加（低コストで、型コードのエンコーディングのバグを早期に検出できる）
- `litsea-cli/src/main.rs`: `--language` のヘルプ文字列3箇所

**モデルに依存するテスト（先に学習済みモデルが必要）:**

- `litsea/tests/golden.rs`: 新しい `golden_segment_<language>`（POS モデルも学習した場合は `golden_segment_with_pos_<language>_two_stage` も）。**理想化した出力ではなく、モデルの実際の出力を固定してください** -- まず実際の出力を `println!` で出力するテストを書き、その出力をアサーションにコピーしてから、デバッグ出力を削除します。直後に**サボタージュ検証**を行ってください: 期待値の1つを一時的に間違った値に変更し、テストが RED になることを確認してから、正しい値に戻します。作成時に一度も失敗しなかった golden テストは、何かを守っていることを証明していません
- `litsea/src/segmenter.rs`: packed スコアラーと文字列キーの参照実装を比較する差分テスト（`test_segment_differential_<language>_model`）、およびモデルが tag-free の場合は `segment_into` のタイリング/パリティのケース
- `litsea/benches/bench.rs`: 4つの言語ごとのタプルリスト（`bench_segment_short`、`bench_external_corpus` の2つのケースリスト、`bench_segment_into`）すべてに新しい言語を追加します。これには `resources/` 配下にコーパスファイル（既存の言語ごとのコーパスと同程度のサイズのパブリックドメインのテキスト）が必要です
- `litsea-cli/tests/cli.rs`: CLI の出力をエンドツーエンドで固定する分割のスモークテストを最低1つ

**検証コマンド**（言語の対応が完了したとみなす前に、すべて実行してください）:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench -- external_corpus   # 新しいベンチケースが実行できることを確認し、剪定が必要か判断する
markdownlint-cli2 "docs/src/**/*.md"
markdownlint-cli2 "docs/ja/src/**/*.md"
mdbook build docs
mdbook build docs/ja
```

**ドキュメント**（本プロジェクトのドキュメント方針に従い、英語のソースを先に、その後 `docs/ja/src/` 配下の日本語ミラーを更新します）:

- 新しい `language-support/<language>.md` ページ（[英語](english.md)または[韓国語](korean.md)をテンプレートとして使用）と `docs/src/SUMMARY.md` のエントリ（および `docs/ja/src/SUMMARY.md` のミラー）
- `language-support/overview.md` と `algorithm/character-type-classification.md` -- どちらも言語ごとのテーブル/セクションを拡張する必要があります
- `pre-trained-models.md` -- 同梱モデルごとのモデルカードに加え、該当する場合は「タグなし（pointwise）モデル」と「二段構成の品詞推定モデル」の比較表
- ルートの `README.md` と `litsea/src/lib.rs` のクレートレベルドキュメント
- スイープが完了したとみなす前に `grep -rln "<既存の言語名>" docs/src docs/ja/src` を実行し、見つかった全ファイルを確認してください -- 言語を列挙しているドキュメントは見落としやすいです（本プロジェクト自身の経験: issue #165 はこの確認を一度スキップしてしまい、14 箇所の古い記述が後続の PR でのクリーンアップ待ちになりました）
