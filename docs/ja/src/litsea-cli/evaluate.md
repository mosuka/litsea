# evaluate

学習済みモデルを held-out のゴールドコーパスに対して評価し、品質メトリクスを
出力します。`train` が出力する in-sample 指標とは異なり、モデルが学習で一度も
見ていないテキストに対する品質を測定します。

**ゴールドコーパス**とは、正解が書かれたテキストファイルです: 1 行 1 文で、
人手アノテーションにより正しいトークンへ分割済みのもの（学習用コーパスと同じ
ファイル形式 — 空白区切り・タブ区切り、`--pos` 時は `word/POS`）。「ゴールド」は
モデル出力の採点基準となる正解（ゴールドスタンダード）を指します。意味のある
**held-out** 評価にするには、モデルの学習に使っていない文で構成されている必要が
あります — 同梱の `resources/eval/` は UD GSD の **test** 分割で、同梱モデルは
train 分割で学習されているため、この条件を満たします。

## 使い方

```sh
litsea evaluate [OPTIONS] <MODEL_URI> <GOLD_FILE>
```

## 引数

| Argument | Description |
|----------|------------|
| `MODEL_URI` | 学習済みモデルファイルのパスまたはURL。サポート形式: ローカルファイルパス, `file://`, `http://`, `https://` |
| `GOLD_FILE` | ゴールドコーパスのパス（1行1文） |

## オプション

| Option | Default | Description |
|--------|---------|------------|
| `-l`, `--language <LANGUAGE>` | `japanese` | モデルとゴールドコーパスの言語。指定可能な値: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko`, `english` / `en` |
| `--pos` | off | 単語分割と品詞推定を同時に評価します。この場合、ゴールドコーパスは `word/POS` 形式である必要があります。[二段構成](../advanced/model-file-format.md#二段構成モデル形式litsea-two-stage-v1)モデル（`train --pos`）が必要です |
| `--format <FORMAT>` | `space` | ゴールドコーパスの形式: `space`（スペース区切りトークン）または `tsv`（タブ区切りトークン。韓国語/英語の空白保持コーパスのように、トークンとして空白文字そのものを含められます）。`--pos` 指定時は無視されます |

## メトリクス

各文について、次の 2 つのトークン列を比較します:

- **ゴールドトークン** -- ゴールドコーパスに記録された基準の分割。人手で
  アノテーションされた正解（ここでは UD GSD ツリーバンク test 分割の
  トークン分割）です。評価対象の文テキストは、これらを連結して復元します
- **予測トークン** -- 復元した文テキストを `segment`（`--pos` 時は
  `segment --pos`）に与えたときにモデルが出力する分割。ユーザーが推論時に
  得るものと完全に同じです

予測トークンとゴールドトークンは、復元した文上の文字オフセットスパンの
完全一致で対応付けます。空白のみのトークンはスコア計算から除外されるため、
韓国語/英語の空白保持プロトコルが数値を押し上げることはありません。

| メトリクス | 測るもの | 低い場合の意味 |
|--------|----------|-------------------|
| Word Precision（単語適合率） | **予測**した単語のうち、ゴールドの単語と完全一致（両端が正しい）した割合 | 余計な単語が多い: 過分割、または誤った結合 |
| Word Recall（単語再現率） | **ゴールド**の単語のうち、完全一致で復元できた割合 | 取りこぼしたゴールド単語が多い |
| Word F1（単語 F1） | 単語適合率と再現率の調和平均 | 分割品質の総合指標 |
| Boundary Precision（境界適合率） | **予測**した単語開始位置のうち、ゴールドの境界だった割合 | 誤った境界が多い（過分割） |
| Boundary Recall（境界再現率） | **ゴールド**の単語開始位置のうち、検出できた割合 | 見逃した境界が多い（分割不足） |
| Boundary F1（境界 F1） | 境界適合率と再現率の調和平均 | 境界判定の総合指標 |
| Tagged Word Precision / Recall / F1（`--pos`） | 単語メトリクスと同様だが、予測 POS タグの一致も必要 | スパンは正しいがタグが誤っている |

単語は**両端の境界がともに正しい**場合のみ正解と数えるため、単語メトリクスは
常に境界メトリクスと同等以上に厳しくなります — 境界が 1 つずれるだけで、その
両側の 2 単語が不正解になります。`Sentences` は評価対象（非空）のゴールド文数です。

## 使用例

同梱のゴールドデータ（`resources/eval/`、UD GSD テスト分割から変換）を使って、
ドキュメントに記載の held-out の数値を再現します:

```sh
litsea evaluate -l japanese models/japanese.model resources/eval/japanese_gsd_test.txt
litsea evaluate -l korean --format tsv models/korean.model resources/eval/korean_gsd_test.tsv
litsea evaluate -l chinese models/chinese.model resources/eval/chinese_gsd_test.txt
litsea evaluate -l english --format tsv models/english.model resources/eval/english_ewt_test.tsv
litsea evaluate --pos -l japanese models/japanese_pos.model resources/eval/japanese_gsd_test_pos.txt
```

出力:

```text
Evaluation Metrics:
  Sentences: 543
  Word Precision: 96.73%
  Word Recall: 96.66%
  Word F1: 96.70%
  Boundary Precision: 98.63%
  Boundary Recall: 98.56%
  Boundary F1: 98.59%
```
