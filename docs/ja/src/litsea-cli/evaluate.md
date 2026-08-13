# evaluate

学習済みモデルを held-out のゴールドコーパスに対して評価し、品質メトリクスを
出力します。`train` が出力する in-sample 指標とは異なり、モデルが学習で一度も
見ていないテキストに対する品質を測定します。

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
| `-l`, `--language <LANGUAGE>` | `japanese` | モデルとゴールドコーパスの言語。指定可能な値: `japanese` / `ja`, `chinese` / `zh`, `korean` / `ko` |
| `--pos` | off | 単語分割と品詞推定を同時に評価します。この場合、ゴールドコーパスは `word/POS` 形式である必要があります |
| `--format <FORMAT>` | `space` | ゴールドコーパスの形式: `space`（スペース区切りトークン）または `tsv`（タブ区切りトークン。韓国語の空白保持コーパスのように、トークンとして空白文字そのものを含められます）。`--pos` 指定時は無視されます |

## メトリクス

予測トークンとゴールドトークンは、復元した文（ゴールドトークンの連結）上の
文字オフセットスパンの完全一致で対応付けます。空白のみのトークンはスコア
計算から除外されるため、韓国語の空白保持プロトコルが数値を押し上げることは
ありません。

- **単語適合率 / 再現率 / F1** -- トークンの完全一致
- **境界適合率 / 再現率 / F1** -- 個々のトークン開始位置の判定（文頭は除く）
- `--pos` 指定時: **タグ付き単語適合率 / 再現率 / F1** -- スパン**と** POS
  タグの両方が一致

## 使用例

同梱のゴールドデータ（`resources/eval/`、UD GSD テスト分割から変換）を使って、
ドキュメントに記載の held-out の数値を再現します:

```sh
litsea evaluate -l japanese models/japanese.model resources/eval/japanese_gsd_test.txt
litsea evaluate -l korean --format tsv models/korean.model resources/eval/korean_gsd_test.tsv
litsea evaluate -l chinese models/chinese.model resources/eval/chinese_gsd_test.txt
litsea evaluate --pos -l japanese models/japanese_pos.model resources/eval/japanese_gsd_test_pos.txt
```

出力:

```text
Evaluation Metrics:
  Sentences: 543
  Word Precision: 91.50%
  Word Recall: 91.47%
  Word F1: 91.48%
  Boundary Precision: 96.32%
  Boundary Recall: 96.29%
  Boundary F1: 96.31%
```
