# モデルの評価

モデルの品質を理解することは、良好な分割結果を得るために不可欠です。

## メトリクス

`train` コマンドは学習後に3つの主要なメトリクスを出力します。これらは
**in-sample** 指標（学習データ自身で測った値）であり、未知のテキストに対する
性能を過大評価します。実際の品質を知るには、必ず学習に使用していない held-out
コーパスで評価してください（後述のベンチマークを参照）。

### Accuracy（正解率）

```text
Accuracy = (TP + TN) / Total Instances
```

すべての文字位置のうち、正しく分類された割合（境界と非境界の両方を含む）です。モデル品質の最も広範な指標です。

### Precision（適合率）

```text
Precision = TP / (TP + FP)
```

モデルが**予測した**境界のうち、**正しかった**割合です。高い適合率は、誤った境界（過分割）が少ないことを意味します。

### Recall（再現率）

```text
Recall = TP / (TP + FN)
```

**実際の**境界のうち、モデルが**検出した**割合です。高い再現率は、見逃された境界（不足分割）が少ないことを意味します。

## 混同行列

| | 境界と予測 (+1) | 非境界と予測 (-1) |
|---|---|---|
| **実際の境界** | True Positive (TP) | False Negative (FN) |
| **実際の非境界** | False Positive (FP) | True Negative (TN) |

## 事前学習済みモデルのベンチマーク

同梱の `japanese.model`、`chinese.model`、`korean.model` は、通常の AdaBoost
`-t`/`-i` 学習ではなく binary-perceptron 畳み込み手順で学習しています --
正確な手順は[学習手順](../pre-trained-models.md#学習手順)を参照してください。
いずれも学習コーパスの held-out テスト分割で評価しています。**単語 F1** は
単語の完全一致、**境界 F1** は個々の境界判定のスコアです。

| モデル | 単語 F1 | 境界 F1 | 学習コーパス |
|-------|---------|---------|-----------------|
| japanese.model | 96.70% | 98.59% | UD Japanese-GSD |
| korean.model | 99.91% | 99.96% | UD Korean-GSD |
| chinese.model | 90.69% | 95.64% | UD Chinese-GSD |

韓国語は、元の語節（어절）間の空白を保持したテキスト（空白保持 TSV コーパス。
空白トークンは F1 の計算から除外）で学習・評価しています。韓国語では空白が
ほとんどの語境界を示すため、空白を使わずに表記される日本語・中国語に比べて
タスクが大幅に容易になります — このスコアは言語間で直接比較できません。

### ベンチマークの再現

上の表のすべての数値は、同梱のゴールドデータ（`resources/eval/`、UD GSD の
**test** 分割から変換。同梱モデルは train 分割で学習しているため held-out に
あたります）を使って、それぞれ 1 コマンドで再現できます:

```sh
litsea evaluate -l japanese models/japanese.model resources/eval/japanese_gsd_test.txt
litsea evaluate -l korean --format tsv models/korean.model resources/eval/korean_gsd_test.tsv
litsea evaluate -l chinese models/chinese.model resources/eval/chinese_gsd_test.txt
```

コマンドリファレンスは [evaluate](../litsea-cli/evaluate.md) を参照して
ください。POS モデルは `*_gsd_test_pos.txt` ファイルに対して `--pos` で評価し、
その held-out の数値は[事前学習済みモデル](../pre-trained-models.md)に記載して
います。なお、韓国語の POS ゴールドは POS パイプラインの慣例（空白トークン
なし）に従うため、上のセグメンテーション行とは異なり、空白なしのテキストで
評価します。

## モデル品質の改善

精度が不十分な場合は、以下を検討してください:

1. **より多くの学習データ** -- より大規模で多様なコーパスを用意する
2. **閾値を下げる** -- `-t 0.0001` を試して、より多くのブースティング反復を許可する
3. **反復回数を増やす** -- `-i 20000` 以上を試す。AdaBoost は 1 反復につき弱学習器
   （特徴）を 1 つ選択するため、反復回数がモデルの特徴数の上限になります。CLI の
   デフォルト（`-i 100`）では非常に小さいモデルになり、held-out 精度が大幅に
   低くなります
4. **コーパスの品質向上** -- 一貫したトークン化とクリーンなテキストを確保する
5. **再学習** -- 既存のモデルから開始し、追加データで学習する（[モデルの再学習](retraining-models.md)を参照）

上記の閾値・反復回数のチューニングは、通常の AdaBoost 学習
（`--perceptron`/`--pos` を付けない `litsea train`）に適用されるものです。
同梱モデル自身が通常の AdaBoost に対して達成している +5〜13pt の held-out
品質向上は、`-t`/`-i` のチューニングによるものではなく、2 クラスの
Averaged Perceptron を学習してから無損失に AdaBoost の重みへ畳み込む手順に
よるものです。段階的な改善ではなく同梱モデルと同水準の品質を目指す場合は、
[学習手順](../pre-trained-models.md#学習手順)のレシピを参照してください。
