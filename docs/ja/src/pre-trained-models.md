# 事前学習済みモデル

Litsea は `models/` ディレクトリに複数の事前学習済みモデルを同梱しています。

## モデルカタログ

単語分割モデルは、学習コーパスの held-out テスト分割（学習に使用していない文）で
評価しています。**単語 F1（Word F1）** は単語の完全一致、**境界 F1（Boundary F1）**
は個々の境界判定のスコアです。なお `train` コマンドが出力するのは学習データ自身で
測った *in-sample* 指標であり、ここに示す held-out の値より高くなる点に注意して
ください。

**アルゴリズムについての注記**: `japanese.model`、`chinese.model`、
`korean.model` は 2 クラス（境界／非境界）の Averaged Perceptron として学習した後、
スカラーの特徴量重みへ畳み込んでいます（issue #165）。ファイル自体は従来どおり
プレーンな AdaBoost テキスト形式のままで、`Segmenter::with_learner` /
`AdaBoost::load_model_from_path` は無変更で動作します。この畳み込みは
無損失な変換であり（導出は `scripts/collapse_binary_perceptron.py` の
docstring を参照）近似ではありません: この方法で学習した perceptron は、
同じコーパス・同じテンプレートで AdaBoost の presence-stump 弱学習器より
大幅に高い held-out 品質に達します。代わりにモデルファイルは大きくなり
（非ゼロ重みを持つ特徴量が増えるため）、学習手順も通常の `train` ではなく
`train --pos` を経由します（詳細は下記の[学習手順](#学習手順)を参照）。

### japanese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| 学習コーパス | UD Japanese-GSD |
| エポック数 | 50 |
| 剪定後の特徴量数 | \|重み\| 上位 40,000 |
| 単語 F1（held-out） | 96.70% |
| 境界 F1（held-out） | 98.59% |
| ファイルサイズ | 約 1.1 MB |

### korean.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 韓国語 |
| 学習コーパス | UD Korean-GSD（空白保持 TSV コーパス） |
| エポック数 | 30 |
| 剪定後の特徴量数 | 剪定なし（3,994 特徴量） |
| 単語 F1（held-out） | 99.90% |
| 境界 F1（held-out） | 99.95% |
| ファイルサイズ | 約 110 KB |

韓国語モデルは、元の語節（어절）間の空白を保持したテキストで学習・評価して
います（各空白は独立したトークンとして扱い、空白トークンは F1 の計算から
除外します）。韓国語では空白がほとんどの語境界を示すため、学習時に空白を
参照できるモデルは UD Korean-GSD の基準をほぼ決定的に解決できます -- これが
韓国語の特徴量数・ファイルサイズが小さいままである理由でもあります（モデルが
学習すべき曖昧性がほとんど残っていないため）。品質も旧 AdaBoost モデルから
実質的に変化していません（99.90% vs 99.91%、計測誤差の範囲内）。日本語と
中国語は空白を使わずに表記されるため、プロトコルは従来のままです。

### chinese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 中国語（簡体字・繁体字） |
| 学習コーパス | UD Chinese-GSD |
| エポック数 | 100 |
| 剪定後の特徴量数 | \|重み\| 上位 70,000 |
| 単語 F1（held-out） | 90.69% |
| 境界 F1（held-out） | 95.64% |
| ファイルサイズ | 約 2.0 MB |

### RWCP.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| ソース | オリジナルの [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) から抽出 |
| ライセンス | BSD 3-Clause (Taku Kudo) |
| ファイルサイズ | 約 22 KB |

### JEITA_Genpaku_ChaSen_IPAdic.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| 学習コーパス | JEITA プロジェクト 杉田玄白コーパス |
| トークナイザ | ChaSen with IPAdic |
| ファイルサイズ | 約 16 KB |

## 学習手順

`RWCP.model` と `JEITA_Genpaku_ChaSen_IPAdic.model` はレガシー・互換用モデルで、
従来どおりに学習（または取得）しています -- 通常の AdaBoost 手順は
[モデルの学習](training-guide/training-models.md)を参照してください。
`japanese.model`、`chinese.model`、`korean.model` は binary-perceptron 畳み込み手順
（issue #165）で再学習しています。エンジンの変更は不要ですが、通常の
`litsea train` に加えて数ステップが必要です:

```sh
# 1. プレーンな境界特徴量を抽出（従来と同じステップ）。
litsea extract -l <language> [韓国語なら --format tsv] <corpus> <features.txt>

# 2. 境界ラベル 1/-1 を B/O にリマップする。これは見た目の問題ではなく
#    正しさのために必須: perceptron 自身のタイブレーク（クラスインデックスが
#    小さい方が勝つ）を、AdaBoost の「score >= 0.0 は境界を優先する」という
#    規約と一致させるためのもの。"1"/"-1" のまま学習すると、タイの解決方向が
#    黙って逆転してしまう。
sed -i 's/^1\t/B\t/; s/^-1\t/O\t/' <features.txt>

# 3. 2 クラスの Averaged Perceptron として学習する。--pos はここでは
#    汎用的に流用しているだけで（PosTrainer はラベルを不透明な文字列として
#    扱う）、実際の品詞タグ付けのためではない。
litsea train --pos --num-epochs <N> <features.txt> <perceptron.model>

# 4. プレーンな AdaBoost モデル形式へ畳み込む（無損失 -- 導出はスクリプトの
#    docstring を参照）。
scripts/collapse_binary_perceptron.py <perceptron.model> <collapsed.model>

# 5. 任意: 特徴量数の増加が `cargo bench -- external_corpus` のスループットを
#    許容範囲を超えて悪化させる場合、上位 N 特徴量に剪定し held-out 品質と
#    速度の両方を再確認する。
scripts/prune_adaboost_model.py <collapsed.model> <pruned.model> <n>
```

エポック数と剪定閾値は固定値ではなく言語ごとのチューニング項目です --
上記の同梱モデルを選んだのと同じように、エポックスイープと品質・スループットの
トレードオフスイープから決めてください（スイープの全データは issue を参照）。
大まかな傾向として、品質は少数のエポックを大きく超えて向上し続け、最終的には
プラトーに達するか（日本語は約 50 エポックを超えると軽度のオーバーフィットが
見られます）、単一の「正しい」エポック数があるわけではありません。剪定による
品質劣化は言語固有の崖に達するまで緩やかに進む傾向があるため、数値を推測するの
ではなく、`cargo bench` のスループットが回復し始める付近の剪定レベルを
いくつか試してください。

## 品詞推定モデル

in-sample 行は `train` コマンドが学習データ自身で測ったメトリクス、held-out
行は UD GSD テスト分割に対して `litsea evaluate --pos` で測定した単語 /
タグ付き単語 F1 です（[モデルの評価](training-guide/evaluating-models.md)を
参照）。韓国語の POS ゴールドは POS パイプラインの慣例（空白トークンなし）に
従うため、空白なしのテキストで評価しています。

### japanese_pos.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| アルゴリズム | Averaged Perceptron |
| 学習コーパス | UD Japanese-GSD（7,050 文） |
| エポック数 | 10 |
| 正解率（in-sample） | 98.23% |
| マクロ適合率（in-sample） | 96.82% |
| マクロ再現率（in-sample） | 93.30% |
| 単語 F1（held-out） | 96.56% |
| タグ付き単語 F1（held-out） | 92.51% |
| ファイルサイズ | 約 11 MB |

### chinese_pos.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 中国語（簡体字・繁体字） |
| アルゴリズム | Averaged Perceptron |
| 学習コーパス | UD Chinese-GSD（3,997 文） |
| エポック数 | 10 |
| 正解率（in-sample） | 97.04% |
| マクロ適合率（in-sample） | 97.17% |
| マクロ再現率（in-sample） | 96.14% |
| 単語 F1（held-out） | 90.52% |
| タグ付き単語 F1（held-out） | 81.18% |
| ファイルサイズ | 約 19 MB |

### korean_pos.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 韓国語 |
| アルゴリズム | Averaged Perceptron |
| 学習コーパス | UD Korean-GSD（4,400 文） |
| エポック数 | 10 |
| 正解率（in-sample） | 95.14% |
| マクロ適合率（in-sample） | 95.00% |
| マクロ再現率（in-sample） | 86.15% |
| 単語 F1（held-out） | 80.51% |
| タグ付き単語 F1（held-out） | 71.03% |
| ファイルサイズ | 約 8.9 MB |

#### 使用方法

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_pos.model
```

出力:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## 二段構成の品詞推定モデル

[二段構成アーキテクチャ](algorithm/two-stage-tagging.md)（issue #147）は、
文字位置ごとに全 UPOS クラスを採点する代わりに、二値の境界分類器で分割し、
確定した各単語を候補タグ語彙表と単語単位のタガーでタグ付けします。
これは純粋な追加機能です: 上記の `*_pos.model` ファイルは影響を受けず、
`segment --pos` / `evaluate --pos` はファイルからどちらの種類かを
自動判別します。アーキテクチャと言語別の推奨については
[二段構成 vs Joint タグ付け](algorithm/two-stage-tagging.md)を参照してください。

in-sample・held-out の各行は上記 joint モデルと同じプロトコルです。
「stage-2 特徴量セット」は単語単位テンプレートの選択
（`fast`、`balanced`、`full`。[特徴量の抽出](training-guide/extracting-features.md)を参照）
で、[二段構成 vs Joint タグ付け](algorithm/two-stage-tagging.md#stage-2-特徴量セットの選び方)の
実測トレードオフから言語ごとに同梱モデル用に選定しています。スループットは
このページの他のベンチマーク数値と同じ開発機での `cargo bench --
external_corpus` によるもので、専用のアイドルハードウェアではありません
（そのページの方法論の注記を参照）。joint との比較値は、実行間のばらつきが
大きいため `*_pos.model` の表の値ではなく同一実行内で計測したスループットです。

**エポック数についての注記**: 上記の joint モデルは、元々の 10 エポック学習の
まま公開されています。二段構成の同梱にあたって行ったエポックスイープ
（10〜150 エポック）では、stage 1 の**分割**品質は特に 10 エポックを大きく
超えて向上し続け、50 エポック付近でプラトーに達することが判明しました
-- 以下の同梱二段構成モデルは、joint モデルと同じ 10 エポック慣習ではなく、
このスイープから得た 50 エポックを使用しています。同じエポック数で比較すると、
このスイープでは joint の中国語分割が依然として小さく一貫した優位
（テストした全エポック数で概ね 0.5〜0.9pt）を保つことも分かりましたが、
硬い上限ではありません: 以下の `chinese_two_stage.model`（50 エポック）は
公開済み（10 エポック）の `chinese_pos.model` より高い held-out Word F1 を
達成しています。

### japanese_two_stage.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| 学習コーパス | UD Japanese-GSD（7,050 文） |
| エポック数 | 50 |
| stage-2 特徴量セット | `fast` |
| 単語 F1（held-out） | 96.78%（joint: 96.56%） |
| タグ付き単語 F1（held-out） | 92.95%（joint: 92.51%） |
| joint比スループット | 約2.8x（3回計測: 2.65〜3.05x） |
| ファイルサイズ | 約 5.4 MB |

### chinese_two_stage.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 中国語（簡体字・繁体字） |
| 学習コーパス | UD Chinese-GSD（3,997 文） |
| エポック数 | 50 |
| stage-2 特徴量セット | `balanced` |
| 単語 F1（held-out） | 90.82%（joint: 90.52%） |
| タグ付き単語 F1（held-out） | 82.29%（joint: 81.18%） |
| joint比スループット | 約2.3x（3回計測: 2.13〜2.44x） |
| ファイルサイズ | 約 8.0 MB |

### korean_two_stage.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 韓国語 |
| 学習コーパス | UD Korean-GSD（4,400 文、空白非保持の word/POS プロトコル -- 下記の注記を参照） |
| エポック数 | 50 |
| stage-2 特徴量セット | `balanced` |
| 単語 F1（held-out） | 83.24%（joint: 80.51%） |
| タグ付き単語 F1（held-out） | 78.86%（joint: 71.03%） |
| joint比スループット | 約1.8x（3回計測: 1.75〜1.90x） |
| ファイルサイズ | 約 5.0 MB |

韓国語のスループット向上が小さいのは語彙表に起因します: held-out
テキストの 34.5% が未知語（学習時に未出現の表層）で、未知語は常に
stage 2 の全クラスフォールバックを払うことになり、安価な dominance
スキップや候補マスクの経路を使えません。そのため日本語・中国語より
多くの割合の韓国語の単語が stage 2 のフルコストを負担します。

**韓国語のプロトコルについての注記**: `korean_two_stage.model` は
`korean_pos.model` と同じ空白非保持の `word/POS` コーパスで学習されており、
`korean.model` が使う空白保持 TSV コーパス（issue #152）ではありません。
二段構成の extractor は両ステージに単一のコーパスを使うため、空白保持＋
POS 付きを組み合わせた形式の構築は別機能であり未実装です。上記の数値は
`korean_pos.model` とは比較可能ですが、`korean.model` の 99.91%
（全く異なるコーパス・プロトコル）とは比較できません -- 二段構成が
優れている/劣っているという結果ではありません。

#### 使用方法

```sh
echo "これはテストです。" | litsea segment --pos -l japanese models/japanese_two_stage.model
```

出力は joint モデルと同じ形です:

```text
これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT
```

## モデルの選択

- **日本語**には、最高精度を求める場合は `japanese.model` を、オリジナルの TinySegmenter との互換性を重視する場合は `RWCP.model` を使用
- **中国語**には `chinese.model` を使用
- **韓国語**には `korean.model` を使用
- **品詞推定**には**二段構成**モデル（`japanese_two_stage.model`、
  `chinese_two_stage.model`、`korean_two_stage.model`）を推奨します --
  同梱されている状態で、現在公開されている joint モデルを Word F1・
  Tagged F1 の両方でどの言語でも上回り、スループットは 1.8〜2.8 倍です。
  joint（`*_pos.model`）は、`with_pos_learner()` に依存する既存コードとの
  互換性のために、また同一学習量では joint が中国語分割で小さいながら
  優位を保つケースのために引き続き利用可能にしています（詳細は上記の
  エポック数についての注記を参照）。詳しい比較は
  [二段構成 vs Joint タグ付け](algorithm/two-stage-tagging.md)を参照してください。
- **ドメイン固有**の用途には、[独自モデルの学習](training-guide/preparing-corpus.md)または既存モデルの[再学習](training-guide/retraining-models.md)を検討

## サンプルデータ

`resources/` ディレクトリには以下も含まれています:

- **bocchan.txt** -- 坊っちゃん（夏目漱石）、約 307 KB。`segment_long_japanese` ベンチマークと差分テストに使用。
- **wagahaiwa_nekodearu.txt** -- 吾輩は猫である（夏目漱石）、約 1.1 MB、青空文庫。
- **mujeong.txt** -- 무정（李光洙、1917）、約 786 KB、ko.wikisource。
- **rulin_waishi.txt** -- 儒林外史（呉敬梓）、約 985 KB、zh.wikisource。

後半の 3 つは外部の
[tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
のコーパスとバイト同一で、`external_corpus` ベンチマークグループが使用します
（[ベンチマーク](advanced/benchmarking.md)を参照）。いずれもパブリックドメインです。
