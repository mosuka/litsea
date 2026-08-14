# 事前学習済みモデル

Litsea は `models/` ディレクトリに複数の事前学習済みモデルを同梱しています。

## モデルカタログ

単語分割モデルは、学習コーパスの held-out テスト分割（学習に使用していない文）で
評価しています。**単語 F1（Word F1）** は単語の完全一致、**境界 F1（Boundary F1）**
は個々の境界判定のスコアです。なお `train` コマンドが出力するのは学習データ自身で
測った *in-sample* 指標であり、ここに示す held-out の値より高くなる点に注意して
ください。

### japanese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| 学習コーパス | UD Japanese-GSD |
| 学習オプション | `-t 0.0001 -i 20000` |
| 単語 F1（held-out） | 91.48% |
| 境界 F1（held-out） | 96.31% |
| ファイルサイズ | 約 20 KB |

### korean.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 韓国語 |
| 学習コーパス | UD Korean-GSD（空白保持 TSV コーパス） |
| 学習オプション | `--format tsv`, `-t 0.0001 -i 20000` |
| 単語 F1（held-out） | 99.91% |
| 境界 F1（held-out） | 99.96% |
| ファイルサイズ | 約 9.4 KB |

韓国語モデルは、元の語節（어절）間の空白を保持したテキストで学習・評価して
います（各空白は独立したトークンとして扱い、空白トークンは F1 の計算から
除外します）。韓国語では空白がほとんどの語境界を示すため、学習時に空白を
参照できるモデルは UD Korean-GSD の基準をほぼ決定的に解決できます。日本語と
中国語は空白を使わずに表記されるため、プロトコルは従来のままです。

### chinese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 中国語（簡体字・繁体字） |
| 学習コーパス | UD Chinese-GSD |
| 学習オプション | `-t 0.0001 -i 20000` |
| 単語 F1（held-out） | 77.56% |
| 境界 F1（held-out） | 87.81% |
| ファイルサイズ | 約 18 KB |

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

## モデルの選択

- **日本語**には、最高精度を求める場合は `japanese.model` を、オリジナルの TinySegmenter との互換性を重視する場合は `RWCP.model` を使用
- **中国語**には `chinese.model` を使用
- **韓国語**には `korean.model` を使用
- **品詞推定**には、対応する `*_pos.model`（`japanese_pos.model`、`chinese_pos.model`、`korean_pos.model`）を使用して単語分割と品詞推定を同時実行
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
