# 事前学習済みモデル

Litsea は `resources/` ディレクトリに複数の事前学習済みモデルを同梱しています。

## モデルカタログ

### japanese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 日本語 |
| 学習コーパス | 日本語 Wikipedia 記事 |
| トークナイザ | Lindera (UniDic) |
| 正解率 | 94.15% |
| 適合率 | 95.57% |
| 再現率 | 94.36% |
| ファイルサイズ | 約 2.9 KB |

### korean.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 韓国語 |
| 学習コーパス | 韓国語 Wikipedia 記事 |
| トークナイザ | Lindera (ko-dic) |
| 正解率 | 85.08% |
| ファイルサイズ | 約 1.8 KB |

### chinese.model

| プロパティ | 値 |
|----------|-------|
| 言語 | 中国語（簡体字・繁体字） |
| 学習コーパス | 中国語 Wikipedia 記事 |
| トークナイザ | Lindera (CC-CEDICT) |
| 正解率 | 80.72% |
| ファイルサイズ | 約 1.3 KB |

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
| ファイルサイズ | 約 17 KB |

## モデルの選択

- **日本語**には、最高精度を求める場合は `japanese.model` を、オリジナルの TinySegmenter との互換性を重視する場合は `RWCP.model` を使用
- **中国語**には `chinese.model` を使用
- **韓国語**には `korean.model` を使用
- **ドメイン固有**の用途には、[独自モデルの学習](training-guide/preparing-corpus.md)または既存モデルの[再学習](training-guide/retraining-models.md)を検討

## サンプルデータ

`resources/` ディレクトリには以下も含まれています:

- **bocchan.txt** -- 夏目漱石の小説「坊っちゃん」のサンプル日本語コーパス（約 307 KB）。ベンチマークに使用されます。
