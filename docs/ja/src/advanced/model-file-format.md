# モデルファイル形式

Litsea のモデルは、シンプルなプレーンテキストファイルとして保存されます。

## 形式の仕様

```text
<feature_name>\t<weight>
<feature_name>\t<weight>
...
<bias>
```

- 最終行を除く各行は、タブ文字で区切られた**特徴量名**と**重み**を含む
- **重みがゼロの特徴量**は、ファイルをコンパクトに保つために省略される
- **最終行**はバイアス項を単一の数値として含む

## 例

```text
BC1:IK	0.3456
BC2:KI	-0.1234
UW4:は	0.5678
UC4:I	0.2345
...
-0.0891
```

## バイアスの復元

モデルの読み込み時に、バイアスは以下の式で復元されます:

```text
bias_bucket_weight = -bias_value * 2 - sum(all_feature_weights)
```

予測時:

```text
bias = -sum(all_model_weights) / 2.0
score = bias + sum(model[feature] for feature in input_attributes)
```

## ファイルサイズ

モデルファイルは非常にコンパクトです:

| モデル | サイズ | 特徴量 |
|-------|------|----------|
| japanese.model | 約 2.9 KB | Wikipedia で学習 |
| korean.model | 約 1.8 KB | Wikipedia で学習 |
| chinese.model | 約 1.3 KB | Wikipedia で学習 |
| RWCP.model | 約 22 KB | オリジナルの TinySegmenter |
| JEITA_Genpaku_ChaSen_IPAdic.model | 約 17 KB | JEITA コーパス |

コンパクトなサイズは Litsea の主要な利点の一つです。モデルはアプリケーションに直接埋め込んだり、最小限のオーバーヘッドで HTTP 経由で配信したりできます。

## 互換性

- モデルファイルは**エンコーディング非依存**です（特徴量名はそのまま保存されます）
- 形式は**決定的**です（特徴量は BTreeMap により整列されます）
- モデルは**前方互換性**があります。入力に含まれるがモデルにない新しい特徴量は、予測時に単純に無視されます
