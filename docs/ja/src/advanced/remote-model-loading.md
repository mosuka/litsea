# リモートモデルの読み込み

Litsea は、ローカルファイルに加えて HTTP/HTTPS URL からのモデル読み込みに対応しています。

## 対応する URI スキーム

| スキーム | 例 | 説明 |
|--------|---------|-------------|
| (なし) | `./model.model` | ローカルファイルパス（デフォルト） |
| `file://` | `file:///path/to/model` | 明示的な File URI |
| `http://` | `http://example.com/model` | HTTP URL |
| `https://` | `https://example.com/model` | HTTPS URL |

## CLI での使用

```sh
echo "テスト" | litsea segment -l japanese https://example.com/japanese.model
```

## ライブラリでの使用

```rust
let mut learner = AdaBoost::new(0.01, 100);

// ローカルファイル
learner.load_model("./resources/japanese.model").await?;

// HTTP URL
learner.load_model("https://example.com/models/japanese.model").await?;
```

## 実装の詳細

- HTTP クライアント: **reqwest** + **rustls**（OpenSSL 依存なし）
- カスタム User-Agent: `Litsea/<version>`
- `load_model` メソッドが**非同期（async）**なのは、HTTP 読み込みに非同期ランタイムが必要なため
- CLI では `tokio` が非同期ランタイムを提供

## WASM に関する注意事項

`wasm32` ターゲットでは:

- **ローカルファイルパスは非対応** -- ファイルシステムへのアクセスが利用できない
- **`file://` スキームは非対応**
- **HTTP/HTTPS の読み込みは動作する** -- ブラウザの fetch API 経由（reqwest の WASM サポート）

WASM 環境で実行する場合、ファイルパスの代わりに URL を使用するようエラーメッセージが案内します。
