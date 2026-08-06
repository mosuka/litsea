# インストール

## 前提条件

- **Rust 1.87 以降**（stable チャンネル）-- [rust-lang.org](https://www.rust-lang.org/) から入手
- **Cargo**（Rust のパッケージマネージャ、Rust に同梱）

## CLI ツールのインストール

### crates.io から

```sh
cargo install litsea-cli
```

### ソースから

```sh
git clone https://github.com/mosuka/litsea.git
cd litsea
cargo build --release
```

バイナリは `./target/release/litsea` に生成されます。

インストールの確認:

```sh
./target/release/litsea --help
```

## ライブラリとしての利用

プロジェクトの `Cargo.toml` に Litsea を追加します:

```toml
[dependencies]
litsea = "0.6.0"
```

http(s) 経由のリモートモデル読み込みはオプトイン（opt-in）です。必要な場合は `remote_model` フィーチャーを有効にしてください:

```toml
litsea = { version = "0.6.0", features = ["remote_model"] }
```

> **注意:** ローカルファイルからのモデル読み込み（`load_model_from_path`）は同期処理のため、非同期ランタイムは不要です。非同期ランタイム（`tokio` など）が必要になるのは、非同期の `load_model` メソッドを使って HTTP/HTTPS 経由でモデルを読み込む場合のみです（`remote_model` フィーチャーで有効化され、このフィーチャーはデフォルトで有効です）。

## サポートプラットフォーム

Litsea は以下のプラットフォームでテストされています:

| OS | Architecture |
|----|-------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64 (Intel), aarch64 (Apple Silicon) |
| Windows | x86_64, aarch64 |
