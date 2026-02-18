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
litsea = "0.4.0"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

> **注意:** モデルの読み込み（`load_model`）は HTTP/HTTPS URL をサポートする非同期操作のため、`tokio` が必要です。

## サポートプラットフォーム

Litsea は以下のプラットフォームでテストされています:

| OS | Architecture |
|----|-------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64 (Intel), aarch64 (Apple Silicon) |
| Windows | x86_64, aarch64 |
