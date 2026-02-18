# ワークスペース構成

Litsea は 2 つのクレートとサポートディレクトリで構成される **Cargo ワークスペース**として組織されています。

## ディレクトリ構成

```text
litsea/
├── Cargo.toml              # Workspace manifest
├── Cargo.lock              # Dependency lock file
├── Makefile                # Build convenience targets
├── rustfmt.toml            # Rust formatting configuration
├── LICENSE                 # MIT
├── README.md               # Project overview
├── litsea/                 # Core library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # Module declarations and version
│   │   ├── adaboost.rs     # AdaBoost algorithm
│   │   ├── segmenter.rs    # Word segmentation
│   │   ├── extractor.rs    # Feature extraction from corpus
│   │   ├── trainer.rs      # Training orchestration
│   │   ├── language.rs     # Language definitions and char patterns
│   │   └── util.rs         # URI scheme utilities
│   └── benches/
│       └── bench.rs        # Criterion benchmarks
├── litsea-cli/             # CLI binary crate
│   ├── Cargo.toml
│   └── src/
│       └── main.rs         # CLI entry point
├── resources/              # Pre-trained models and sample data
│   ├── japanese.model
│   ├── chinese.model
│   ├── korean.model
│   ├── RWCP.model
│   ├── JEITA_Genpaku_ChaSen_IPAdic.model
│   └── bocchan.txt         # Sample corpus
├── scripts/                # Corpus preparation utilities
│   ├── wikitexts.sh        # Download Wikipedia texts
│   └── corpus.sh           # Create training corpus with Lindera
├── docs/                   # mdbook documentation (this book)
└── .github/
    └── workflows/          # CI/CD pipelines
        ├── regression.yml  # Test on push/PR
        ├── release.yml     # Release builds and publishing
        └── periodic.yml    # Weekly stability tests
```

## クレートの詳細

### `litsea`（コアライブラリ）

コアライブラリは、分割、学習、モデル I/O のすべての機能を提供します。

| Dependency | Version | 用途 |
|-----------|---------|------|
| `regex` | 1.12 | 文字種パターンマッチング |
| `reqwest` | 0.13 | HTTP/HTTPS モデル読み込み（rustls） |
| `tokio` | 1.49 | リモートモデル読み込み用非同期ランタイム |
| `criterion` | 0.8 | ベンチマーク（開発依存） |
| `tempfile` | 3.25 | テスト用一時ファイル（開発依存） |

### `litsea-cli`（CLI バイナリ）

CLI は Litsea の機能へのコマンドラインインターフェースを提供します。

| Dependency | Version | 用途 |
|-----------|---------|------|
| `clap` | 4.5 | コマンドライン引数の解析 |
| `ctrlc` | 3.5 | 学習中の Ctrl+C のグレースフルハンドリング |
| `icu_segmenter` | 2.1 | Unicode UAX #29 文分割 |
| `tokio` | 1.49 | 非同期ランタイム |
| `litsea` | 0.4 | コアライブラリ（ワークスペースメンバー） |

## ワークスペース設定

ワークスペースは Cargo resolver バージョン 3（Rust Edition 2024）を使用します:

```toml
[workspace]
resolver = "3"
members = ["litsea", "litsea-cli"]

[workspace.package]
version = "0.4.0"
edition = "2024"
rust-version = "1.87"
```

共有依存関係はワークスペースレベルの `[workspace.dependencies]` で定義され、各クレートから `{ workspace = true }` で参照されます。
