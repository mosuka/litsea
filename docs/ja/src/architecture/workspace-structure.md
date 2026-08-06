# ワークスペース構成

Litsea は 2 つのクレートとサポートディレクトリで構成される **Cargo ワークスペース**として組織されています。

## ディレクトリ構成

```text
litsea/
├── Cargo.toml              # Workspace manifest (incl. release profile)
├── Cargo.lock              # Dependency lock file
├── Makefile                # Build convenience targets
├── rustfmt.toml            # Rust formatting configuration
├── LICENSE                 # MIT
├── README.md               # Project overview
├── litsea/                 # Core library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # Module declarations, re-exports, and version
│   │   ├── adaboost.rs     # AdaBoost algorithm (word segmentation)
│   │   ├── perceptron.rs   # Averaged Perceptron (joint segmentation + POS)
│   │   ├── segmenter.rs    # Segmentation pipeline and feature templates
│   │   ├── extractor.rs    # Feature extraction from corpus
│   │   ├── trainer.rs      # Training orchestration (Trainer / PosTrainer)
│   │   ├── language.rs     # Language enum and character classification
│   │   ├── upos.rs         # UPOS tags and SegmentLabel
│   │   ├── model_io.rs     # Model URI resolution and download limits
│   │   ├── metrics.rs      # BinaryMetrics / MulticlassMetrics
│   │   └── error.rs        # LitseaError / Result
│   ├── benches/
│   │   └── bench.rs        # Criterion benchmarks
│   └── tests/
│       └── golden.rs       # Golden snapshots for every bundled model
├── litsea-cli/             # CLI binary crate
│   ├── Cargo.toml
│   ├── src/
│   │   └── main.rs         # CLI entry point
│   └── tests/
│       └── cli.rs          # CLI integration tests
├── models/                 # Pre-trained models
│   ├── japanese.model
│   ├── chinese.model
│   ├── korean.model
│   ├── japanese_pos.model
│   ├── chinese_pos.model
│   ├── korean_pos.model
│   ├── RWCP.model
│   └── JEITA_Genpaku_ChaSen_IPAdic.model
├── resources/              # Sample data and test fixtures
│   └── bocchan.txt         # Sample corpus
├── scripts/                # Corpus preparation utilities
│   ├── download_udtreebank.sh   # Download UD Treebanks (prints CoNLL-U file path)
│   ├── corpus_udtreebank.sh     # Convert CoNLL-U to Litsea corpus format
│   ├── download_wikidump.sh     # Download Wikipedia dumps
│   ├── corpus_wikidump.sh       # Convert Wikipedia dumps to corpus format
│   ├── split_sentences.sh       # Split text into sentences
│   └── wikitexts.sh             # Download and prepare Wikipedia text data
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
| `rustc-hash` | 2.1 | 内部の特徴量マップ用の高速ハッシュ |
| `thiserror` | 2.0 | エラー型の導出 |
| `reqwest` | 0.13 | HTTP/HTTPS モデル読み込み（rustls、任意の `remote_model` フィーチャー） |
| `criterion` | 0.8 | ベンチマーク（開発依存） |
| `tempfile` | 3.27 | テスト用一時ファイル（開発依存） |
| `tokio` | 1.52+ | テスト用非同期ランタイム（開発依存） |

### `litsea-cli`（CLI バイナリ）

CLI は Litsea の機能へのコマンドラインインターフェースを提供します。

| Dependency | Version | 用途 |
|-----------|---------|------|
| `clap` | 4.6 | コマンドライン引数の解析 |
| `ctrlc` | 3.5 | 学習中の Ctrl+C のグレースフルハンドリング |
| `tokio` | 1.52+ | 非同期ランタイム |
| `litsea` | 0.6 | コアライブラリ（ワークスペースメンバー、`remote_model` 有効化） |
| `tempfile` | 3.27 | 統合テスト用一時ファイル（開発依存） |

## ワークスペース設定

ワークスペースは Cargo resolver バージョン 3（Rust Edition 2024）を使用します。リリースビルドでは
thin LTO と単一のコード生成ユニット（codegen unit）を有効にし、文字単位の特徴量スコアリングの
呼び出しチェーンをコード生成ユニットをまたいでインライン化できるようにしています
（`panic = "abort"` も検討しましたが、CLI バイナリだけに適用範囲を限定できず、リリースプロファイルの
テストやベンチマークが複雑になるため見送りました）:

```toml
[workspace]
resolver = "3"
members = ["litsea", "litsea-cli"]

[workspace.package]
version = "0.6.0"
edition = "2024"
rust-version = "1.87"

[profile.release]
lto = "thin"
codegen-units = 1
```

共有依存関係はワークスペースレベルの `[workspace.dependencies]` で定義され、各クレートから `{ workspace = true }` で参照されます。
