# Workspace Structure

Litsea is organized as a **Cargo workspace** with two crates and supporting directories.

## Directory Layout

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

## Crate Details

### `litsea` (Core Library)

The core library provides all segmentation, training, and model I/O functionality.

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `rustc-hash` | 2.1 | Fast hashing for internal feature maps |
| `thiserror` | 2.0 | Error type derivation |
| `reqwest` | 0.13 | HTTP/HTTPS model loading (rustls, optional `remote_model` feature) |
| `criterion` | 0.8 | Benchmarking (dev dependency) |
| `tempfile` | 3.27 | Temporary files for tests (dev dependency) |
| `tokio` | 1.52+ | Async runtime for tests (dev dependency) |

### `litsea-cli` (CLI Binary)

The CLI provides a command-line interface to Litsea's functionality.

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `clap` | 4.6 | Command-line argument parsing |
| `ctrlc` | 3.5 | Graceful Ctrl+C handling during training |
| `tokio` | 1.52+ | Async runtime |
| `litsea` | 0.5 | Core library (workspace member) |
| `tempfile` | 3.27 | Temporary files for integration tests (dev dependency) |

## Workspace Configuration

The workspace uses Cargo resolver version 3 (Rust Edition 2024). Release
builds enable thin LTO and a single codegen unit so the per-character
feature-scoring call chain can be inlined across codegen units
(`panic = "abort"` was considered and rejected — it cannot be scoped to the
CLI binary and would complicate release-profile tests and benches):

```toml
[workspace]
resolver = "3"
members = ["litsea", "litsea-cli"]

[workspace.package]
version = "0.5.0"
edition = "2024"
rust-version = "1.87"

[profile.release]
lto = "thin"
codegen-units = 1
```

Shared dependencies are defined at the workspace level in `[workspace.dependencies]` and referenced by each crate with `{ workspace = true }`.
