# Workspace Structure

Litsea is organized as a **Cargo workspace** with two crates and supporting directories.

## Directory Layout

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
├── models/                 # Pre-trained models
│   ├── japanese.model
│   ├── chinese.model
│   ├── korean.model
│   ├── RWCP.model
│   └── JEITA_Genpaku_ChaSen_IPAdic.model
├── resources/              # Sample data and test fixtures
│   └── bocchan.txt         # Sample corpus
├── scripts/                # Corpus preparation utilities
│   └── corpus.sh           # Download UD Treebanks and create training corpora
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
| `regex` | 1.12 | Character type pattern matching |
| `reqwest` | 0.13 | HTTP/HTTPS model loading (rustls) |
| `tokio` | 1.49 | Async runtime for remote model loading |
| `criterion` | 0.8 | Benchmarking (dev dependency) |
| `tempfile` | 3.25 | Temporary files for tests (dev dependency) |

### `litsea-cli` (CLI Binary)

The CLI provides a command-line interface to Litsea's functionality.

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `clap` | 4.5 | Command-line argument parsing |
| `ctrlc` | 3.5 | Graceful Ctrl+C handling during training |
| `icu_segmenter` | 2.1 | Unicode UAX #29 sentence segmentation |
| `tokio` | 1.49 | Async runtime |
| `litsea` | 0.4 | Core library (workspace member) |

## Workspace Configuration

The workspace uses Cargo resolver version 3 (Rust Edition 2024):

```toml
[workspace]
resolver = "3"
members = ["litsea", "litsea-cli"]

[workspace.package]
version = "0.4.0"
edition = "2024"
rust-version = "1.87"
```

Shared dependencies are defined at the workspace level in `[workspace.dependencies]` and referenced by each crate with `{ workspace = true }`.
