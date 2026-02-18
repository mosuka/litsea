# Installation

## Prerequisites

- **Rust 1.87 or later** (stable channel) from [rust-lang.org](https://www.rust-lang.org/)
- **Cargo** (Rust's package manager, included with Rust)

## Installing the CLI Tool

### From crates.io

```sh
cargo install litsea-cli
```

### From Source

```sh
git clone https://github.com/mosuka/litsea.git
cd litsea
cargo build --release
```

The binary will be available at `./target/release/litsea`.

Verify the installation:

```sh
./target/release/litsea --help
```

## Using as a Library

Add Litsea to your project's `Cargo.toml`:

```toml
[dependencies]
litsea = "0.4.0"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

> **Note:** `tokio` is required because model loading (`load_model`) is an async operation that supports HTTP/HTTPS URLs.

## Supported Platforms

Litsea is tested on the following platforms:

| OS | Architecture |
|----|-------------|
| Linux | x86_64, aarch64 |
| macOS | x86_64 (Intel), aarch64 (Apple Silicon) |
| Windows | x86_64, aarch64 |
