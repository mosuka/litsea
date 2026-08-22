# Language Bindings

Litsea is a Rust library, but it is also usable from other languages. The bindings live in this repository as workspace members, so they are versioned and released together with `litsea` itself.

## Crates

| Crate | Target runtime | FFI stack | Status |
|-------|----------------|-----------|--------|
| [`litsea-binding-core`](bindings/binding-core.md) | (shared, no FFI) | — | Available |
| [`litsea-python`](bindings/python.md) | Python 3.10+ | PyO3 + maturin | Available |
| [`litsea-nodejs`](bindings/nodejs.md) | Node.js 20+ | napi-rs | Available |
| [`litsea-php`](bindings/php.md) | PHP 8.1+ | ext-php-rs | Available |
| `litsea-ruby` | Ruby 3.1+ | magnus + rb-sys | Planned ([#205](https://github.com/mosuka/litsea/issues/205)) |
| `litsea-wasm` | Browser / Deno | wasm-bindgen | Planned ([#206](https://github.com/mosuka/litsea/issues/206)) |

## Design principles

These apply to every binding.

### Models are never embedded

A binding package ships code only. The caller supplies a model as raw bytes, a filesystem path, a `file://` path, or an `http(s)://` URL. The bundled models range from 84 KB to 8 MB each, so embedding all four languages would push a wheel or npm package past 20 MB; keeping models external also means a model can be updated without republishing the binding.

See [Pre-trained Models](pre-trained-models.md) for where to get a model.

### The model kind is detected, not declared

The CLI needs `--pos` to know whether it is loading a segmentation model or a two-stage POS model. Bindings do not: they read the model bytes once and dispatch on the detected kind, so `has_pos` is a property of the loaded model rather than something the caller has to get right.

### Cancellation is explicit

Training can run for a long time, and `litsea`'s trainers stop early when their `running` flag is cleared. The CLI drives that flag from a Ctrl-C handler, but a library must not install one: signal handling is process-global and the host language usually owns it already. Bindings therefore expose a cancellation token that the caller triggers.

### Shared logic lives in one crate

Everything that is not FFI-specific — language-name parsing, model loading and kind detection, token offsets, trainer orchestration, error categories — lives in [`litsea-binding-core`](bindings/binding-core.md). Each binding only maps that surface onto its host language's types and exception model.
