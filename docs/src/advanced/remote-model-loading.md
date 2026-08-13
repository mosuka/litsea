# Remote Model Loading

Litsea supports loading models from HTTP/HTTPS URLs in addition to local files.

## Supported URI Schemes

| Scheme | Example | Description |
|--------|---------|-------------|
| (none) | `./model.model` | Local file path (default) |
| `file://` | `file:///path/to/model` | Explicit file URI |
| `http://` | `http://example.com/model` | HTTP URL |
| `https://` | `https://example.com/model` | HTTPS URL |

## CLI Usage

```sh
echo "テスト" | litsea segment -l japanese https://example.com/japanese.model
```

## Library Usage

```rust
let mut learner = AdaBoost::new(0.01, 100);

// Local file
learner.load_model_from_path(Path::new("./models/japanese.model"))?; // local, synchronous

// HTTP URL
learner.load_model("https://example.com/models/japanese.model").await?;
```

## Enabling the Feature

Since 0.6.0 the `remote_model` feature is **opt-in** (the library default is
local loading only, keeping the dependency tree compact). The CLI enables it,
so `litsea segment https://...` keeps working out of the box; library users
need:

```toml
litsea = { version = "0.8.0", features = ["remote_model"] }
```

## Implementation Details

- HTTP client: **reqwest** with **rustls** (no OpenSSL dependency)
- Custom User-Agent: `Litsea/<version>`
- The `load_model` method is **async** because HTTP loading requires an async runtime
- For the CLI, `tokio` provides the async runtime

## Limits and Failure Handling

- **Connect timeout**: 10 seconds; **overall request timeout**: 60 seconds --
  a stalled server can no longer block model loading indefinitely
- **Maximum model size**: 256 MiB. A larger advertised `Content-Length` is
  rejected before the body is read, and an oversized body is rejected after
- **Incomplete downloads**: when the server sends a `Content-Length`, a
  shorter received body is reported as an incomplete download
- Non-2xx responses are reported as download errors with the HTTP status
- The model parser additionally rejects truncated files (a model without its
  trailing bias line fails to load; see
  [Model File Format](model-file-format.md))

## WASM Considerations

The `wasm32` target is CI-checked only **without** the `remote_model` feature
(`cargo check -p litsea --target wasm32-unknown-unknown --no-default-features`):

- **HTTP/HTTPS loading is not currently supported on `wasm32`** -- the HTTP
  client configuration uses reqwest's `ClientBuilder::connect_timeout` and
  `timeout`, which reqwest's WASM client does not provide, so the
  `remote_model` feature does not build for this target
- **Local file paths and the `file://` scheme are not supported either** --
  file system access is unavailable, so `read_file_bytes` returns an
  `Unsupported` error

Models must therefore be supplied by other means on `wasm32`, e.g. by calling
`load_model_from_reader` with model bytes provided by the host environment.
