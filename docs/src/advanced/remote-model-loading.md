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
learner.load_model("./resources/japanese.model").await?;

// HTTP URL
learner.load_model("https://example.com/models/japanese.model").await?;
```

## Implementation Details

- HTTP client: **reqwest** with **rustls** (no OpenSSL dependency)
- Custom User-Agent: `Litsea/<version>`
- The `load_model` method is **async** because HTTP loading requires an async runtime
- For the CLI, `tokio` provides the async runtime

## WASM Considerations

On `wasm32` targets:

- **Local file paths are not supported** -- file system access is unavailable
- **`file://` scheme is not supported**
- **HTTP/HTTPS loading works** via the browser's fetch API (through reqwest's WASM support)

Error messages guide users to use URLs instead of file paths when running in WASM.
