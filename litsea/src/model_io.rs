//! Shared model loading I/O.
//!
//! Resolves a model URI (plain filesystem path, `file://` path, or
//! `http(s)://` URL when the `remote_model` feature is enabled) and returns
//! the raw model bytes. Parsing the model content is left to each learner.

use std::str::FromStr;

use crate::error::{LitseaError, Result};

/// URI scheme for loading models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelScheme {
    /// HTTP scheme for remote model loading.
    Http,
    /// HTTPS scheme for secure remote model loading.
    Https,
    /// Local file system scheme.
    File,
}

impl FromStr for ModelScheme {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "http" => Ok(ModelScheme::Http),
            "https" => Ok(ModelScheme::Https),
            "file" => Ok(ModelScheme::File),
            _ => Err(format!("Invalid model scheme: {}", s)),
        }
    }
}

/// Reads the raw bytes of a model from a URI.
///
/// Supported URI forms:
/// - a plain filesystem path (not supported on wasm32)
/// - `file://<path>` (not supported on wasm32)
/// - `http://` / `https://` URLs (requires the `remote_model` feature)
pub(crate) async fn read_model_bytes(uri: &str) -> Result<Vec<u8>> {
    let Some((scheme_str, rest)) = uri.split_once("://") else {
        return read_file_bytes(uri);
    };

    let scheme = ModelScheme::from_str(scheme_str).map_err(LitseaError::InvalidInput)?;

    match scheme {
        ModelScheme::Http | ModelScheme::Https => {
            #[cfg(not(feature = "remote_model"))]
            {
                Err(LitseaError::Unsupported(
                    "http:// and https:// scheme is not supported in this build. Use file:// URLs.",
                ))
            }
            #[cfg(feature = "remote_model")]
            {
                download(uri).await
            }
        }
        ModelScheme::File => read_file_bytes(rest),
    }
}

/// Reads a model from the local filesystem.
#[cfg(not(target_arch = "wasm32"))]
fn read_file_bytes(path: &str) -> Result<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

/// Local filesystem access is unavailable on wasm32.
#[cfg(target_arch = "wasm32")]
fn read_file_bytes(_path: &str) -> Result<Vec<u8>> {
    Err(LitseaError::Unsupported(
        "File system access is not supported in WASM environment. Use http:// or https:// URLs.",
    ))
}

/// Downloads a model over HTTP(S).
#[cfg(feature = "remote_model")]
async fn download(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .user_agent(format!("Litsea/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| LitseaError::Download(format!("failed to create HTTP client: {}", e)))?;

    let resp = client.get(url).send().await.map_err(|e| LitseaError::Download(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(LitseaError::Download(format!("HTTP {}", resp.status())));
    }

    let content = resp.bytes().await.map_err(|e| LitseaError::Download(e.to_string()))?;

    Ok(content.to_vec())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    use std::io::Write;

    use tempfile::NamedTempFile;

    #[test]
    fn test_model_scheme_from_str() {
        assert!(matches!("http".parse::<ModelScheme>(), Ok(ModelScheme::Http)));
        assert!(matches!("https".parse::<ModelScheme>(), Ok(ModelScheme::Https)));
        assert!(matches!("file".parse::<ModelScheme>(), Ok(ModelScheme::File)));
        assert!("ftp".parse::<ModelScheme>().is_err());
        assert!("".parse::<ModelScheme>().is_err());
    }

    #[tokio::test]
    async fn test_read_plain_path() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        write!(file, "feat1\t0.5")?;
        file.as_file().sync_all()?;

        let bytes = read_model_bytes(file.path().to_str().unwrap()).await?;
        assert_eq!(bytes, b"feat1\t0.5");
        Ok(())
    }

    #[tokio::test]
    async fn test_read_file_scheme() -> Result<()> {
        let mut file = NamedTempFile::new()?;
        write!(file, "feat1\t0.5")?;
        file.as_file().sync_all()?;

        let uri = format!("file://{}", file.path().to_str().unwrap());
        let bytes = read_model_bytes(&uri).await?;
        assert_eq!(bytes, b"feat1\t0.5");
        Ok(())
    }

    #[tokio::test]
    async fn test_read_invalid_scheme() {
        let result = read_model_bytes("ftp://example.com/model").await;
        assert!(matches!(result, Err(LitseaError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let result = read_model_bytes("/nonexistent/path/to.model").await;
        assert!(matches!(result, Err(LitseaError::Io(_))));
    }
}
