//! Error types for the litsea library.

/// Errors returned by litsea operations.
#[derive(Debug, thiserror::Error)]
pub enum LitseaError {
    /// I/O failure while reading or writing files.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Model or training data content could not be parsed.
    #[error("invalid data: {0}")]
    InvalidData(String),

    /// Invalid caller input, such as an unknown URI scheme or an attempt to
    /// save an empty model.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The operation is not supported in this build or environment
    /// (e.g. remote models without the `remote_model` feature, or file
    /// system access on wasm32).
    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    /// Downloading a remote model failed.
    #[cfg(feature = "remote_model")]
    #[error("failed to download model: {0}")]
    Download(String),
}

/// Convenience alias for `Result` with [`LitseaError`].
pub type Result<T> = std::result::Result<T, LitseaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        let err = LitseaError::InvalidData("bad line".to_string());
        assert_eq!(err.to_string(), "invalid data: bad line");

        let err = LitseaError::Unsupported("no file system");
        assert_eq!(err.to_string(), "unsupported: no file system");
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: LitseaError = io_err.into();
        assert!(matches!(err, LitseaError::Io(_)));
    }
}
