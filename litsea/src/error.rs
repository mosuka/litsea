//! Error types for the litsea library.

/// Errors returned by litsea operations.
///
/// Marked `#[non_exhaustive]`: new variants are added as the library grows
/// (e.g. `Download` and `PosLearnerNotSet` were later additions), so
/// external matches must carry a wildcard arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    ///
    /// Holds `&'static str` (unlike the owned `String` variants) on
    /// purpose: every message is a compile-time constant, so an owned
    /// string would allocate for no benefit.
    #[error("unsupported: {0}")]
    Unsupported(&'static str),

    /// `segment_with_pos` was called on a segmenter without a two-stage
    /// learner.
    #[error("POS learner is not set; build the segmenter with with_two_stage_learner()")]
    PosLearnerNotSet,

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
