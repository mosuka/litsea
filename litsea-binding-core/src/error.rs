//! Shared, FFI-independent error type for the language bindings.
//!
//! Every binding needs to turn a [`LitseaError`] into its host language's
//! exception type. Doing that per binding would duplicate the mapping five
//! times, so this module reduces the error to a stable [`ErrorKind`]
//! category plus a message. Each binding then writes a single conversion
//! from [`CoreError`] into its native exception (for example, an
//! [`ErrorKind::InvalidArgument`] becomes a Python `ValueError` rather than
//! a bare `RuntimeError`).

use std::fmt;

use litsea::LitseaError;

/// Category of a binding-facing error.
///
/// The set is deliberately independent of the `remote_model` feature so
/// that a binding's exception hierarchy does not change shape when the
/// feature is toggled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The caller supplied an invalid argument, such as an unknown language
    /// name or an unsupported URI scheme.
    InvalidArgument,
    /// A model could not be obtained or is not the kind the operation needs
    /// (including a failed download and a legacy joint POS model).
    Model,
    /// An input/output operation failed.
    Io,
    /// Model or training data could not be parsed.
    Parse,
    /// The operation is not supported in this build or environment.
    Unsupported,
    /// POS tagging was requested from a segmenter built on a
    /// segmentation-only model.
    PosUnavailable,
    /// A runtime failure that does not fit a more specific category.
    Runtime,
}

impl ErrorKind {
    /// Returns a stable, lowercase identifier for the kind.
    ///
    /// Bindings expose this to their host language (as an exception
    /// attribute or an error `code`), so the strings must not change once
    /// released.
    ///
    /// # Returns
    /// The identifier, for example `"invalid_argument"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidArgument => "invalid_argument",
            ErrorKind::Model => "model",
            ErrorKind::Io => "io",
            ErrorKind::Parse => "parse",
            ErrorKind::Unsupported => "unsupported",
            ErrorKind::PosUnavailable => "pos_unavailable",
            ErrorKind::Runtime => "runtime",
        }
    }
}

impl fmt::Display for ErrorKind {
    /// Writes the identifier returned by [`ErrorKind::as_str`].
    ///
    /// # Arguments
    /// * `f` - The formatter to write to.
    ///
    /// # Returns
    /// `Ok(())` on success, or a [`fmt::Error`] if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An FFI-independent error shared by all Litsea language bindings.
///
/// Carries a [`ErrorKind`] category plus a human-readable message. It is
/// `Clone` (unlike [`LitseaError`], which wraps [`std::io::Error`]) because
/// bindings frequently need to store or re-raise an error after the
/// original has been consumed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct CoreError {
    /// The error category.
    kind: ErrorKind,
    /// The human-readable message.
    message: String,
}

impl CoreError {
    /// Creates an error with the given kind and message.
    ///
    /// # Arguments
    /// * `kind` - The error category.
    /// * `message` - The human-readable message.
    ///
    /// # Returns
    /// The new [`CoreError`].
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Creates an [`ErrorKind::InvalidArgument`] error.
    ///
    /// # Arguments
    /// * `message` - The human-readable message.
    ///
    /// # Returns
    /// The new [`CoreError`].
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidArgument, message)
    }

    /// Creates an [`ErrorKind::Model`] error.
    ///
    /// # Arguments
    /// * `message` - The human-readable message.
    ///
    /// # Returns
    /// The new [`CoreError`].
    pub fn model(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Model, message)
    }

    /// Creates an [`ErrorKind::Runtime`] error.
    ///
    /// # Arguments
    /// * `message` - The human-readable message.
    ///
    /// # Returns
    /// The new [`CoreError`].
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Runtime, message)
    }

    /// Returns the error category.
    ///
    /// # Returns
    /// The [`ErrorKind`] this error was created with.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the human-readable message.
    ///
    /// # Returns
    /// The message, without the kind prefix.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<LitseaError> for CoreError {
    /// Maps a [`LitseaError`] onto a [`CoreError`].
    ///
    /// [`LitseaError::PosLearnerNotSet`] is rewritten: its message names the
    /// Rust constructor `with_two_stage_learner()`, which means nothing to a
    /// Python or Ruby caller, so the binding-facing message points at the
    /// model file instead.
    ///
    /// # Arguments
    /// * `error` - The error to convert.
    ///
    /// # Returns
    /// The equivalent [`CoreError`].
    fn from(error: LitseaError) -> Self {
        let kind = match error {
            LitseaError::Io(_) => ErrorKind::Io,
            LitseaError::InvalidData(_) => ErrorKind::Parse,
            LitseaError::InvalidInput(_) => ErrorKind::InvalidArgument,
            LitseaError::Unsupported(_) => ErrorKind::Unsupported,
            LitseaError::PosLearnerNotSet => {
                return Self::new(
                    ErrorKind::PosUnavailable,
                    "this segmenter was built from a segmentation-only model; \
                     load a two-stage POS model (for example japanese_pos.model) to tag parts of speech",
                );
            }
            // `LitseaError::Download` only exists when the `remote_model`
            // feature is enabled on the `litsea` crate.
            #[cfg(feature = "remote_model")]
            LitseaError::Download(_) => ErrorKind::Model,
            // `LitseaError` is `#[non_exhaustive]`, and `litsea/remote_model`
            // may also be enabled by another crate in the graph without this
            // crate's feature.
            _ => ErrorKind::Runtime,
        };

        Self::new(kind, error.to_string())
    }
}

/// Result alias for the binding core.
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_identifiers_are_stable() {
        assert_eq!(ErrorKind::InvalidArgument.as_str(), "invalid_argument");
        assert_eq!(ErrorKind::Model.as_str(), "model");
        assert_eq!(ErrorKind::Io.as_str(), "io");
        assert_eq!(ErrorKind::Parse.as_str(), "parse");
        assert_eq!(ErrorKind::Unsupported.as_str(), "unsupported");
        assert_eq!(ErrorKind::PosUnavailable.as_str(), "pos_unavailable");
        assert_eq!(ErrorKind::Runtime.as_str(), "runtime");
        assert_eq!(ErrorKind::Io.to_string(), "io");
    }

    #[test]
    fn test_from_litsea_error_maps_kinds() {
        let io = CoreError::from(LitseaError::Io(std::io::Error::other("boom")));
        assert_eq!(io.kind(), ErrorKind::Io);
        assert_eq!(io.message(), "boom");

        let data = CoreError::from(LitseaError::InvalidData("bad line".to_string()));
        assert_eq!(data.kind(), ErrorKind::Parse);
        assert_eq!(data.message(), "invalid data: bad line");

        let input = CoreError::from(LitseaError::InvalidInput("bad scheme".to_string()));
        assert_eq!(input.kind(), ErrorKind::InvalidArgument);

        let unsupported = CoreError::from(LitseaError::Unsupported("nope"));
        assert_eq!(unsupported.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn test_pos_learner_not_set_is_rewritten() {
        let error = CoreError::from(LitseaError::PosLearnerNotSet);
        assert_eq!(error.kind(), ErrorKind::PosUnavailable);
        assert!(
            error.message().contains("two-stage POS model"),
            "message should point at the model file, got: {}",
            error.message()
        );
        assert!(
            !error.message().contains("with_two_stage_learner"),
            "the Rust constructor name must not leak into bindings, got: {}",
            error.message()
        );
    }

    #[test]
    fn test_constructors() {
        assert_eq!(CoreError::invalid_argument("x").kind(), ErrorKind::InvalidArgument);
        assert_eq!(CoreError::model("x").kind(), ErrorKind::Model);
        assert_eq!(CoreError::runtime("x").kind(), ErrorKind::Runtime);
        assert_eq!(CoreError::model("boom").to_string(), "boom");
    }
}
