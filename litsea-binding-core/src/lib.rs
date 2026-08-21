//! FFI-independent helpers shared by the Litsea language bindings
//! (`litsea-python`, `litsea-nodejs`, `litsea-php`, `litsea-ruby`,
//! `litsea-wasm`).
//!
//! Each binding wraps the [`litsea`] crate for its own FFI layer (PyO3,
//! napi, ext-php-rs, magnus, wasm-bindgen). Without a shared layer, all five
//! would reimplement the same plumbing: parsing a language name, deciding
//! whether a model file is a segmentation model or a two-stage POS model,
//! keeping a reusable [`litsea::SegmentBuffer`], attaching byte offsets to
//! tokens, driving the trainers with a cancellation flag, and turning
//! [`litsea::LitseaError`] into something the host language can raise. That
//! logic lives here as plain Rust, so it can be unit-tested without any FFI
//! toolchain.
//!
//! # Model loading
//!
//! Models are never embedded in a binding package (they range from 84 KB to
//! 8 MB); the caller supplies one as bytes, a filesystem path, or a URI:
//!
//! ```no_run
//! use litsea::Language;
//! use litsea_binding_core::CoreSegmenter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let segmenter = CoreSegmenter::from_path(Language::Japanese, "japanese.model".as_ref())?;
//! assert!(!segmenter.has_pos());
//! println!("{:?}", segmenter.segment("すもももももももものうち"));
//! # Ok(())
//! # }
//! ```
//!
//! The model kind is detected from the file itself
//! ([`litsea::ModelKind::detect`]), so bindings do not expose a `--pos`-style
//! flag: loading `japanese_pos.model` yields a segmenter where
//! [`CoreSegmenter::has_pos`] is `true` and
//! [`CoreSegmenter::segment_with_pos`] works.
//!
//! # Platform support
//!
//! [`trainer`] and the blocking model loaders are unavailable on
//! `wasm32-unknown-unknown`: feature extraction and training are file-based,
//! and there is no filesystem. WASM callers fetch the model bytes in
//! JavaScript and use [`CoreSegmenter::from_bytes`].

pub mod cancel;
pub mod error;
pub mod language;
pub mod model;
#[cfg(not(target_arch = "wasm32"))]
pub mod runtime;
pub mod segmenter;
pub mod token;
#[cfg(not(target_arch = "wasm32"))]
pub mod trainer;

pub use cancel::CancelToken;
pub use error::{CoreError, CoreResult, ErrorKind};
pub use language::{SUPPORTED_LANGUAGES, language_name, parse_language, supported_language_names};
#[cfg(not(target_arch = "wasm32"))]
pub use model::read_model_file;
pub use model::{BuiltSegmenter, build_segmenter, read_model_uri};
pub use segmenter::CoreSegmenter;
pub use token::TokenView;
#[cfg(not(target_arch = "wasm32"))]
pub use trainer::{
    CoreExtractor, CorePerceptronTrainer, CoreTrainer, CoreTwoStageTrainer, CorpusFormat,
    parse_feature_set,
};

/// Returns the version of the `litsea-binding-core` crate.
///
/// Bindings expose this next to [`litsea::version`] so a host-language user
/// can report both in a bug report.
///
/// # Returns
/// The crate's `CARGO_PKG_VERSION`, for example `"0.12.0"`.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_matches_the_workspace_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
        // Every workspace crate shares `workspace.package.version`.
        assert_eq!(version(), litsea::version());
    }
}
