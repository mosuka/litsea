//! Node.js binding for Litsea.
//!
//! Built with [napi-rs](https://napi.rs). Everything FFI-independent lives
//! in `litsea-binding-core`, so this crate only maps that surface onto
//! JavaScript values, promises, and errors.
//!
//! Blocking work - downloading a model, extracting features, training -
//! runs through `AsyncTask` on libuv's threadpool, so the event loop keeps
//! turning and a `CancelToken` can stop a job that is already running.
//!
//! ```js
//! const { Segmenter } = require('litsea')
//!
//! const seg = Segmenter.open('japanese', 'models/japanese.model')
//! seg.segment('これはテストです。')
//! ```

#[macro_use]
extern crate napi_derive;

pub mod error;
pub mod metrics;
pub mod segmenter;
pub mod token;
pub mod trainer;

/// Returns the version of the underlying `litsea` crate.
///
/// # Returns
/// The version string, for example `"0.12.0"`.
#[napi]
pub fn version() -> String {
    litsea::version().to_string()
}

/// Returns the names of every supported language.
///
/// # Returns
/// The canonical names, in documentation order.
#[napi]
pub fn supported_languages() -> Vec<String> {
    litsea_binding_core::supported_language_names()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_matches_litsea() {
        assert_eq!(version(), litsea::version());
    }

    #[test]
    fn test_supported_languages() {
        assert_eq!(supported_languages(), vec!["japanese", "chinese", "korean", "english"]);
    }
}
