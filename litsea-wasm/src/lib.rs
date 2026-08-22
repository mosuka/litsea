//! WebAssembly binding for Litsea.
//!
//! Built with [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/).
//! Everything FFI-independent lives in `litsea-binding-core`, so this crate
//! only maps that surface onto JavaScript values and errors.
//!
//! Two things the host removes, both measured rather than assumed:
//!
//! - **No `fromUri`.** reqwest's wasm backend cannot build with the timeouts
//!   `litsea::model_io` sets, so the page fetches the model and passes the
//!   bytes to [`Segmenter::from_bytes`].
//! - **No training.** `litsea`'s extractor and trainers are path-based, and
//!   wasm32 has no filesystem.
//!
//! ```js
//! import init, { Segmenter } from 'litsea-wasm'
//!
//! await init()
//! const bytes = new Uint8Array(await (await fetch('/models/japanese.model')).arrayBuffer())
//! const seg = Segmenter.fromBytes('japanese', bytes)
//! seg.segment('これはテストです。')
//! ```

pub mod error;
pub mod segmenter;
pub mod token;

use wasm_bindgen::prelude::*;

/// Returns the version of the underlying `litsea` crate.
///
/// # Returns
/// The version string, for example `"0.12.0"`.
#[wasm_bindgen]
pub fn version() -> String {
    litsea::version().to_string()
}

/// Returns the names of every supported language.
///
/// # Returns
/// The canonical names, in documentation order.
#[wasm_bindgen(js_name = supportedLanguages)]
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
