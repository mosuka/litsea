//! The `Segmenter` class.

use js_sys::Array;
use litsea_binding_core::{CoreSegmenter, parse_language};
use wasm_bindgen::prelude::*;

use crate::error::map_err;
use crate::token::Token;

/// A word segmenter, optionally with POS tagging.
///
/// Built from model bytes: there is no URI constructor, because reqwest's
/// wasm backend cannot build with the timeouts `litsea` sets. Fetch the
/// model in JavaScript and pass the bytes, which also leaves caching, CORS,
/// and progress reporting to the page that knows about them.
///
/// WebAssembly objects are not garbage collected, so call `free()` when a
/// segmenter is no longer needed - a POS model can hold several megabytes.
#[wasm_bindgen]
pub struct Segmenter {
    /// The wrapped core segmenter.
    inner: CoreSegmenter,
}

#[wasm_bindgen]
impl Segmenter {
    /// Builds a segmenter from raw model bytes.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code (`"japanese"`, `"ja"`).
    /// * `data` - The model file contents.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Throws an error whose `code` is `invalid_argument`, `parse`, or
    /// `model`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(language: &str, data: &[u8]) -> Result<Segmenter, JsValue> {
        let language = map_err(parse_language(language))?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes(language, data))?,
        })
    }

    /// The language this segmenter was built for.
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.inner.language().to_string()
    }

    /// Whether this segmenter can tag parts of speech.
    #[wasm_bindgen(getter, js_name = hasPos)]
    pub fn has_pos(&self) -> bool {
        self.inner.has_pos()
    }

    /// Splits a sentence into tokens.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// An array of strings, in order.
    #[wasm_bindgen]
    pub fn segment(&self, text: &str) -> Vec<String> {
        self.inner.segment(text)
    }

    /// Splits several sentences into tokens, reusing one scratch buffer.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// An array of string arrays, in input order.
    #[wasm_bindgen(js_name = segmentBatch)]
    pub fn segment_batch(&self, texts: Vec<String>) -> Array {
        self.inner
            .segment_batch(&texts)
            .into_iter()
            .map(|tokens| tokens.into_iter().map(JsValue::from).collect::<Array>())
            .map(JsValue::from)
            .collect()
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// An array of `Token`s with `pos` unset.
    #[wasm_bindgen(js_name = segmentTokens)]
    pub fn segment_tokens(&self, text: &str) -> Vec<Token> {
        self.inner.segment_tokens(text).into_iter().map(Token::from).collect()
    }

    /// Splits a sentence into tokens and tags each with a UPOS tag.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment and tag.
    ///
    /// # Returns
    /// An array of tagged `Token`s with byte offsets into `text`.
    ///
    /// # Errors
    /// Throws an error with the `pos_unavailable` code when this segmenter
    /// was built from a segmentation-only model.
    #[wasm_bindgen(js_name = segmentWithPos)]
    pub fn segment_with_pos(&self, text: &str) -> Result<Vec<Token>, JsValue> {
        Ok(map_err(self.inner.segment_with_pos(text))?
            .into_iter()
            .map(Token::from)
            .collect())
    }

    /// Splits and tags several sentences.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment and tag.
    ///
    /// # Returns
    /// An array of `Token` arrays, in input order.
    ///
    /// # Errors
    /// Throws an error with the `pos_unavailable` code when this segmenter
    /// was built from a segmentation-only model.
    #[wasm_bindgen(js_name = segmentWithPosBatch)]
    pub fn segment_with_pos_batch(&self, texts: Vec<String>) -> Result<Array, JsValue> {
        Ok(map_err(self.inner.segment_with_pos_batch(&texts))?
            .into_iter()
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|view| JsValue::from(Token::from(view)))
                    .collect::<Array>()
            })
            .map(JsValue::from)
            .collect())
    }
}
