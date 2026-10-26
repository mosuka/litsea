//! The `Segmenter` class.

use js_sys::{Array, Object};
use litsea_binding_core::{CoreSegmenter, build_user_dictionary_with_tags, parse_language};
use wasm_bindgen::prelude::*;

use crate::error::{invalid_argument, map_err};
use crate::token::Token;

/// Reads the `userWordTags` object of `fromBytes` (issue #264) into
/// `(word, tag)` pairs.
///
/// # Arguments
/// * `user_word_tags` - An object mapping words to UPOS tags, or
///   `undefined` / `null`.
///
/// # Returns
/// The object's own enumerable string-keyed entries, empty without an
/// object.
///
/// # Errors
/// Throws an error with the `invalid_argument` code if a value is not a
/// string; whether a string is a UPOS tag is checked by the caller.
fn word_tags(user_word_tags: Option<Object>) -> Result<Vec<(String, String)>, JsValue> {
    let Some(user_word_tags) = user_word_tags else {
        return Ok(Vec::new());
    };
    Object::entries(&user_word_tags)
        .iter()
        .map(|entry| {
            let entry = Array::from(&entry);
            // `Object.entries` keys are always strings.
            let word = entry.get(0).as_string().unwrap_or_default();
            let tag = entry.get(1).as_string().ok_or_else(|| {
                invalid_argument(format!(
                    "the UPOS tag for the user dictionary word {word:?} must be a string"
                ))
            })?;
            Ok((word, tag))
        })
        .collect()
}

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
    /// * `user_words` - Optional words to emit as single tokens (issue
    ///   #263); `userWords` in JavaScript.
    /// * `user_words_gate` - Optional flip-ratio gate for `userWords`, in
    ///   `0.0..=1.0`; `userWordsGate` in JavaScript.
    /// * `user_word_tags` - Optional object mapping words to emit as single
    ///   tokens to the UPOS tag `segmentWithPos` gives them (issue #264);
    ///   `userWordTags` in JavaScript.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Throws an error whose `code` is `invalid_argument` (also for a tag
    /// that is not a UPOS tag or a gate outside `0.0..=1.0`), `parse`, or
    /// `model`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(
        language: &str,
        data: &[u8],
        // Every trailing argument names its optional TypeScript type. With
        // an override on `userWordTags` alone, wasm-bindgen 0.2.127 emitted
        // `userWords` and `userWordsGate` without `?`, which would make them
        // required for TypeScript callers.
        #[wasm_bindgen(js_name = userWords, unchecked_optional_param_type = "string[] | null")]
        user_words: Option<Vec<String>>,
        #[wasm_bindgen(js_name = userWordsGate, unchecked_optional_param_type = "number | null")]
        user_words_gate: Option<f64>,
        #[wasm_bindgen(
            js_name = userWordTags,
            unchecked_optional_param_type = "Record<string, string> | null"
        )]
        user_word_tags: Option<Object>,
    ) -> Result<Segmenter, JsValue> {
        let language = map_err(parse_language(language))?;
        let word_tags = word_tags(user_word_tags)?;
        let dictionary = map_err(build_user_dictionary_with_tags(
            user_words.as_deref().unwrap_or_default(),
            &word_tags,
            user_words_gate,
        ))?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes_with(language, data, dictionary))?,
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
