//! The `Segmenter` class.

use std::path::Path;
use std::sync::Arc;

use litsea::Language;
use litsea_binding_core::{CoreSegmenter, parse_language};
use napi::bindgen_prelude::{AsyncTask, Buffer};
use napi::{Env, Task};

use crate::error::{KindError, error_with_code, map_err, task_call};
use crate::token::Token;

/// A word segmenter, optionally with POS tagging.
///
/// Build one with `Segmenter.open`, `Segmenter.fromBytes`, or
/// `Segmenter.fromUri`. The kind of model is detected from the file itself,
/// so `hasPos` describes what was loaded rather than something the caller
/// has to declare.
#[napi(js_name = "Segmenter")]
pub struct JsSegmenter {
    /// The wrapped core segmenter, shared with the async loaders.
    inner: Arc<CoreSegmenter>,
}

/// Loads a model from a URI on a worker thread.
///
/// Downloading must not block the event loop, so it runs through
/// [`AsyncTask`], which uses libuv's threadpool.
pub struct FromUriTask {
    /// The language the model was trained for.
    language: Language,
    /// The model URI.
    uri: String,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for FromUriTask {
    type Output = CoreSegmenter;
    type JsValue = JsSegmenter;

    /// Fetches and builds the segmenter on a worker thread.
    ///
    /// # Returns
    /// The loaded segmenter.
    ///
    /// # Errors
    /// Returns the mapped core error if the URI cannot be resolved or the
    /// model is unusable.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let loaded = CoreSegmenter::from_uri_blocking(self.language, &self.uri);
        task_call(&mut self.kind, loaded)
    }

    /// Hands the loaded segmenter back to JavaScript.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The segmenter built by `compute`.
    ///
    /// # Returns
    /// The JavaScript `Segmenter` instance.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(JsSegmenter {
            inner: Arc::new(output),
        })
    }

    /// Rethrows the failure with its kind as the JavaScript `code`.
    ///
    /// # Arguments
    /// * `env` - The N-API environment.
    /// * `error` - The error `compute` returned.
    ///
    /// # Returns
    /// Always an error; the promise rejects.
    fn reject(&mut self, env: Env, error: napi::Error) -> napi::Result<Self::JsValue> {
        Err(error_with_code(env, self.kind.as_deref(), error))
    }
}

#[napi]
impl JsSegmenter {
    /// Loads a model from a filesystem path.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code (`"japanese"`, `"ja"`).
    /// * `path` - Path to the model file.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Throws an error whose `code` is `invalid_argument`, `io`, `parse`, or
    /// `model`.
    #[napi(factory)]
    pub fn open(language: String, path: String) -> Result<Self, KindError> {
        let language = map_err(parse_language(&language))?;
        Ok(Self {
            inner: Arc::new(map_err(CoreSegmenter::from_path(language, Path::new(&path)))?),
        })
    }

    /// Loads a model from raw bytes.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    /// * `data` - The model file contents.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Throws an error whose `code` is `invalid_argument`, `parse`, or
    /// `model`.
    #[napi(factory)]
    pub fn from_bytes(language: String, data: Buffer) -> Result<Self, KindError> {
        let language = map_err(parse_language(&language))?;
        Ok(Self {
            inner: Arc::new(map_err(CoreSegmenter::from_bytes(language, &data))?),
        })
    }

    /// Loads a model from a URI, resolving once it is available.
    ///
    /// Accepts a filesystem path, a `file://` path, or an `http(s)://` URL.
    /// The download runs on a worker thread, so the event loop keeps
    /// turning.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    /// * `uri` - The model URI.
    ///
    /// # Returns
    /// A promise resolving to the new `Segmenter`.
    ///
    /// # Errors
    /// Rejects with an error whose `code` is `invalid_argument`, `model`,
    /// `io`, or `parse`.
    #[napi(ts_return_type = "Promise<Segmenter>")]
    pub fn from_uri(language: String, uri: String) -> Result<AsyncTask<FromUriTask>, KindError> {
        let language = map_err(parse_language(&language))?;
        Ok(AsyncTask::new(FromUriTask {
            language,
            uri,
            kind: None,
        }))
    }

    /// The language this segmenter was built for.
    #[napi(getter)]
    pub fn language(&self) -> String {
        self.inner.language().to_string()
    }

    /// Whether this segmenter can tag parts of speech.
    #[napi(getter)]
    pub fn has_pos(&self) -> bool {
        self.inner.has_pos()
    }

    /// Splits a sentence into tokens.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, in order.
    #[napi]
    pub fn segment(&self, text: String) -> Vec<String> {
        self.inner.segment(&text)
    }

    /// Splits several sentences into tokens, reusing one scratch buffer.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// One token array per input sentence, in input order.
    #[napi]
    pub fn segment_batch(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        self.inner.segment_batch(&texts)
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, with `pos` set to `null`.
    #[napi]
    pub fn segment_tokens(&self, text: String) -> Vec<Token> {
        self.inner.segment_tokens(&text).into_iter().map(Token::from).collect()
    }

    /// Splits a sentence into tokens and tags each with a UPOS tag.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment and tag.
    ///
    /// # Returns
    /// The tagged tokens, with byte offsets into `text`.
    ///
    /// # Errors
    /// Throws an error with the `pos_unavailable` code when this segmenter
    /// was built from a segmentation-only model.
    #[napi]
    pub fn segment_with_pos(&self, text: String) -> Result<Vec<Token>, KindError> {
        Ok(map_err(self.inner.segment_with_pos(&text))?
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
    /// One tagged-token array per input sentence, in input order.
    ///
    /// # Errors
    /// Throws an error with the `pos_unavailable` code when this segmenter
    /// was built from a segmentation-only model.
    #[napi]
    pub fn segment_with_pos_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<Token>>, KindError> {
        Ok(map_err(self.inner.segment_with_pos_batch(&texts))?
            .into_iter()
            .map(|tokens| tokens.into_iter().map(Token::from).collect())
            .collect())
    }
}
