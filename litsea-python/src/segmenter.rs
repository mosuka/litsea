//! The `Segmenter` class.

use std::path::PathBuf;

use litsea_binding_core::CoreSegmenter;
use pyo3::prelude::*;

use crate::error::map_err;
use crate::language::{LanguageArg, PyLanguage};
use crate::upos::PyToken;

/// A word segmenter, optionally with POS tagging.
///
/// Construct one with `Segmenter.open`, `Segmenter.from_bytes`, or
/// `Segmenter.from_uri`. The kind of model is detected from the file, so a
/// two-stage POS model produces a segmenter where `has_pos` is `True` and
/// `segment_with_pos` works; a segmentation-only model produces one where it
/// raises `PosUnavailableError`.
///
/// Instances are immutable and safe to share between threads. Reusing one
/// across calls is the intended usage: an internal scratch buffer reaches a
/// steady state where segmentation allocates only the output strings.
#[pyclass(name = "Segmenter", frozen, module = "litsea")]
pub struct PySegmenter {
    /// The wrapped core segmenter.
    inner: CoreSegmenter,
}

#[pymethods]
impl PySegmenter {
    /// Loads a model from a filesystem path.
    ///
    /// # Arguments
    /// * `language` - A `Language` member or its name (`"ja"`, `"japanese"`).
    /// * `path` - Path to the model file.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Raises `IoError` if the file cannot be read, `ParseError` if it is
    /// malformed, `ModelError` if it is a legacy joint POS model, or
    /// `InvalidArgumentError` for an unknown language.
    #[staticmethod]
    fn open(py: Python<'_>, language: LanguageArg, path: PathBuf) -> PyResult<Self> {
        let language = language.resolve()?;
        let inner = py.detach(|| map_err(CoreSegmenter::from_path(language, &path)))?;
        Ok(Self { inner })
    }

    /// Loads a model from raw bytes.
    ///
    /// # Arguments
    /// * `language` - A `Language` member or its name.
    /// * `data` - The model file contents.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Raises `ParseError` if the bytes are malformed, `ModelError` if they
    /// are a legacy joint POS model, or `InvalidArgumentError` for an
    /// unknown language.
    #[staticmethod]
    fn from_bytes(language: LanguageArg, data: &[u8]) -> PyResult<Self> {
        let language = language.resolve()?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes(language, data))?,
        })
    }

    /// Loads a model from a URI, blocking until it is available.
    ///
    /// Accepts a filesystem path, a `file://` path, or an `http(s)://` URL.
    /// The GIL is released while the model is fetched.
    ///
    /// # Arguments
    /// * `language` - A `Language` member or its name.
    /// * `uri` - The model URI.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Raises `ModelError` if the download fails, plus the same errors as
    /// `Segmenter.open`.
    #[staticmethod]
    fn from_uri(py: Python<'_>, language: LanguageArg, uri: String) -> PyResult<Self> {
        let language = language.resolve()?;
        let inner = py.detach(|| map_err(CoreSegmenter::from_uri_blocking(language, &uri)))?;
        Ok(Self { inner })
    }

    /// The language this segmenter was built for.
    #[getter]
    fn language(&self) -> PyLanguage {
        PyLanguage::from(self.inner.language())
    }

    /// Whether this segmenter can tag parts of speech.
    #[getter]
    fn has_pos(&self) -> bool {
        self.inner.has_pos()
    }

    /// Splits a sentence into tokens.
    ///
    /// This method keeps the GIL: releasing it would require copying the
    /// input string (PyO3 forbids touching Python-owned memory with the GIL
    /// released), which costs more than segmenting one sentence. Use
    /// `segment_batch` for bulk work, which does release it.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, in order.
    fn segment(&self, text: &str) -> Vec<String> {
        self.inner.segment(text)
    }

    /// Splits several sentences into tokens, releasing the GIL.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// One token list per input sentence, in input order.
    fn segment_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<String>> {
        py.detach(|| self.inner.segment_batch(&texts))
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, with `pos` set to `None`.
    fn segment_tokens(&self, text: &str) -> Vec<PyToken> {
        self.inner.segment_tokens(text).into_iter().map(PyToken::from).collect()
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
    /// Raises `PosUnavailableError` when this segmenter was built from a
    /// segmentation-only model.
    fn segment_with_pos(&self, text: &str) -> PyResult<Vec<PyToken>> {
        Ok(map_err(self.inner.segment_with_pos(text))?
            .into_iter()
            .map(PyToken::from)
            .collect())
    }

    /// Splits and tags several sentences, releasing the GIL.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment and tag.
    ///
    /// # Returns
    /// One tagged-token list per input sentence, in input order.
    ///
    /// # Errors
    /// Raises `PosUnavailableError` when this segmenter was built from a
    /// segmentation-only model.
    fn segment_with_pos_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
    ) -> PyResult<Vec<Vec<PyToken>>> {
        let batches = py.detach(|| map_err(self.inner.segment_with_pos_batch(&texts)))?;
        Ok(batches
            .into_iter()
            .map(|tokens| tokens.into_iter().map(PyToken::from).collect())
            .collect())
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `Segmenter(language='japanese', has_pos=True)`.
    fn __repr__(&self) -> String {
        format!(
            "Segmenter(language='{}', has_pos={})",
            self.inner.language(),
            if self.inner.has_pos() { "True" } else { "False" }
        )
    }
}
