//! Feature extraction, training, and cancellation classes.

use std::path::PathBuf;

use litsea_binding_core::{
    CancelToken, CoreExtractor, CorePerceptronTrainer, CoreTrainer, CoreTwoStageTrainer,
    CorpusFormat, parse_feature_set,
};
use pyo3::prelude::*;

use crate::error::map_err;
use crate::language::LanguageArg;
use crate::metrics::{PyBinaryMetrics, PyMulticlassMetrics, PyTwoStageMetrics};

/// A flag that asks a running training job to stop.
///
/// Cancelling is cooperative and is **not** an error: the trainer stops at
/// its next check point, still writes the partially trained model, and
/// returns its metrics normally. Training releases the GIL, so another
/// Python thread can cancel a run in progress.
///
/// The binding never installs a signal handler - handling Ctrl-C is the
/// application's decision, and Python already owns SIGINT.
#[pyclass(name = "CancelToken", frozen, skip_from_py_object, module = "litsea")]
#[derive(Debug, Clone, Default)]
pub struct PyCancelToken {
    /// The wrapped token.
    inner: CancelToken,
}

#[pymethods]
impl PyCancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The new `CancelToken`.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    fn cancel(&self) {
        self.inner.cancel();
    }

    /// Returns the token to the "keep running" state.
    fn reset(&self) {
        self.inner.reset();
    }

    /// Whether cancellation has been requested.
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `CancelToken(cancelled=False)`.
    fn __repr__(&self) -> String {
        format!(
            "CancelToken(cancelled={})",
            if self.inner.is_cancelled() { "True" } else { "False" }
        )
    }
}

impl PyCancelToken {
    /// Returns the wrapped token, or a fresh one when no token was given.
    ///
    /// # Arguments
    /// * `token` - The caller-supplied token, if any.
    ///
    /// # Returns
    /// The token training should observe.
    fn resolve(token: Option<&PyCancelToken>) -> CancelToken {
        token.map_or_else(CancelToken::new, |token| token.inner.clone())
    }
}

/// Extracts training features from a corpus.
#[pyclass(name = "Extractor", frozen, module = "litsea")]
pub struct PyExtractor {
    /// The wrapped extractor.
    inner: CoreExtractor,
}

#[pymethods]
impl PyExtractor {
    /// Creates an extractor for a language.
    ///
    /// # Arguments
    /// * `language` - A `Language` member or its name.
    ///
    /// # Returns
    /// The new `Extractor`.
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` for an unknown language.
    #[new]
    fn new(language: LanguageArg) -> PyResult<Self> {
        Ok(Self {
            inner: CoreExtractor::new(language.resolve()?),
        })
    }

    /// Extracts boundary-classification features, releasing the GIL.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the training corpus.
    /// * `features_path` - Path of the features file to write.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    /// * `tag_free` - Omit tag-dependent feature templates.
    ///
    /// # Returns
    /// `None` once the features file has been written.
    ///
    /// # Errors
    /// Raises `IoError` if the corpus cannot be read or the output cannot be
    /// written.
    #[pyo3(signature = (corpus_path, features_path, *, tsv=false, tag_free=false))]
    fn extract(
        &self,
        py: Python<'_>,
        corpus_path: PathBuf,
        features_path: PathBuf,
        tsv: bool,
        tag_free: bool,
    ) -> PyResult<()> {
        py.detach(|| {
            map_err(self.inner.extract(
                &corpus_path,
                &features_path,
                CorpusFormat::from_tsv_flag(tsv),
                tag_free,
            ))
        })
    }

    /// Extracts two-stage (segmentation + POS) features, releasing the GIL.
    ///
    /// Writes `{output_prefix}.stage1`, `.stage2`, and `.lexicon`.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the POS-tagged training corpus.
    /// * `output_prefix` - Prefix for the three output files.
    /// * `feature_set` - `"full"`, `"balanced"`, or `"fast"`.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    ///
    /// # Returns
    /// `None` once all three files have been written.
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` for an unknown feature set, or
    /// `IoError` if the files cannot be read or written.
    #[pyo3(signature = (corpus_path, output_prefix, *, feature_set="fast", tsv=false))]
    fn extract_two_stage(
        &self,
        py: Python<'_>,
        corpus_path: PathBuf,
        output_prefix: PathBuf,
        feature_set: &str,
        tsv: bool,
    ) -> PyResult<()> {
        let feature_set = map_err(parse_feature_set(feature_set))?;
        py.detach(|| {
            map_err(self.inner.extract_two_stage(
                &corpus_path,
                &output_prefix,
                feature_set,
                CorpusFormat::from_tsv_flag(tsv),
            ))
        })
    }
}

/// Trains a segmentation model.
#[pyclass(name = "Trainer", module = "litsea")]
pub struct PyTrainer {
    /// The wrapped trainer.
    inner: CoreTrainer,
}

#[pymethods]
impl PyTrainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `threshold` - Early-stopping threshold for weak classifiers.
    /// * `num_iterations` - Maximum number of boosting iterations.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new `Trainer`.
    ///
    /// # Errors
    /// Raises `IoError` or `ParseError` if the features file cannot be read
    /// or parsed.
    #[new]
    fn new(
        py: Python<'_>,
        threshold: f64,
        num_iterations: usize,
        features_path: PathBuf,
    ) -> PyResult<Self> {
        let inner =
            py.detach(|| map_err(CoreTrainer::new(threshold, num_iterations, &features_path)))?;
        Ok(Self { inner })
    }

    /// Loads an existing model to continue training from it.
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `None` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Raises `ModelError`, `IoError`, or `ParseError` if the model cannot
    /// be fetched or parsed.
    fn load_model(&mut self, py: Python<'_>, model_uri: String) -> PyResult<()> {
        py.detach(|| map_err(self.inner.load_model_blocking(&model_uri)))
    }

    /// Trains the model and writes it to `model_path`, releasing the GIL.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`; cancelling stops training early
    ///   and still writes the partially trained model.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Raises `IoError` if the model cannot be written.
    #[pyo3(signature = (model_path, *, cancel=None))]
    fn train(
        &mut self,
        py: Python<'_>,
        model_path: PathBuf,
        cancel: Option<&PyCancelToken>,
    ) -> PyResult<PyBinaryMetrics> {
        let token = PyCancelToken::resolve(cancel);
        let metrics = py.detach(|| map_err(self.inner.train(&token, &model_path)))?;
        Ok(PyBinaryMetrics::from(metrics))
    }
}

/// Trains a label-agnostic Averaged Perceptron model.
#[pyclass(name = "PerceptronTrainer", module = "litsea")]
pub struct PyPerceptronTrainer {
    /// The wrapped trainer.
    inner: CorePerceptronTrainer,
}

#[pymethods]
impl PyPerceptronTrainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new `PerceptronTrainer`.
    ///
    /// # Errors
    /// Raises `IoError` or `ParseError` if the features file cannot be read
    /// or parsed.
    #[new]
    fn new(py: Python<'_>, num_epochs: usize, features_path: PathBuf) -> PyResult<Self> {
        let inner =
            py.detach(|| map_err(CorePerceptronTrainer::new(num_epochs, &features_path)))?;
        Ok(Self { inner })
    }

    /// Loads an existing model to continue training from it.
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `None` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Raises `ModelError`, `IoError`, or `ParseError` if the model cannot
    /// be fetched or parsed.
    fn load_model(&mut self, py: Python<'_>, model_uri: String) -> PyResult<()> {
        py.detach(|| map_err(self.inner.load_model_blocking(&model_uri)))
    }

    /// Trains the model and writes it to `model_path`, releasing the GIL.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Raises `IoError` if the model cannot be written.
    #[pyo3(signature = (model_path, *, cancel=None))]
    fn train(
        &mut self,
        py: Python<'_>,
        model_path: PathBuf,
        cancel: Option<&PyCancelToken>,
    ) -> PyResult<PyMulticlassMetrics> {
        let token = PyCancelToken::resolve(cancel);
        let metrics = py.detach(|| map_err(self.inner.train(&token, &model_path)))?;
        Ok(PyMulticlassMetrics::from(metrics))
    }
}

/// Trains a two-stage segmentation + POS model.
///
/// A trainer can only be used once: `litsea` collapses stage 1 into an
/// AdaBoost model during training, which consumes the trainer. Check
/// `available` before reusing one.
#[pyclass(name = "TwoStageTrainer", module = "litsea")]
pub struct PyTwoStageTrainer {
    /// The wrapped trainer.
    inner: CoreTwoStageTrainer,
}

#[pymethods]
impl PyTwoStageTrainer {
    /// Loads a two-stage features prefix and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `dominance` - Lexicon dominance threshold, in `(0.5, 1.0]`.
    /// * `features_prefix` - Prefix of the `.stage1` / `.stage2` /
    ///   `.lexicon` files written by `Extractor.extract_two_stage`.
    ///
    /// # Returns
    /// The new `TwoStageTrainer`.
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` if `dominance` is out of range, or
    /// `IoError` / `ParseError` if the feature files cannot be read.
    #[new]
    #[pyo3(signature = (num_epochs, features_prefix, *, dominance=0.99))]
    fn new(
        py: Python<'_>,
        num_epochs: usize,
        features_prefix: PathBuf,
        dominance: f64,
    ) -> PyResult<Self> {
        let inner = py.detach(|| {
            map_err(CoreTwoStageTrainer::new(num_epochs, dominance, &features_prefix))
        })?;
        Ok(Self { inner })
    }

    /// Whether this trainer can still be used.
    #[getter]
    fn available(&self) -> bool {
        self.inner.is_available()
    }

    /// Trains both stages and writes the model, releasing the GIL.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// The metrics of both stages.
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` if the trainer has already been used,
    /// or `IoError` if the model cannot be written.
    #[pyo3(signature = (model_path, *, cancel=None))]
    fn train(
        &mut self,
        py: Python<'_>,
        model_path: PathBuf,
        cancel: Option<&PyCancelToken>,
    ) -> PyResult<PyTwoStageMetrics> {
        let token = PyCancelToken::resolve(cancel);
        let metrics = py.detach(|| map_err(self.inner.train(&token, &model_path)))?;
        Ok(PyTwoStageMetrics::from(metrics))
    }
}
