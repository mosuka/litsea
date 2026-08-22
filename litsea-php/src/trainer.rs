//! Feature extraction, training, and cancellation.
//!
//! # Cancellation in PHP
//!
//! The Python binding releases the GIL and the Node.js binding moves training
//! onto libuv's threadpool, so in both a token can stop a run that is already
//! going. **PHP has neither.** A request is single-threaded, and `pcntl`
//! signal handlers cannot interrupt a blocking native call, so no PHP code
//! runs while `train()` executes.
//!
//! [`CancelToken`] is still useful - cancelling it before `train()` makes the
//! trainer stop at its first check point and write the partially trained
//! model - but in-flight cancellation is not available here. That is a
//! property of the host, not a gap in this binding.

use std::cell::RefCell;
use std::path::Path;

use ext_php_rs::prelude::*;
use litsea_binding_core::{
    CancelToken as CoreCancelToken, CoreExtractor, CorePerceptronTrainer, CoreTrainer,
    CoreTwoStageTrainer, CorpusFormat, parse_feature_set, parse_language,
};

use crate::error::map_err;
use crate::metrics::{PhpBinaryMetrics, PhpMulticlassMetrics, PhpTwoStageMetrics};

/// A flag that asks a training job to stop.
///
/// Cancelling is cooperative and is **not** an error: training stops at its
/// next check point, still writes the partially trained model, and returns
/// its metrics.
///
/// In PHP the token must be cancelled **before** `train()` is called; see the
/// module documentation for why.
#[php_class]
#[php(name = "Litsea\\CancelToken")]
#[derive(Default)]
pub struct CancelToken {
    /// The wrapped token; clones share one flag.
    inner: CoreCancelToken,
}

#[php_impl]
impl CancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The new `CancelToken`.
    pub fn __construct() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Returns the token to the "keep running" state.
    pub fn reset(&self) {
        self.inner.reset();
    }

    /// Returns whether cancellation has been requested.
    ///
    /// # Returns
    /// `true` once `cancel()` has been called.
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

impl CancelToken {
    /// Returns the token a training run should observe.
    ///
    /// # Arguments
    /// * `token` - The caller-supplied token, if any.
    ///
    /// # Returns
    /// A clone of the supplied token, or a fresh one.
    fn resolve(token: Option<&CancelToken>) -> CoreCancelToken {
        token.map_or_else(CoreCancelToken::new, |token| token.inner.clone())
    }
}

/// Extracts training features from a corpus.
#[php_class]
#[php(name = "Litsea\\Extractor")]
pub struct Extractor {
    /// The wrapped extractor.
    inner: CoreExtractor,
}

#[php_impl]
impl Extractor {
    /// Creates an extractor for a language.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    ///
    /// # Returns
    /// The new `Extractor`.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` for an unknown language.
    pub fn __construct(language: String) -> PhpResult<Self> {
        Ok(Self {
            inner: CoreExtractor::new(map_err(parse_language(&language))?),
        })
    }

    /// Extracts boundary-classification features.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the training corpus.
    /// * `features_path` - Path of the features file to write.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    /// * `tag_free` - Omit tag-dependent feature templates.
    ///
    /// # Returns
    /// Nothing; the features file is written.
    ///
    /// # Errors
    /// Throws `Litsea\IoException` if the corpus cannot be read or the
    /// output cannot be written.
    #[php(defaults(tsv = false, tag_free = false))]
    pub fn extract(
        &self,
        corpus_path: String,
        features_path: String,
        tsv: bool,
        tag_free: bool,
    ) -> PhpResult<()> {
        map_err(self.inner.extract(
            Path::new(&corpus_path),
            Path::new(&features_path),
            CorpusFormat::from_tsv_flag(tsv),
            tag_free,
        ))
    }

    /// Extracts two-stage (segmentation + POS) features.
    ///
    /// Writes `{outputPrefix}.stage1`, `.stage2`, and `.lexicon`.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the POS-tagged training corpus.
    /// * `output_prefix` - Prefix for the three output files.
    /// * `feature_set` - `"full"`, `"balanced"`, or `"fast"`; `null`
    ///   selects `"fast"`.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    ///
    /// # Returns
    /// Nothing; the three files are written.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` for an unknown feature set,
    /// or `Litsea\IoException` on I/O failure.
    // `defaults` only accepts const-evaluable literals, and a `String`
    // default is not one, so the fallback lives in the body.
    #[php(defaults(feature_set = None, tsv = false))]
    pub fn extract_two_stage(
        &self,
        corpus_path: String,
        output_prefix: String,
        feature_set: Option<String>,
        tsv: bool,
    ) -> PhpResult<()> {
        let feature_set = map_err(parse_feature_set(feature_set.as_deref().unwrap_or("fast")))?;
        map_err(self.inner.extract_two_stage(
            Path::new(&corpus_path),
            Path::new(&output_prefix),
            feature_set,
            CorpusFormat::from_tsv_flag(tsv),
        ))
    }
}

/// Trains a segmentation model.
#[php_class]
#[php(name = "Litsea\\Trainer")]
pub struct Trainer {
    /// The wrapped trainer; `RefCell` because PHP methods take `&self`.
    inner: RefCell<CoreTrainer>,
}

#[php_impl]
impl Trainer {
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
    /// Throws `Litsea\IoException` or `Litsea\ParseException` if the features
    /// file cannot be read.
    pub fn __construct(
        threshold: f64,
        num_iterations: i64,
        features_path: String,
    ) -> PhpResult<Self> {
        let inner = map_err(CoreTrainer::new(
            threshold,
            num_iterations.max(0) as usize,
            Path::new(&features_path),
        ))?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Loads an existing model to continue training from it.
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// Nothing; the model is merged into the learner.
    ///
    /// # Errors
    /// Throws `Litsea\ModelException`, `Litsea\IoException`, or
    /// `Litsea\ParseException`.
    pub fn load_model(&self, model_uri: String) -> PhpResult<()> {
        map_err(self.inner.borrow_mut().load_model_blocking(&model_uri))
    }

    /// Trains the model and writes it to `modelPath`.
    ///
    /// Blocks the request until training finishes.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`, which must already be cancelled
    ///   to have any effect.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Throws `Litsea\IoException` if the model cannot be written.
    #[php(defaults(cancel = None))]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&CancelToken>,
    ) -> PhpResult<PhpBinaryMetrics> {
        let token = CancelToken::resolve(cancel);
        let metrics = map_err(self.inner.borrow_mut().train(&token, Path::new(&model_path)))?;
        Ok(PhpBinaryMetrics::from(metrics))
    }
}

/// Trains a label-agnostic Averaged Perceptron model.
#[php_class]
#[php(name = "Litsea\\PerceptronTrainer")]
pub struct PerceptronTrainer {
    /// The wrapped trainer.
    inner: RefCell<CorePerceptronTrainer>,
}

#[php_impl]
impl PerceptronTrainer {
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
    /// Throws `Litsea\IoException` or `Litsea\ParseException` if the features
    /// file cannot be read.
    pub fn __construct(num_epochs: i64, features_path: String) -> PhpResult<Self> {
        let inner = map_err(CorePerceptronTrainer::new(
            num_epochs.max(0) as usize,
            Path::new(&features_path),
        ))?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Trains the model and writes it to `modelPath`.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Throws `Litsea\IoException` if the model cannot be written.
    #[php(defaults(cancel = None))]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&CancelToken>,
    ) -> PhpResult<PhpMulticlassMetrics> {
        let token = CancelToken::resolve(cancel);
        let metrics = map_err(self.inner.borrow_mut().train(&token, Path::new(&model_path)))?;
        Ok(PhpMulticlassMetrics::from(metrics))
    }
}

/// Trains a two-stage segmentation + POS model.
///
/// A trainer can only be used once: training collapses stage 1 into an
/// AdaBoost model, which consumes it. `isAvailable()` reports the state, and
/// a second `train()` throws.
#[php_class]
#[php(name = "Litsea\\TwoStageTrainer")]
pub struct TwoStageTrainer {
    /// The wrapped trainer.
    inner: RefCell<CoreTwoStageTrainer>,
}

#[php_impl]
impl TwoStageTrainer {
    /// Loads a two-stage features prefix and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `features_prefix` - Prefix of the `.stage1` / `.stage2` /
    ///   `.lexicon` files.
    /// * `dominance` - Lexicon dominance threshold in `(0.5, 1.0]`.
    ///
    /// # Returns
    /// The new `TwoStageTrainer`.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` if `dominance` is out of
    /// range, or `Litsea\IoException` / `Litsea\ParseException` if the
    /// feature files cannot be read.
    #[php(defaults(dominance = 0.99))]
    pub fn __construct(
        num_epochs: i64,
        features_prefix: String,
        dominance: f64,
    ) -> PhpResult<Self> {
        let inner = map_err(CoreTwoStageTrainer::new(
            num_epochs.max(0) as usize,
            dominance,
            Path::new(&features_prefix),
        ))?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Returns whether this trainer can still be used.
    ///
    /// # Returns
    /// `false` once `train()` has run.
    pub fn is_available(&self) -> bool {
        self.inner.borrow().is_available()
    }

    /// Trains both stages and writes the model.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// The metrics of both stages.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` if the trainer has already
    /// been used, or `Litsea\IoException` if the model cannot be written.
    #[php(defaults(cancel = None))]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&CancelToken>,
    ) -> PhpResult<PhpTwoStageMetrics> {
        let token = CancelToken::resolve(cancel);
        let metrics = map_err(self.inner.borrow_mut().train(&token, Path::new(&model_path)))?;
        Ok(PhpTwoStageMetrics::from(metrics))
    }
}
