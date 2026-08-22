//! Feature extraction, training, and cancellation.
//!
//! Extraction and training are blocking CPU work, so they run through
//! [`AsyncTask`] on libuv's threadpool rather than on the event loop. That
//! is also what makes [`JsCancelToken`] useful: JavaScript keeps running
//! while a job is in flight, so it can cancel one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use litsea_binding_core::{
    CancelToken, CoreExtractor, CorePerceptronTrainer, CoreTrainer, CoreTwoStageTrainer,
    CorpusFormat, parse_feature_set, parse_language,
};
use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Task};

use crate::error::{KindError, error_with_code, map_err, task_call};
use crate::metrics::{JsBinaryMetrics, JsMulticlassMetrics, JsTwoStageMetrics};

/// A flag that asks a running training job to stop.
///
/// Cancelling is cooperative and is **not** an error: training stops at its
/// next check point, still writes the partially trained model, and resolves
/// with its metrics. The binding never installs a signal handler.
#[napi(js_name = "CancelToken")]
#[derive(Default)]
pub struct JsCancelToken {
    /// The wrapped token; clones share one flag.
    inner: CancelToken,
}

#[napi]
impl JsCancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The new `CancelToken`.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    #[napi]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Returns the token to the "keep running" state.
    #[napi]
    pub fn reset(&self) {
        self.inner.reset();
    }

    /// Whether cancellation has been requested.
    #[napi(getter)]
    pub fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

impl JsCancelToken {
    /// Returns the token a job should observe.
    ///
    /// # Arguments
    /// * `token` - The caller-supplied token, if any.
    ///
    /// # Returns
    /// A clone of the supplied token, or a fresh one.
    fn resolve(token: Option<&JsCancelToken>) -> CancelToken {
        token.map_or_else(CancelToken::new, |token| token.inner.clone())
    }
}

/// Extracts boundary-classification features on a worker thread.
pub struct ExtractTask {
    /// The extractor, built on the main thread.
    extractor: CoreExtractor,
    /// Path to the training corpus.
    corpus_path: PathBuf,
    /// Path of the features file to write.
    features_path: PathBuf,
    /// The corpus layout.
    format: CorpusFormat,
    /// Whether to omit tag-dependent templates.
    tag_free: bool,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for ExtractTask {
    type Output = ();
    type JsValue = ();

    /// Runs the extraction.
    ///
    /// # Returns
    /// `()` once the features file has been written.
    ///
    /// # Errors
    /// Returns the mapped core error on I/O failure.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let extracted = self.extractor.extract(
            &self.corpus_path,
            &self.features_path,
            self.format,
            self.tag_free,
        );
        task_call(&mut self.kind, extracted)
    }

    /// Resolves the promise with `undefined`.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The unit value from `compute`.
    ///
    /// # Returns
    /// `()`.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
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

/// Extracts two-stage features on a worker thread.
pub struct ExtractTwoStageTask {
    /// The extractor, built on the main thread.
    extractor: CoreExtractor,
    /// Path to the POS-tagged training corpus.
    corpus_path: PathBuf,
    /// Prefix for the three output files.
    output_prefix: PathBuf,
    /// Which feature templates stage 2 uses.
    feature_set: litsea::TwoStageFeatureSet,
    /// The corpus layout.
    format: CorpusFormat,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for ExtractTwoStageTask {
    type Output = ();
    type JsValue = ();

    /// Runs the extraction.
    ///
    /// # Returns
    /// `()` once all three files have been written.
    ///
    /// # Errors
    /// Returns the mapped core error on I/O failure.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let extracted = self.extractor.extract_two_stage(
            &self.corpus_path,
            &self.output_prefix,
            self.feature_set,
            self.format,
        );
        task_call(&mut self.kind, extracted)
    }

    /// Resolves the promise with `undefined`.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The unit value from `compute`.
    ///
    /// # Returns
    /// `()`.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
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

/// Extracts training features from a corpus.
#[napi(js_name = "Extractor")]
pub struct JsExtractor {
    /// The language the corpus is written in.
    language: litsea::Language,
}

#[napi]
impl JsExtractor {
    /// Creates an extractor for a language.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    ///
    /// # Returns
    /// The new `Extractor`.
    ///
    /// # Errors
    /// Throws an error with the `invalid_argument` code for an unknown
    /// language.
    #[napi(constructor)]
    pub fn new(language: String) -> Result<Self, KindError> {
        Ok(Self {
            language: map_err(parse_language(&language))?,
        })
    }

    /// Extracts boundary-classification features off the event loop.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the training corpus.
    /// * `features_path` - Path of the features file to write.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    /// * `tag_free` - Omit tag-dependent feature templates.
    ///
    /// # Returns
    /// A promise resolving once the features file has been written.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn extract(
        &self,
        corpus_path: String,
        features_path: String,
        tsv: Option<bool>,
        tag_free: Option<bool>,
    ) -> AsyncTask<ExtractTask> {
        AsyncTask::new(ExtractTask {
            extractor: CoreExtractor::new(self.language),
            corpus_path: PathBuf::from(corpus_path),
            features_path: PathBuf::from(features_path),
            format: CorpusFormat::from_tsv_flag(tsv.unwrap_or(false)),
            tag_free: tag_free.unwrap_or(false),
            kind: None,
        })
    }

    /// Extracts two-stage (segmentation + POS) features off the event loop.
    ///
    /// Writes `{outputPrefix}.stage1`, `.stage2`, and `.lexicon`.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the POS-tagged training corpus.
    /// * `output_prefix` - Prefix for the three output files.
    /// * `feature_set` - `"full"`, `"balanced"`, or `"fast"`; defaults to
    ///   `"fast"`.
    /// * `tsv` - Read the corpus in the space-preserving TSV format.
    ///
    /// # Returns
    /// A promise resolving once all three files have been written.
    ///
    /// # Errors
    /// Throws an error with the `invalid_argument` code for an unknown
    /// feature set.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn extract_two_stage(
        &self,
        corpus_path: String,
        output_prefix: String,
        feature_set: Option<String>,
        tsv: Option<bool>,
    ) -> Result<AsyncTask<ExtractTwoStageTask>, KindError> {
        let feature_set = map_err(parse_feature_set(feature_set.as_deref().unwrap_or("fast")))?;
        Ok(AsyncTask::new(ExtractTwoStageTask {
            extractor: CoreExtractor::new(self.language),
            corpus_path: PathBuf::from(corpus_path),
            output_prefix: PathBuf::from(output_prefix),
            feature_set,
            format: CorpusFormat::from_tsv_flag(tsv.unwrap_or(false)),
            kind: None,
        }))
    }
}

/// Trains a segmentation model on a worker thread.
pub struct TrainTask {
    /// The trainer, shared with the handle so the task can take it.
    trainer: Arc<Mutex<CoreTrainer>>,
    /// Path of the model file to write.
    model_path: PathBuf,
    /// The cancellation flag training observes.
    cancel: CancelToken,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for TrainTask {
    type Output = litsea::BinaryMetrics;
    type JsValue = JsBinaryMetrics;

    /// Runs the training.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Returns the mapped core error if the model cannot be written.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let mut trainer = self.trainer.lock().unwrap_or_else(PoisonError::into_inner);
        let trained = trainer.train(&self.cancel, &self.model_path);
        task_call(&mut self.kind, trained)
    }

    /// Resolves the promise with the metrics.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The metrics from `compute`.
    ///
    /// # Returns
    /// The JavaScript metrics object.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(JsBinaryMetrics::from(output))
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

/// Trains a segmentation model.
#[napi(js_name = "Trainer")]
pub struct JsTrainer {
    /// The wrapped trainer, shared with the worker thread.
    inner: Arc<Mutex<CoreTrainer>>,
}

#[napi]
impl JsTrainer {
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
    /// Throws an error with the `io` or `parse` code if the features file
    /// cannot be read.
    #[napi(constructor)]
    pub fn new(
        threshold: f64,
        num_iterations: u32,
        features_path: String,
    ) -> Result<Self, KindError> {
        let inner = map_err(CoreTrainer::new(
            threshold,
            num_iterations as usize,
            Path::new(&features_path),
        ))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Trains the model off the event loop and writes it to `modelPath`.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`; cancelling stops training early
    ///   and still writes the partially trained model.
    ///
    /// # Returns
    /// A promise resolving to the training metrics.
    #[napi(ts_return_type = "Promise<BinaryMetrics>")]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&JsCancelToken>,
    ) -> AsyncTask<TrainTask> {
        AsyncTask::new(TrainTask {
            trainer: Arc::clone(&self.inner),
            model_path: PathBuf::from(model_path),
            cancel: JsCancelToken::resolve(cancel),
            kind: None,
        })
    }
}

/// Trains an Averaged Perceptron model on a worker thread.
pub struct PerceptronTrainTask {
    /// The trainer, shared with the handle.
    trainer: Arc<Mutex<CorePerceptronTrainer>>,
    /// Path of the model file to write.
    model_path: PathBuf,
    /// The cancellation flag training observes.
    cancel: CancelToken,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for PerceptronTrainTask {
    type Output = litsea::MulticlassMetrics;
    type JsValue = JsMulticlassMetrics;

    /// Runs the training.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Returns the mapped core error if the model cannot be written.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let mut trainer = self.trainer.lock().unwrap_or_else(PoisonError::into_inner);
        let trained = trainer.train(&self.cancel, &self.model_path);
        task_call(&mut self.kind, trained)
    }

    /// Resolves the promise with the metrics.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The metrics from `compute`.
    ///
    /// # Returns
    /// The JavaScript metrics object.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(JsMulticlassMetrics::from(output))
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

/// Trains a label-agnostic Averaged Perceptron model.
#[napi(js_name = "PerceptronTrainer")]
pub struct JsPerceptronTrainer {
    /// The wrapped trainer, shared with the worker thread.
    inner: Arc<Mutex<CorePerceptronTrainer>>,
}

#[napi]
impl JsPerceptronTrainer {
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
    /// Throws an error with the `io` or `parse` code if the features file
    /// cannot be read.
    #[napi(constructor)]
    pub fn new(num_epochs: u32, features_path: String) -> Result<Self, KindError> {
        let inner =
            map_err(CorePerceptronTrainer::new(num_epochs as usize, Path::new(&features_path)))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Trains the model off the event loop and writes it to `modelPath`.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// A promise resolving to the training metrics.
    #[napi(ts_return_type = "Promise<MulticlassMetrics>")]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&JsCancelToken>,
    ) -> AsyncTask<PerceptronTrainTask> {
        AsyncTask::new(PerceptronTrainTask {
            trainer: Arc::clone(&self.inner),
            model_path: PathBuf::from(model_path),
            cancel: JsCancelToken::resolve(cancel),
            kind: None,
        })
    }
}

/// Trains a two-stage model on a worker thread.
pub struct TwoStageTrainTask {
    /// The trainer, emptied by a successful run.
    trainer: Arc<Mutex<CoreTwoStageTrainer>>,
    /// Path of the model file to write.
    model_path: PathBuf,
    /// The cancellation flag training observes.
    cancel: CancelToken,
    /// The failing error kind, recorded for `reject`.
    kind: Option<String>,
}

impl Task for TwoStageTrainTask {
    type Output = litsea::TwoStageMetrics;
    type JsValue = JsTwoStageMetrics;

    /// Runs the training.
    ///
    /// # Returns
    /// The metrics of both stages.
    ///
    /// # Errors
    /// Returns an `invalid_argument` error if the trainer has already been
    /// used, or the mapped core error if the model cannot be written.
    fn compute(&mut self) -> napi::Result<Self::Output> {
        let mut trainer = self.trainer.lock().unwrap_or_else(PoisonError::into_inner);
        let trained = trainer.train(&self.cancel, &self.model_path);
        task_call(&mut self.kind, trained)
    }

    /// Resolves the promise with the metrics.
    ///
    /// # Arguments
    /// * `_env` - The N-API environment.
    /// * `output` - The metrics from `compute`.
    ///
    /// # Returns
    /// The JavaScript metrics object.
    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(JsTwoStageMetrics::from(output))
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

/// Trains a two-stage segmentation + POS model.
///
/// A trainer can only be used once: training collapses stage 1 into an
/// AdaBoost model, which consumes it. `available` reports the state, and a
/// second `train()` rejects with an `invalid_argument` error.
#[napi(js_name = "TwoStageTrainer")]
pub struct JsTwoStageTrainer {
    /// The wrapped trainer, shared with the worker thread.
    inner: Arc<Mutex<CoreTwoStageTrainer>>,
}

#[napi]
impl JsTwoStageTrainer {
    /// Loads a two-stage features prefix and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `features_prefix` - Prefix of the `.stage1` / `.stage2` /
    ///   `.lexicon` files.
    /// * `dominance` - Lexicon dominance threshold in `(0.5, 1.0]`;
    ///   defaults to 0.99.
    ///
    /// # Returns
    /// The new `TwoStageTrainer`.
    ///
    /// # Errors
    /// Throws an error with the `invalid_argument` code if `dominance` is
    /// out of range, or `io` / `parse` if the feature files cannot be read.
    #[napi(constructor)]
    pub fn new(
        num_epochs: u32,
        features_prefix: String,
        dominance: Option<f64>,
    ) -> Result<Self, KindError> {
        let inner = map_err(CoreTwoStageTrainer::new(
            num_epochs as usize,
            dominance.unwrap_or(0.99),
            Path::new(&features_prefix),
        ))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    /// Whether this trainer can still be used.
    #[napi(getter)]
    pub fn available(&self) -> bool {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).is_available()
    }

    /// Trains both stages off the event loop and writes the model.
    ///
    /// # Arguments
    /// * `model_path` - Path of the model file to write.
    /// * `cancel` - Optional `CancelToken`.
    ///
    /// # Returns
    /// A promise resolving to the metrics of both stages.
    ///
    /// # Errors
    /// Rejects with an `invalid_argument` error if the trainer has already
    /// been used.
    #[napi(ts_return_type = "Promise<TwoStageMetrics>")]
    pub fn train(
        &self,
        model_path: String,
        cancel: Option<&JsCancelToken>,
    ) -> AsyncTask<TwoStageTrainTask> {
        AsyncTask::new(TwoStageTrainTask {
            trainer: Arc::clone(&self.inner),
            model_path: PathBuf::from(model_path),
            cancel: JsCancelToken::resolve(cancel),
            kind: None,
        })
    }
}
