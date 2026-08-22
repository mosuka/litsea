//! Feature extraction, training, and cancellation.
//!
//! Training releases the GVL (see [`crate::gvl`]), so other Ruby threads keep
//! running while it works - which is what makes [`CancelToken`] usable while
//! a run is already going, as in the Python and Node.js bindings.

use std::cell::RefCell;
use std::path::PathBuf;

use litsea_binding_core::{
    CancelToken as CoreCancelToken, CoreExtractor, CorePerceptronTrainer, CoreTrainer,
    CoreTwoStageTrainer, CorpusFormat, parse_feature_set,
};
use magnus::{
    Module, Object, RModule, Ruby, Value, error::Error, function, method, scan_args::scan_args,
};

use crate::error::map_err;
use crate::gvl::without_gvl;
use crate::language::language_from_value;
use crate::metrics::{RbBinaryMetrics, RbMulticlassMetrics, RbTwoStageMetrics};

/// A flag that asks a running training job to stop.
///
/// Cancelling is cooperative and is **not** an error: training stops at its
/// next check point, still writes the partially trained model, and returns
/// its metrics. Because training releases the GVL, another Ruby thread can
/// cancel a run that is already going.
#[magnus::wrap(class = "Litsea::CancelToken", free_immediately, size)]
pub struct CancelToken {
    /// The wrapped token; clones share one flag.
    inner: CoreCancelToken,
}

impl CancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The new token.
    fn new() -> Self {
        Self {
            inner: CoreCancelToken::new(),
        }
    }

    /// Requests cancellation.
    fn cancel(&self) {
        self.inner.cancel();
    }

    /// Returns the token to the "keep running" state.
    fn reset(&self) {
        self.inner.reset();
    }

    /// Returns whether cancellation has been requested.
    ///
    /// # Returns
    /// `true` once `cancel` has been called.
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// Returns the flag a training run should observe.
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

/// Reads the optional `cancel:` keyword argument.
///
/// magnus cannot express an optional keyword argument in a `method!` arity,
/// so the trainers take their arguments through `scan_args`.
///
/// # Arguments
/// * `args` - The raw Ruby arguments.
///
/// # Returns
/// The model path and the cancellation flag to observe.
///
/// # Errors
/// Returns a Ruby `ArgumentError` if the arguments do not match.
fn scan_train_args(args: &[magnus::Value]) -> Result<(PathBuf, CoreCancelToken), Error> {
    let args = scan_args::<(String,), (), (), (), _, ()>(args)?;
    let (model_path,) = args.required;
    let kwargs = magnus::scan_args::get_kwargs::<_, (), (Option<Option<&CancelToken>>,), ()>(
        args.keywords,
        &[],
        &["cancel"],
    )?;
    let (cancel,) = kwargs.optional;

    Ok((PathBuf::from(model_path), CancelToken::resolve(cancel.flatten())))
}

/// Extracts training features from a corpus.
#[magnus::wrap(class = "Litsea::Extractor", free_immediately, size)]
pub struct Extractor {
    /// The wrapped extractor.
    inner: CoreExtractor,
}

impl Extractor {
    /// Creates an extractor for a language.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code, as a String or Symbol.
    ///
    /// # Returns
    /// The new extractor.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` for an unknown language.
    fn new(language: Value) -> Result<Self, Error> {
        Ok(Self {
            inner: CoreExtractor::new(language_from_value(language)?),
        })
    }

    /// Extracts boundary-classification features.
    ///
    /// Accepts `tsv:` and `tag_free:` keyword arguments.
    ///
    /// # Arguments
    /// * `args` - `corpus_path`, `features_path`, and the keywords.
    ///
    /// # Returns
    /// `nil`; the features file is written.
    ///
    /// # Errors
    /// Raises `Litsea::IoError` if the corpus cannot be read or the output
    /// cannot be written.
    fn extract(&self, args: &[magnus::Value]) -> Result<(), Error> {
        let args = scan_args::<(String, String), (), (), (), _, ()>(args)?;
        let (corpus_path, features_path) = args.required;
        let kwargs = magnus::scan_args::get_kwargs::<_, (), (Option<bool>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["tsv", "tag_free"],
        )?;
        let (tsv, tag_free) = kwargs.optional;

        let extracted = without_gvl(|| {
            self.inner.extract(
                std::path::Path::new(&corpus_path),
                std::path::Path::new(&features_path),
                CorpusFormat::from_tsv_flag(tsv.unwrap_or(false)),
                tag_free.unwrap_or(false),
            )
        });
        map_err(extracted)
    }

    /// Extracts two-stage (segmentation + POS) features.
    ///
    /// Writes `{output_prefix}.stage1`, `.stage2`, and `.lexicon`. Accepts
    /// `feature_set:` and `tsv:` keyword arguments.
    ///
    /// # Arguments
    /// * `args` - `corpus_path`, `output_prefix`, and the keywords.
    ///
    /// # Returns
    /// `nil`; the three files are written.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` for an unknown feature set, or
    /// `Litsea::IoError` on I/O failure.
    fn extract_two_stage(&self, args: &[magnus::Value]) -> Result<(), Error> {
        let args = scan_args::<(String, String), (), (), (), _, ()>(args)?;
        let (corpus_path, output_prefix) = args.required;
        let kwargs = magnus::scan_args::get_kwargs::<_, (), (Option<String>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["feature_set", "tsv"],
        )?;
        let (feature_set, tsv) = kwargs.optional;

        let feature_set = map_err(parse_feature_set(feature_set.as_deref().unwrap_or("fast")))?;

        let extracted = without_gvl(|| {
            self.inner.extract_two_stage(
                std::path::Path::new(&corpus_path),
                std::path::Path::new(&output_prefix),
                feature_set,
                CorpusFormat::from_tsv_flag(tsv.unwrap_or(false)),
            )
        });
        map_err(extracted)
    }
}

/// Trains a segmentation model.
#[magnus::wrap(class = "Litsea::Trainer", free_immediately, size)]
pub struct Trainer {
    /// The wrapped trainer; `RefCell` because Ruby methods take `&self`.
    inner: RefCell<CoreTrainer>,
}

impl Trainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `threshold` - Early-stopping threshold for weak classifiers.
    /// * `num_iterations` - Maximum number of boosting iterations.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new trainer.
    ///
    /// # Errors
    /// Raises `Litsea::IoError` or `Litsea::ParseError` if the features file
    /// cannot be read.
    fn new(threshold: f64, num_iterations: usize, features_path: String) -> Result<Self, Error> {
        let inner = map_err(CoreTrainer::new(
            threshold,
            num_iterations,
            std::path::Path::new(&features_path),
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
    /// `nil`; the model is merged into the learner.
    ///
    /// # Errors
    /// Raises `Litsea::ModelError`, `Litsea::IoError`, or
    /// `Litsea::ParseError`.
    fn load_model(&self, model_uri: String) -> Result<(), Error> {
        let loaded = without_gvl(|| self.inner.borrow_mut().load_model_blocking(&model_uri));
        map_err(loaded)
    }

    /// Trains the model and writes it to `model_path`.
    ///
    /// The GVL is released while training runs, so another Ruby thread can
    /// cancel it through the `cancel:` token.
    ///
    /// # Arguments
    /// * `args` - `model_path` and an optional `cancel:` token.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Raises `Litsea::IoError` if the model cannot be written.
    fn train(&self, args: &[magnus::Value]) -> Result<RbBinaryMetrics, Error> {
        let (model_path, cancel) = scan_train_args(args)?;
        let trained = without_gvl(|| self.inner.borrow_mut().train(&cancel, &model_path));
        Ok(RbBinaryMetrics::from(map_err(trained)?))
    }
}

/// Trains a label-agnostic Averaged Perceptron model.
#[magnus::wrap(class = "Litsea::PerceptronTrainer", free_immediately, size)]
pub struct PerceptronTrainer {
    /// The wrapped trainer.
    inner: RefCell<CorePerceptronTrainer>,
}

impl PerceptronTrainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new trainer.
    ///
    /// # Errors
    /// Raises `Litsea::IoError` or `Litsea::ParseError` if the features file
    /// cannot be read.
    fn new(num_epochs: usize, features_path: String) -> Result<Self, Error> {
        let inner =
            map_err(CorePerceptronTrainer::new(num_epochs, std::path::Path::new(&features_path)))?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Trains the model and writes it to `model_path`.
    ///
    /// # Arguments
    /// * `args` - `model_path` and an optional `cancel:` token.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Raises `Litsea::IoError` if the model cannot be written.
    fn train(&self, args: &[magnus::Value]) -> Result<RbMulticlassMetrics, Error> {
        let (model_path, cancel) = scan_train_args(args)?;
        let trained = without_gvl(|| self.inner.borrow_mut().train(&cancel, &model_path));
        Ok(RbMulticlassMetrics::from(map_err(trained)?))
    }
}

/// Trains a two-stage segmentation + POS model.
///
/// A trainer can only be used once: training collapses stage 1 into an
/// AdaBoost model, which consumes it. `available?` reports the state, and a
/// second `train` raises.
#[magnus::wrap(class = "Litsea::TwoStageTrainer", free_immediately, size)]
pub struct TwoStageTrainer {
    /// The wrapped trainer.
    inner: RefCell<CoreTwoStageTrainer>,
}

impl TwoStageTrainer {
    /// Loads a two-stage features prefix and prepares training.
    ///
    /// Accepts a `dominance:` keyword argument.
    ///
    /// # Arguments
    /// * `args` - `num_epochs`, `features_prefix`, and the keyword.
    ///
    /// # Returns
    /// The new trainer.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` if `dominance` is out of range,
    /// or `Litsea::IoError` / `Litsea::ParseError` if the feature files
    /// cannot be read.
    fn new(args: &[magnus::Value]) -> Result<Self, Error> {
        let args = scan_args::<(usize, String), (), (), (), _, ()>(args)?;
        let (num_epochs, features_prefix) = args.required;
        let kwargs = magnus::scan_args::get_kwargs::<_, (), (Option<f64>,), ()>(
            args.keywords,
            &[],
            &["dominance"],
        )?;
        let (dominance,) = kwargs.optional;

        let inner = map_err(CoreTwoStageTrainer::new(
            num_epochs,
            dominance.unwrap_or(0.99),
            std::path::Path::new(&features_prefix),
        ))?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Returns whether this trainer can still be used.
    ///
    /// # Returns
    /// `false` once `train` has run.
    fn is_available(&self) -> bool {
        self.inner.borrow().is_available()
    }

    /// Trains both stages and writes the model.
    ///
    /// # Arguments
    /// * `args` - `model_path` and an optional `cancel:` token.
    ///
    /// # Returns
    /// The metrics of both stages.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` if the trainer has already been
    /// used, or `Litsea::IoError` if the model cannot be written.
    fn train(&self, args: &[magnus::Value]) -> Result<RbTwoStageMetrics, Error> {
        let (model_path, cancel) = scan_train_args(args)?;
        let trained = without_gvl(|| self.inner.borrow_mut().train(&cancel, &model_path));
        Ok(RbTwoStageMetrics::from(map_err(trained)?))
    }
}

/// Defines the training classes.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `module` - The `Litsea` module to define the classes on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a Ruby exception if a class cannot be defined.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let cancel = module.define_class("CancelToken", ruby.class_object())?;
    cancel.define_singleton_method("new", function!(CancelToken::new, 0))?;
    cancel.define_method("cancel", method!(CancelToken::cancel, 0))?;
    cancel.define_method("reset", method!(CancelToken::reset, 0))?;
    cancel.define_method("cancelled?", method!(CancelToken::is_cancelled, 0))?;

    let extractor = module.define_class("Extractor", ruby.class_object())?;
    extractor.define_singleton_method("new", function!(Extractor::new, 1))?;
    extractor.define_method("extract", method!(Extractor::extract, -1))?;
    extractor.define_method("extract_two_stage", method!(Extractor::extract_two_stage, -1))?;

    let trainer = module.define_class("Trainer", ruby.class_object())?;
    trainer.define_singleton_method("new", function!(Trainer::new, 3))?;
    trainer.define_method("load_model", method!(Trainer::load_model, 1))?;
    trainer.define_method("train", method!(Trainer::train, -1))?;

    let perceptron = module.define_class("PerceptronTrainer", ruby.class_object())?;
    perceptron.define_singleton_method("new", function!(PerceptronTrainer::new, 2))?;
    perceptron.define_method("train", method!(PerceptronTrainer::train, -1))?;

    let two_stage = module.define_class("TwoStageTrainer", ruby.class_object())?;
    two_stage.define_singleton_method("new", function!(TwoStageTrainer::new, -1))?;
    two_stage.define_method("available?", method!(TwoStageTrainer::is_available, 0))?;
    two_stage.define_method("train", method!(TwoStageTrainer::train, -1))?;

    Ok(())
}
