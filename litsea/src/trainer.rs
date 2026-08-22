//! High-level training front-ends.
//!
//! Defines [`Trainer`] (AdaBoost word-boundary model), [`PerceptronTrainer`]
//! (a generic label-agnostic Averaged Perceptron trainer, used as the
//! training step of the boundary-collapse recipe below), and
//! [`TwoStageTrainer`] (the two-stage boundary + word-tagger model of issue
//! #147). `Trainer` and `PerceptronTrainer` each read a single features file
//! produced by [`Extractor`](crate::extractor::Extractor); `TwoStageTrainer`
//! reads the three files
//! [`Extractor::extract_two_stage`](crate::extractor::Extractor::extract_two_stage)
//! writes from a common prefix. All three optionally load an existing
//! model, train, save the result, and report training metrics.
//!
//! # The lossless boundary-perceptron-to-AdaBoost collapse
//!
//! `TwoStageTrainer` trains its stage-1 boundary classifier as a 2-class
//! (`B`/`O`) [`AveragedPerceptron`] but saves it in the [`AdaBoost`] text
//! format, via the private `collapse_boundary_perceptron` helper in this
//! module. This is also how the bundled
//! `models/{japanese,chinese,korean}.model` segmentation models are produced
//! (issue #165): trained as a 2-class Averaged Perceptron, then losslessly
//! collapsed to scalar AdaBoost-format weights, rather than trained by
//! AdaBoost boosting.
//!
//! The collapse is exact because a 2-class perceptron has no bias term, so
//! for any feature set `f` its class scores are `score_B = sum_{feat in f}
//! w_B[feat]` and `score_O = sum_{feat in f} w_O[feat]`, and therefore
//! `score_B - score_O = sum_{feat in f} (w_B[feat] - w_O[feat])` exactly.
//! Writing one line `feat\t(w_B[feat] - w_O[feat])` per feature plus a
//! literal `"0"` bias line and loading the result with
//! [`AdaBoost::load_model_from_reader`] reproduces this: the AdaBoost model
//! format defines `bias()` to equal the written bias line verbatim
//! (independent of the feature weights), so the collapsed model's decision
//! rule `score >= 0.0` becomes exactly `score_B >= score_O`. This includes
//! the tie case (an empty or entirely-unseen feature set): the perceptron's
//! first-wins argmax picks class index 0, which is `"B"` because classes are
//! always ordered `["B", "O"]` (`B` sorts first alphabetically), the same
//! class AdaBoost's `0.0 >= 0.0` picks.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::sync::atomic::AtomicBool;

// The path-based entry points are compiled out on wasm32, which has no
// filesystem; the in-memory ones below work everywhere.
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::metrics::{BinaryMetrics, MulticlassMetrics};
use crate::perceptron::AveragedPerceptron;
use crate::two_stage::{TwoStageLearner, parse_lexicon};
// Only `TwoStageTrainer::new` needs the path helper.
#[cfg(not(target_arch = "wasm32"))]
use crate::two_stage::two_stage_paths;
use crate::upos::Upos;

/// Trainer struct for managing the AdaBoost training process.
/// It initializes the AdaBoost learner with the specified parameters (from a
/// features file) and provides methods to optionally load an existing model
/// (see [`load_model`](Self::load_model)), train, and save the trained
/// model.
#[derive(Debug)]
pub struct Trainer {
    /// The underlying AdaBoost learner.
    learner: AdaBoost,
}

/// Generic Averaged Perceptron trainer.
/// Manages multiclass classification training with the Averaged Perceptron;
/// labels are treated as opaque strings, so it works for any label space.
/// Its main use is training the 2-class (`B`/`O`) boundary perceptron that
/// the collapse recipe (see the module docs) turns into the bundled
/// AdaBoost-format segmentation models.
#[derive(Debug)]
pub struct PerceptronTrainer {
    /// The underlying Averaged Perceptron learner.
    learner: AveragedPerceptron,
    /// The number of training epochs to run.
    num_epochs: usize,
}

impl Trainer {
    /// Creates a new instance of [`Trainer`].
    ///
    /// # Arguments
    /// * `threshold` - The threshold for the AdaBoost algorithm.
    /// * `num_iterations` - The number of iterations for the training.
    /// * `features_path` - The path to the features file.
    ///
    /// # Returns
    /// Returns a new instance of `Trainer`.
    ///
    /// # Errors
    /// Returns an error if the features or instances cannot be initialized.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(threshold: f64, num_iterations: usize, features_path: &Path) -> Result<Self> {
        let mut learner = AdaBoost::new(threshold, num_iterations);

        learner.initialize_features(features_path)?;
        learner.initialize_instances(features_path)?;

        Ok(Trainer { learner })
    }

    /// Creates a `Trainer` from a features file's contents.
    ///
    /// The in-memory counterpart of [`new`](Self::new), for callers with no
    /// filesystem (WebAssembly) or with the features already in memory.
    ///
    /// Takes a `&str` rather than a reader because the content is scanned
    /// twice: once to build the feature vocabulary, once to build the
    /// instances against it.
    ///
    /// # Arguments
    /// * `threshold` - The threshold for the weak classifier's accuracy.
    /// * `num_iterations` - The maximum number of boosting iterations.
    /// * `features` - The contents of a features file.
    ///
    /// # Returns
    /// Returns a new instance of `Trainer`.
    ///
    /// # Errors
    /// Returns an error if the features or instances cannot be initialized.
    pub fn from_features(threshold: f64, num_iterations: usize, features: &str) -> Result<Self> {
        let mut learner = AdaBoost::new(threshold, num_iterations);

        learner.initialize_features_from_str(features)?;
        learner.initialize_instances_from_str(features)?;

        Ok(Trainer { learner })
    }

    /// Load Model from a URI.
    ///
    /// # Arguments
    /// * `model_uri` - The URI of the model to load (file path or http/https URL).
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub async fn load_model(&mut self, model_uri: &str) -> Result<()> {
        self.learner.load_model(model_uri).await
    }

    /// Loads an existing model from a reader, to continue training from it.
    ///
    /// The in-memory counterpart of [`load_model`](Self::load_model), and
    /// synchronous: nothing needs fetching.
    ///
    /// # Arguments
    /// * `reader` - A buffered reader over the model's contents.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the model content is malformed.
    pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        self.learner.load_model_from_reader(reader)
    }

    /// Train the AdaBoost model.
    ///
    /// # Arguments
    /// * `running` - An `AtomicBool` flag to control the running state of the training process.
    /// * `model_path` - The path to save the trained model.
    ///
    /// # Returns
    /// Returns the [`BinaryMetrics`] of the trained model measured on the
    /// training data: accuracy, precision, recall, and the confusion-matrix
    /// counts.
    ///
    /// # Errors
    /// Returns an error if the training fails or if the model cannot be saved.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn train(&mut self, running: &AtomicBool, model_path: &Path) -> Result<BinaryMetrics> {
        self.learner.train(running);

        // Save the trained model to the specified file
        self.learner.save_model(model_path)?;

        Ok(self.learner.metrics())
    }

    /// Trains the model and writes it to a writer.
    ///
    /// The in-memory counterpart of [`train`](Self::train). Cancelling
    /// `running` behaves the same way: training stops at its next check
    /// point and the partially trained model is still written.
    ///
    /// # Arguments
    /// * `running` - An `AtomicBool` flag to control the running state of the
    ///   training process.
    /// * `writer` - Where to write the trained model.
    ///
    /// # Returns
    /// Returns the [`BinaryMetrics`] of the trained model measured on the
    /// training data.
    ///
    /// # Errors
    /// Returns an error if the model cannot be written.
    pub fn train_to_writer<W: Write>(
        &mut self,
        running: &AtomicBool,
        writer: &mut W,
    ) -> Result<BinaryMetrics> {
        self.learner.train(running);
        self.learner.save_model_to_writer(writer)?;

        Ok(self.learner.metrics())
    }
}

impl PerceptronTrainer {
    /// Creates a PerceptronTrainer from a features file.
    ///
    /// Features file format: each line is "label\tfeature1\tfeature2\t...".
    /// Labels are opaque strings (e.g. the boundary labels "B"/"O" of the
    /// collapse recipe).
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs
    /// * `features_path` - The path to the features file
    ///
    /// # Returns
    /// Returns a new instance of `PerceptronTrainer` with the training
    /// instances loaded from the features file.
    ///
    /// # Errors
    /// Returns an error if the features file cannot be opened or read, or
    /// if a feature line is missing its label.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(num_epochs: usize, features_path: &Path) -> Result<Self> {
        Ok(PerceptronTrainer {
            learner: load_perceptron_instances(features_path)?,
            num_epochs,
        })
    }

    /// Creates a `PerceptronTrainer` from a features file's contents.
    ///
    /// The in-memory counterpart of [`new`](Self::new).
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs.
    /// * `features` - The contents of a features file.
    ///
    /// # Returns
    /// Returns a new instance of `PerceptronTrainer`.
    ///
    /// # Errors
    /// Returns an error if a feature line is missing its label.
    pub fn from_features(num_epochs: usize, features: &str) -> Result<Self> {
        Ok(PerceptronTrainer {
            learner: parse_perceptron_instances(features)?,
            num_epochs,
        })
    }

    /// Loads an existing model from a URI.
    ///
    /// # Arguments
    /// * `model_uri` - The URI of the model to load (file path or http/https URL).
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub async fn load_model(&mut self, model_uri: &str) -> Result<()> {
        self.learner.load_model(model_uri).await
    }

    /// Loads an existing model from a reader.
    ///
    /// The in-memory counterpart of [`load_model`](Self::load_model).
    ///
    /// # Arguments
    /// * `reader` - A buffered reader over the model's contents.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the model content is malformed.
    pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        self.learner.load_model_from_reader(reader)
    }

    /// Trains the model and saves it.
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting the training
    /// * `model_path` - The path to save the model to
    ///
    /// # Returns
    /// Returns the [`MulticlassMetrics`] of the trained model measured on
    /// the training data: accuracy plus macro-averaged precision and recall.
    ///
    /// # Errors
    /// Returns an error if the trained model cannot be saved.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn train(&mut self, running: &AtomicBool, model_path: &Path) -> Result<MulticlassMetrics> {
        self.learner.train(self.num_epochs, running);
        self.learner.save_model(model_path)?;
        Ok(self.learner.metrics())
    }

    /// Trains the model and writes it to a writer.
    ///
    /// The in-memory counterpart of [`train`](Self::train).
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting the training.
    /// * `writer` - Where to write the trained model.
    ///
    /// # Returns
    /// Returns the [`MulticlassMetrics`] of the trained model measured on
    /// the training data.
    ///
    /// # Errors
    /// Returns an error if the model cannot be written.
    pub fn train_to_writer<W: Write>(
        &mut self,
        running: &AtomicBool,
        writer: &mut W,
    ) -> Result<MulticlassMetrics> {
        self.learner.train(self.num_epochs, running);
        self.learner.save_model_to_writer(writer)?;
        Ok(self.learner.metrics())
    }
}

/// Loads training instances from a features file (`label\tfeature\t...`
/// rows) into a fresh [`AveragedPerceptron`]. Shared by
/// [`PerceptronTrainer::new`] and [`TwoStageTrainer::new`], which read the
/// same row format for different label spaces (boundary `B`/`O` vs. UPOS
/// tags — the perceptron treats labels as opaque strings either way).
#[cfg(not(target_arch = "wasm32"))]
fn load_perceptron_instances(features_path: &Path) -> Result<AveragedPerceptron> {
    let mut learner = AveragedPerceptron::new();

    let file = File::open(features_path)?;
    let reader = io::BufReader::new(file);

    // Streamed rather than slurped: a features file for a real corpus can be
    // hundreds of megabytes.
    for line in reader.lines() {
        ingest_perceptron_line(&mut learner, &line?)?;
    }

    Ok(learner)
}

/// Parses training instances from a features file's contents into a fresh
/// [`AveragedPerceptron`].
///
/// The in-memory counterpart of `load_perceptron_instances`, shared by
/// [`PerceptronTrainer::from_features`] and
/// [`TwoStageTrainer::from_features`].
///
/// # Arguments
/// * `features` - The contents of a features file.
///
/// # Returns
/// The perceptron, loaded with the instances.
///
/// # Errors
/// Returns [`LitseaError::InvalidData`] if a line is missing its label.
fn parse_perceptron_instances(features: &str) -> Result<AveragedPerceptron> {
    let mut learner = AveragedPerceptron::new();

    for line in features.lines() {
        ingest_perceptron_line(&mut learner, line)?;
    }

    Ok(learner)
}

/// Adds one `label\tfeature...` line to a perceptron's instance set.
///
/// # Arguments
/// * `learner` - The perceptron to add the instance to.
/// * `line` - The line to ingest; blank and feature-less lines are skipped.
///
/// # Returns
/// A result indicating success or failure.
///
/// # Errors
/// Returns [`LitseaError::InvalidData`] if the line is missing its label.
fn ingest_perceptron_line(learner: &mut AveragedPerceptron, line: &str) -> Result<()> {
    // The caller's line iterator already strips the trailing newline/CRLF;
    // trimming further would strip Unicode whitespace (e.g. a trailing
    // U+3000) off the last feature and desync training from inference (#99).
    if line.is_empty() {
        return Ok(());
    }
    let mut parts = line.split('\t');
    let label = parts
        .next()
        .ok_or_else(|| LitseaError::InvalidData("Missing label in feature line".to_string()))?;
    let features: HashSet<String> =
        parts.filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
    if features.is_empty() {
        return Ok(());
    }
    learner.add_instance(features, label.to_string());

    Ok(())
}

/// Collapses a 2-class boundary Averaged Perceptron (classes `B`/`O`,
/// produced by [`TwoStageTrainer`]) into scalar per-feature weights in the
/// existing AdaBoost model format.
///
/// The perceptron scores a position purely as `sum(matched-feature
/// weights)` per class (there is no perceptron-level bias term), so
/// `score_B - score_O = sum(matched (w_B[f] - w_O[f]))` exactly. Writing
/// `feat\t(w_B[f] - w_O[f])` lines plus a literal `0` bias line and parsing
/// them with [`AdaBoost::load_model_from_reader`] reproduces this: the
/// model format defines `bias()` to equal the written bias line verbatim
/// regardless of the feature weights (the algebraic inverse of
/// `save_model`'s bias computation), so the collapsed model's decision
/// `score >= 0.0` becomes exactly `score_B >= score_O` — including the tie
/// case, which both the perceptron's first-wins rule (`"B" < "O"`
/// alphabetically, so `B` is class index 0) and this comparison resolve to
/// `B`.
fn collapse_boundary_perceptron(stage1: &AveragedPerceptron) -> Result<AdaBoost> {
    let classes = stage1.class_names();
    let b = classes.iter().position(|c| c == "B").ok_or_else(|| {
        LitseaError::InvalidData("stage-1 boundary model has no 'B' class".to_string())
    })?;
    let o = classes.iter().position(|c| c == "O").ok_or_else(|| {
        LitseaError::InvalidData("stage-1 boundary model has no 'O' class".to_string())
    })?;

    let mut text = String::new();
    for (feat, weights) in stage1.feature_class_weights() {
        let w = weights[b] - weights[o];
        if w != 0.0 {
            text.push_str(feat);
            text.push('\t');
            text.push_str(&w.to_string());
            text.push('\n');
        }
    }
    text.push_str("0\n");

    let mut adaboost = AdaBoost::default();
    adaboost.load_model_from_reader(text.as_bytes())?;
    Ok(adaboost)
}

/// In-sample training metrics of a [`TwoStageTrainer::train`] run: one
/// [`MulticlassMetrics`] per stage (stage 1 measured over its 2 boundary
/// classes, stage 2 over the UPOS tags).
#[derive(Debug, Clone)]
pub struct TwoStageMetrics {
    /// Metrics of the stage-1 boundary classifier.
    pub stage1: MulticlassMetrics,
    /// Metrics of the stage-2 word-level tagger.
    pub stage2: MulticlassMetrics,
}

/// Trainer for the two-stage model (issue #147): a binary boundary
/// classifier (stage 1) plus a word-level multiclass tagger (stage 2),
/// assembled with a candidate-tag lexicon into a `litsea-two-stage v1`
/// model. Reads the three files written by
/// [`Extractor::extract_two_stage`](crate::extractor::Extractor::extract_two_stage).
#[derive(Debug)]
pub struct TwoStageTrainer {
    /// The stage-1 boundary classifier, trained as a 2-class (`B`/`O`)
    /// Averaged Perceptron and collapsed to AdaBoost format on save.
    stage1: AveragedPerceptron,
    /// The stage-2 word-level multiclass tagger (UPOS classes).
    stage2: AveragedPerceptron,
    /// Candidate-tag lexicon: surface -> observed `(tag, count)` pairs.
    lexicon: FxHashMap<String, Vec<(Upos, u32)>>,
    /// The number of training epochs for both stages.
    num_epochs: usize,
    /// The classifier-skip dominance threshold of the assembled model.
    dominance: f64,
}

impl TwoStageTrainer {
    /// Creates a `TwoStageTrainer` from the three files written by
    /// [`Extractor::extract_two_stage`](crate::extractor::Extractor::extract_two_stage).
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs for both stages.
    /// * `dominance` - The classifier-skip threshold of the assembled
    ///   model, in `(0.5, 1.0]`.
    /// * `features_prefix` - The same prefix passed to `extract_two_stage`;
    ///   the `{prefix}.stage1`, `{prefix}.stage2`, and `{prefix}.lexicon`
    ///   files are read from it.
    ///
    /// # Returns
    /// A new `TwoStageTrainer` with training instances loaded.
    ///
    /// # Errors
    /// Returns an error if `dominance` is out of range, if any of the
    /// three files cannot be opened or read, if the stage-1 features file
    /// has a label other than `B`/`O`, or if the lexicon file is
    /// malformed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(num_epochs: usize, dominance: f64, features_prefix: &Path) -> Result<Self> {
        let (stage1_path, stage2_path, lexicon_path) = two_stage_paths(features_prefix);

        let stage1 = load_perceptron_instances(&stage1_path)?;
        let stage2 = load_perceptron_instances(&stage2_path)?;

        let lexicon_file = File::open(&lexicon_path)?;
        let lines: io::Result<Vec<String>> = io::BufReader::new(lexicon_file).lines().collect();
        let lexicon = parse_lexicon(&lines?)?;

        Self::assemble(num_epochs, dominance, stage1, stage2, lexicon)
    }

    /// Creates a `TwoStageTrainer` from the three features files' contents.
    ///
    /// The in-memory counterpart of [`new`](Self::new), taking the three
    /// pieces [`Extractor::extract_two_stage_to_writers`](crate::extractor::Extractor::extract_two_stage_to_writers)
    /// produces instead of a path prefix.
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs.
    /// * `dominance` - The lexicon dominance threshold, in `(0.5, 1.0]`.
    /// * `stage1` - The contents of the stage-1 features (`B`/`O` labels).
    /// * `stage2` - The contents of the stage-2 features (UPOS labels).
    /// * `lexicon` - The contents of the lexicon
    ///   (`surface\tTAG:count[,TAG:count...]` lines).
    ///
    /// # Returns
    /// Returns a new instance of `TwoStageTrainer`.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if `dominance` is out of range,
    /// or [`LitseaError::InvalidData`] if the stage-1 features carry a label
    /// other than `B`/`O` or the lexicon is malformed.
    pub fn from_features(
        num_epochs: usize,
        dominance: f64,
        stage1: &str,
        stage2: &str,
        lexicon: &str,
    ) -> Result<Self> {
        let stage1 = parse_perceptron_instances(stage1)?;
        let stage2 = parse_perceptron_instances(stage2)?;
        let lexicon_lines: Vec<String> = lexicon.lines().map(|l| l.to_string()).collect();
        let lexicon = parse_lexicon(&lexicon_lines)?;

        Self::assemble(num_epochs, dominance, stage1, stage2, lexicon)
    }

    /// Validates the pieces and builds the trainer.
    ///
    /// Shared by [`new`](Self::new) and
    /// [`from_features`](Self::from_features) so both reject the same inputs.
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs.
    /// * `dominance` - The lexicon dominance threshold.
    /// * `stage1` - The stage-1 perceptron, loaded with instances.
    /// * `stage2` - The stage-2 perceptron, loaded with instances.
    /// * `lexicon` - The surface-to-tag-counts lexicon.
    ///
    /// # Returns
    /// Returns a new instance of `TwoStageTrainer`.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if `dominance` is out of range,
    /// or [`LitseaError::InvalidData`] if stage 1 carries a non-boundary
    /// label.
    fn assemble(
        num_epochs: usize,
        dominance: f64,
        stage1: AveragedPerceptron,
        stage2: AveragedPerceptron,
        lexicon: FxHashMap<String, Vec<(Upos, u32)>>,
    ) -> Result<Self> {
        // Checked here (not just in from_parts at train() time) so an
        // out-of-range value fails before training runs, not after.
        if !(dominance > 0.5 && dominance <= 1.0) {
            return Err(LitseaError::InvalidInput(format!(
                "dominance must be in (0.5, 1.0], got {}",
                dominance
            )));
        }
        if stage1.class_names().iter().any(|c| c != "B" && c != "O") {
            return Err(LitseaError::InvalidData(format!(
                "stage-1 features file has a non-boundary label; expected only 'B'/'O', found {:?}",
                stage1.class_names()
            )));
        }

        Ok(TwoStageTrainer {
            stage1,
            stage2,
            lexicon,
            num_epochs,
            dominance,
        })
    }

    /// Trains both stages and assembles + saves a `litsea-two-stage v1`
    /// model.
    ///
    /// Note: unlike [`Trainer::train`]/[`PerceptronTrainer::train`], this consumes
    /// `self` rather than taking `&mut self`, since the stage-1 perceptron is
    /// collapsed away into an [`AdaBoost`] and cannot be trained again in
    /// place afterward.
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting training.
    /// * `model_path` - The path to save the assembled model to.
    ///
    /// # Returns
    /// The in-sample [`TwoStageMetrics`] of both stages.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidData`] if the trained stage-1 model has
    /// no `B` class or no `O` class (from the internal boundary-perceptron
    /// collapse step, which runs first), or if collapsing it into an
    /// [`AdaBoost`] otherwise fails. Also returns an error if the assembled
    /// model is inconsistent (see [`TwoStageLearner::from_parts`]) or if it
    /// cannot be saved.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn train(self, running: &AtomicBool, model_path: &Path) -> Result<TwoStageMetrics> {
        let (learner, metrics) = self.train_into_learner(running)?;
        learner.save_model(model_path)?;

        Ok(metrics)
    }

    /// Trains both stages and writes the assembled model to a writer.
    ///
    /// The in-memory counterpart of [`train`](Self::train). Like it, this
    /// consumes `self`: stage 1 is collapsed away into an [`AdaBoost`] and
    /// cannot be trained again in place.
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting training.
    /// * `writer` - Where to write the assembled model.
    ///
    /// # Returns
    /// The in-sample [`TwoStageMetrics`] of both stages.
    ///
    /// # Errors
    /// Returns an error if the collapse fails, the assembled model is
    /// inconsistent, or the model cannot be written.
    pub fn train_to_writer<W: Write>(
        self,
        running: &AtomicBool,
        writer: &mut W,
    ) -> Result<TwoStageMetrics> {
        let (learner, metrics) = self.train_into_learner(running)?;
        learner.save_model_to_writer(writer)?;

        Ok(metrics)
    }

    /// Trains both stages and assembles the model, without saving it.
    ///
    /// Shared by [`train`](Self::train) and
    /// [`train_to_writer`](Self::train_to_writer).
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting training.
    ///
    /// # Returns
    /// The assembled learner and the in-sample metrics of both stages.
    ///
    /// # Errors
    /// Returns an error if the stage-1 collapse fails or the assembled model
    /// is inconsistent.
    fn train_into_learner(
        mut self,
        running: &AtomicBool,
    ) -> Result<(TwoStageLearner, TwoStageMetrics)> {
        self.stage1.train(self.num_epochs, running);
        self.stage2.train(self.num_epochs, running);
        let stage1_metrics = self.stage1.metrics();
        let stage2_metrics = self.stage2.metrics();

        let stage1_adaboost = collapse_boundary_perceptron(&self.stage1)?;
        let learner = TwoStageLearner::from_parts(
            stage1_adaboost,
            self.stage2,
            self.lexicon,
            self.dominance,
        )?;

        Ok((
            learner,
            TwoStageMetrics {
                stage1: stage1_metrics,
                stage2: stage2_metrics,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::atomic::AtomicBool;

    use tempfile::NamedTempFile;

    use crate::metrics::BinaryMetrics;

    // Helper: create a dummy features file.
    // This file should contain at least one line for initialize_features and initialize_instances.
    fn create_dummy_features_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file for features");

        // For example, it could contain "1\tfeature1" to represent one feature.
        writeln!(file, "1\tfeature1").expect("Failed to write to features file");
        file
    }

    // Helper: create a dummy model file.
    // This file should contain the model weights and bias.
    fn create_dummy_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file for model");

        // For example, it could contain a single feature weight and a bias term.
        // The feature line is "BW1:こん	-0.1262" and the last line is the bias term "100.0".
        writeln!(file, "BW1:こん\t-0.1262").expect("Failed to write feature");
        writeln!(file, "100.0").expect("Failed to write bias");
        file
    }

    #[tokio::test]
    async fn test_load_model() -> Result<()> {
        // Prepare a dummy features file
        let features_file = create_dummy_features_file();

        // Create a Trainer instance
        let mut trainer = Trainer::new(0.01, 10, features_file.path())?;

        // Prepare a dummy model file
        let model_file = create_dummy_model_file();

        // Load the model file into the Trainer
        // This should not return an error if the model file is correctly formatted.
        // If the model file is not correctly formatted, it will return an error.
        trainer.load_model(model_file.path().to_str().unwrap()).await?;

        Ok(())
    }

    #[test]
    fn test_new_empty_features_file() {
        // A features file with no actual features (only labels) should return an error
        // because initialize_features() requires at least one feature beyond the bias term.
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        // Write a line with only a label and no feature names.
        writeln!(file, "1").expect("Failed to write");
        let result = Trainer::new(0.01, 10, file.path());
        assert!(result.is_err(), "Trainer::new() should fail with an empty feature set");
    }

    #[test]
    fn test_train_immediate_stop() -> Result<()> {
        // Prepare a dummy features file
        let features_file = create_dummy_features_file();

        // Create a Trainer instance with the dummy features file
        let mut trainer = Trainer::new(0.01, 5, features_file.path())?;

        // Prepare a temporary file for the model output
        let model_out = NamedTempFile::new()?;

        // Set AtomicBool to false and immediately exit the learning loop
        let running = AtomicBool::new(false);

        // Execute the train method.
        let metrics: BinaryMetrics = trainer.train(&running, model_out.path())?;

        // Check if the metrics are valid.
        // Since metrics are dummy data, we will consider anything 0 or above to be OK here.
        assert!(metrics.accuracy >= 0.0);
        assert!(metrics.precision >= 0.0);
        assert!(metrics.recall >= 0.0);
        Ok(())
    }

    // --- PerceptronTrainer tests ---

    fn create_dummy_pos_features_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        // Multiclass features file (labels are opaque strings; these happen
        // to be SegmentLabel-style, which the perceptron does not interpret)
        writeln!(file, "B-NOUN\tUW4:猫\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:は\tUC4:I").expect("write");
        writeln!(file, "B-VERB\tUW4:食\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:べ\tUC4:I").expect("write");
        file
    }

    #[test]
    fn test_perceptron_trainer_new() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let trainer = PerceptronTrainer::new(5, features_file.path())?;
        assert_eq!(trainer.num_epochs, 5);
        Ok(())
    }

    #[test]
    fn test_perceptron_trainer_train() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PerceptronTrainer::new(5, features_file.path())?;

        let model_out = NamedTempFile::new()?;
        let running = AtomicBool::new(true);

        let metrics = trainer.train(&running, model_out.path())?;
        assert!(metrics.accuracy >= 0.0);
        assert_eq!(metrics.num_instances, 4);
        Ok(())
    }

    #[test]
    fn test_perceptron_trainer_train_immediate_stop() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PerceptronTrainer::new(5, features_file.path())?;

        let model_out = NamedTempFile::new()?;
        let running = AtomicBool::new(false);

        let metrics = trainer.train(&running, model_out.path())?;
        assert_eq!(metrics.num_instances, 4);
        Ok(())
    }

    #[test]
    fn test_debug_impls() -> Result<()> {
        // #129: trainer types are debuggable.
        let features_file = create_dummy_features_file();
        let trainer = Trainer::new(0.01, 10, features_file.path())?;
        assert!(!format!("{:?}", trainer).is_empty());
        let pos_features = create_dummy_pos_features_file();
        let perceptron_trainer = PerceptronTrainer::new(3, pos_features.path())?;
        assert!(!format!("{:?}", perceptron_trainer).is_empty());
        Ok(())
    }

    #[test]
    fn test_trainer_trains_to_completion() -> Result<()> {
        // Regression test for #106: the file-based pipeline
        // (initialize_features -> initialize_instances -> train -> save)
        // must actually learn when running stays true. The existing trainer
        // tests only stop training before the first iteration.
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        for _ in 0..3 {
            writeln!(file, "1\tfa").expect("write");
            writeln!(file, "-1\tfb").expect("write");
        }

        let mut trainer = Trainer::new(0.01, 100, file.path())?;
        let model_out = NamedTempFile::new()?;
        let running = AtomicBool::new(true);

        let metrics = trainer.train(&running, model_out.path())?;

        assert!(
            (metrics.accuracy - 100.0).abs() < 1e-9,
            "separable data must train to 100%, got {}",
            metrics.accuracy
        );
        assert_eq!(metrics.true_positives, 3);
        assert_eq!(metrics.true_negatives, 3);

        // The trained model was saved with real content (weight lines plus
        // the trailing bias line).
        let saved = std::fs::read_to_string(model_out.path())?;
        assert!(saved.lines().count() >= 2, "saved model looks empty: {saved:?}");
        Ok(())
    }

    #[test]
    fn test_perceptron_trainer_preserves_trailing_unicode_whitespace() -> Result<()> {
        // Regression test for #99: a feature that ends its line with an
        // ideographic space (U+3000) must not be trimmed away while reading
        // the features file, or its trained weight becomes unreachable.
        //
        // The U+3000 instance is labeled "O" on purpose: ties are resolved to
        // the alphabetically first class ("B-NOUN"), so this instance starts
        // out misclassified and its feature is guaranteed a non-zero weight
        // (save_model only writes non-zero weights).
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(file, "O\tUW4:\u{3000}").expect("write");
        writeln!(file, "B-NOUN\tUW4:x").expect("write");

        let mut trainer = PerceptronTrainer::new(5, file.path())?;
        let model_out = NamedTempFile::new()?;
        let running = AtomicBool::new(true);
        trainer.train(&running, model_out.path())?;

        let saved = std::fs::read_to_string(model_out.path())?;
        assert!(
            saved.contains("UW4:\u{3000}"),
            "trailing-U+3000 feature missing from saved model: {saved:?}"
        );
        Ok(())
    }

    #[test]
    fn test_collapse_boundary_perceptron_matches_perceptron_argmax() -> Result<()> {
        let mut learner = AveragedPerceptron::new();
        // Overlapping feature sets so weights end up non-trivial for both
        // classes, including a feature shared by both.
        learner.add_instance(
            ["f1".to_string(), "shared".to_string()].into_iter().collect(),
            "B".to_string(),
        );
        learner.add_instance(
            ["f2".to_string(), "shared".to_string()].into_iter().collect(),
            "O".to_string(),
        );
        learner.add_instance(["f3".to_string()].into_iter().collect(), "B".to_string());
        learner.add_instance(["f4".to_string()].into_iter().collect(), "O".to_string());
        let running = AtomicBool::new(true);
        learner.train(5, &running);

        let adaboost = collapse_boundary_perceptron(&learner)?;

        let check = |feats: &[&str]| {
            let set: HashSet<String> = feats.iter().map(|s| s.to_string()).collect();
            let perceptron_b = learner.predict(&set) == "B";
            let adaboost_b = adaboost.predict(&set) == 1;
            assert_eq!(
                perceptron_b, adaboost_b,
                "mismatch for {:?}: perceptron_b={} adaboost_b={}",
                feats, perceptron_b, adaboost_b
            );
        };

        check(&["f1"]);
        check(&["f2"]);
        check(&["f3"]);
        check(&["f4"]);
        check(&["f1", "f2"]);
        check(&["shared"]);
        check(&[]); // tie on unseen features: both must favor "B"
        check(&["unseen_feature"]);

        Ok(())
    }

    #[test]
    fn test_two_stage_trainer_end_to_end() -> Result<()> {
        use crate::extractor::Extractor;
        use crate::language::Language;
        use crate::segmenter::Segmenter;
        use crate::two_stage::{TwoStageFeatureSet, TwoStageLearner};

        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "太郎/PROPN は/ADP 猫/NOUN が/ADP 好き/ADJ です/AUX 。/PUNCT")?;
        writeln!(corpus_file, "花子/PROPN も/ADP 猫/NOUN が/ADP 好き/ADJ です/AUX 。/PUNCT")?;
        writeln!(corpus_file, "太郎/PROPN は/ADP 犬/NOUN も/ADP 好き/ADJ です/AUX 。/PUNCT")?;
        corpus_file.as_file().sync_all()?;

        let dir = tempfile::tempdir()?;
        let prefix = dir.path().join("features");
        Extractor::new(Language::Japanese).extract_two_stage(
            corpus_file.path(),
            &prefix,
            TwoStageFeatureSet::Fast,
        )?;

        let trainer = TwoStageTrainer::new(5, 0.99, &prefix)?;
        let model_path = dir.path().join("model.two-stage");
        let running = AtomicBool::new(true);
        let metrics = trainer.train(&running, &model_path)?;
        assert!(metrics.stage1.num_instances > 0);
        assert!(metrics.stage2.num_instances > 0);

        let mut learner = TwoStageLearner::new();
        learner.load_model_from_path(&model_path)?;
        let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);

        let tagged = segmenter.segment_with_pos("太郎は猫が好きです。")?;
        let text: String = tagged.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(text, "太郎は猫が好きです。");
        assert!(!tagged.is_empty());

        Ok(())
    }

    #[test]
    fn test_two_stage_trainer_end_to_end_tsv() -> Result<()> {
        use crate::extractor::Extractor;
        use crate::language::Language;
        use crate::segmenter::Segmenter;
        use crate::two_stage::{TwoStageFeatureSet, TwoStageLearner};

        // Space-preserving POS corpus (issue #198): the `" "` lexicon
        // surface must survive extraction, training, save, and load, and
        // must come back as a deterministic tag at inference rather than a
        // full-argmax guess.
        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "I/PRON\t \tdo/AUX\tn't/PART\t \tknow/VERB\t./PUNCT")?;
        writeln!(corpus_file, "I/PRON\t \tdo/AUX\t \tknow/VERB\t./PUNCT")?;
        writeln!(corpus_file, "we/PRON\t \tknow/VERB\t./PUNCT")?;
        corpus_file.as_file().sync_all()?;

        let dir = tempfile::tempdir()?;
        let prefix = dir.path().join("features");
        Extractor::new(Language::English).extract_two_stage_tsv(
            corpus_file.path(),
            &prefix,
            TwoStageFeatureSet::Fast,
        )?;

        let trainer = TwoStageTrainer::new(5, 0.99, &prefix)?;
        let model_path = dir.path().join("model.two-stage");
        let running = AtomicBool::new(true);
        let metrics = trainer.train(&running, &model_path)?;
        assert!(metrics.stage1.num_instances > 0);
        assert!(metrics.stage2.num_instances > 0);

        let mut learner = TwoStageLearner::new();
        learner.load_model_from_path(&model_path)?;
        // The space surface round-trips through the on-disk lexicon, as a
        // single candidate so the packed model's fixed-tag path applies.
        let space_entry = learner
            .lexicon_entry(" ")
            .expect("the lexicon must keep the space surface across save/load");
        assert_eq!(space_entry.len(), 1, "space should have exactly one candidate tag");
        assert_eq!(space_entry[0].0, crate::upos::Upos::X);

        let segmenter = Segmenter::with_two_stage_learner(Language::English, learner);
        let tagged = segmenter.segment_with_pos("I do n't know.")?;
        let text: String = tagged.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(text, "I do n't know.", "segment_with_pos must tile the input exactly");
        // Every whitespace token is tagged deterministically via the
        // single-candidate lexicon entry, not by the stage-2 classifier.
        for (word, tag) in tagged.iter().filter(|(w, _)| w.chars().all(char::is_whitespace)) {
            assert_eq!(*tag, crate::upos::Upos::X, "space {word:?} should be tagged X");
        }

        Ok(())
    }

    /// A small corpus, repeated so training has something to learn from.
    fn sample_corpus() -> String {
        let sentences = [
            "これ は テスト です 。",
            "隣 の 客 は よく 柿 食う 客 だ",
            "東京 都 から 神奈川 県 へ 引っ越し た",
        ];
        let mut corpus = String::new();
        for _ in 0..20 {
            for sentence in sentences {
                corpus.push_str(sentence);
                corpus.push('\n');
            }
        }
        corpus
    }

    /// The same corpus with UPOS tags, for the two-stage pipeline.
    fn sample_pos_corpus() -> String {
        let sentences = [
            "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT",
            "隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX",
            "東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX",
        ];
        let mut corpus = String::new();
        for _ in 0..20 {
            for sentence in sentences {
                corpus.push_str(sentence);
                corpus.push('\n');
            }
        }
        corpus
    }

    /// The central claim of the in-memory API: extracting and training
    /// through writers and strings produces **the same model, byte for
    /// byte**, as going through files. If this ever diverges, the two paths
    /// have stopped being the same pipeline.
    #[test]
    fn test_in_memory_matches_path_for_segmentation() -> Result<()> {
        use std::fs;

        use crate::extractor::Extractor;
        use crate::language::Language;

        let corpus = sample_corpus();
        let dir = tempfile::tempdir()?;
        let corpus_path = dir.path().join("corpus.txt");
        let features_path = dir.path().join("features.txt");
        let model_path = dir.path().join("model.txt");
        fs::write(&corpus_path, &corpus)?;

        let extractor = Extractor::new(Language::Japanese);

        // Path route.
        extractor.extract(&corpus_path, &features_path)?;
        let running = AtomicBool::new(true);
        Trainer::new(0.01, 20, &features_path)?.train(&running, &model_path)?;
        let path_features = fs::read_to_string(&features_path)?;
        let path_model = fs::read_to_string(&model_path)?;

        // In-memory route.
        let mut memory_features = Vec::new();
        extractor.extract_to_writer(&corpus, &mut memory_features)?;
        let memory_features = String::from_utf8(memory_features).expect("features are UTF-8");
        let mut memory_model = Vec::new();
        Trainer::from_features(0.01, 20, &memory_features)?
            .train_to_writer(&running, &mut memory_model)?;
        let memory_model = String::from_utf8(memory_model).expect("model is UTF-8");

        assert_eq!(memory_features, path_features, "extracted features diverged");
        assert_eq!(memory_model, path_model, "trained models diverged");

        Ok(())
    }

    /// The same claim for the two-stage pipeline, whose extractor writes
    /// three outputs and whose trainer collapses stage 1 before saving.
    #[test]
    fn test_in_memory_matches_path_for_two_stage() -> Result<()> {
        use std::fs;

        use crate::extractor::Extractor;
        use crate::language::Language;
        use crate::two_stage::TwoStageFeatureSet;

        let corpus = sample_pos_corpus();
        let dir = tempfile::tempdir()?;
        let corpus_path = dir.path().join("corpus_pos.txt");
        let prefix = dir.path().join("features");
        let model_path = dir.path().join("two_stage.model");
        fs::write(&corpus_path, &corpus)?;

        let extractor = Extractor::new(Language::Japanese);
        let feature_set = TwoStageFeatureSet::Fast;

        // Path route.
        extractor.extract_two_stage(&corpus_path, &prefix, feature_set)?;
        let running = AtomicBool::new(true);
        TwoStageTrainer::new(3, 0.99, &prefix)?.train(&running, &model_path)?;
        let path_stage1 = fs::read_to_string(dir.path().join("features.stage1"))?;
        let path_stage2 = fs::read_to_string(dir.path().join("features.stage2"))?;
        let path_lexicon = fs::read_to_string(dir.path().join("features.lexicon"))?;
        let path_model = fs::read_to_string(&model_path)?;

        // In-memory route.
        let (mut stage1, mut stage2, mut lexicon) = (Vec::new(), Vec::new(), Vec::new());
        extractor.extract_two_stage_to_writers(
            &corpus,
            &mut stage1,
            &mut stage2,
            &mut lexicon,
            feature_set,
        )?;
        let stage1 = String::from_utf8(stage1).expect("stage1 is UTF-8");
        let stage2 = String::from_utf8(stage2).expect("stage2 is UTF-8");
        let lexicon = String::from_utf8(lexicon).expect("lexicon is UTF-8");

        assert_eq!(stage1, path_stage1, "stage-1 features diverged");
        assert_eq!(stage2, path_stage2, "stage-2 features diverged");
        assert_eq!(lexicon, path_lexicon, "lexicon diverged");

        let mut memory_model = Vec::new();
        TwoStageTrainer::from_features(3, 0.99, &stage1, &stage2, &lexicon)?
            .train_to_writer(&running, &mut memory_model)?;
        let memory_model = String::from_utf8(memory_model).expect("model is UTF-8");

        assert_eq!(memory_model, path_model, "trained models diverged");

        Ok(())
    }

    /// The in-memory two-stage model must load back as a working segmenter.
    #[test]
    fn test_in_memory_two_stage_round_trip() -> Result<()> {
        use crate::extractor::Extractor;
        use crate::language::Language;
        use crate::segmenter::Segmenter;
        use crate::two_stage::{TwoStageFeatureSet, TwoStageLearner};

        let corpus = sample_pos_corpus();
        let extractor = Extractor::new(Language::Japanese);

        let (mut stage1, mut stage2, mut lexicon) = (Vec::new(), Vec::new(), Vec::new());
        extractor.extract_two_stage_to_writers(
            &corpus,
            &mut stage1,
            &mut stage2,
            &mut lexicon,
            TwoStageFeatureSet::Fast,
        )?;

        let mut model = Vec::new();
        let metrics = TwoStageTrainer::from_features(
            3,
            0.99,
            &String::from_utf8(stage1).expect("stage1 is UTF-8"),
            &String::from_utf8(stage2).expect("stage2 is UTF-8"),
            &String::from_utf8(lexicon).expect("lexicon is UTF-8"),
        )?
        .train_to_writer(&AtomicBool::new(true), &mut model)?;

        assert!(metrics.stage1.num_instances > 0);
        assert!(metrics.stage2.num_instances > 0);

        let mut learner = TwoStageLearner::new();
        learner.load_model_from_reader(model.as_slice())?;
        let segmenter = Segmenter::with_two_stage_learner(Language::Japanese, learner);
        let tokens = segmenter.segment_with_pos("これはテストです。")?;

        assert!(!tokens.is_empty());

        Ok(())
    }

    /// The perceptron trainer's two routes agree as well.
    #[test]
    fn test_in_memory_matches_path_for_perceptron() -> Result<()> {
        use std::fs;

        let features = "B\tf1\tf2\nO\tf2\tf3\nB\tf1\tf3\n";
        let dir = tempfile::tempdir()?;
        let features_path = dir.path().join("features.txt");
        let model_path = dir.path().join("model.txt");
        fs::write(&features_path, features)?;

        let running = AtomicBool::new(true);
        PerceptronTrainer::new(3, &features_path)?.train(&running, &model_path)?;
        let path_model = fs::read_to_string(&model_path)?;

        let mut memory_model = Vec::new();
        PerceptronTrainer::from_features(3, features)?
            .train_to_writer(&running, &mut memory_model)?;

        assert_eq!(String::from_utf8(memory_model).expect("model is UTF-8"), path_model);

        Ok(())
    }

    /// Training is a function of its input: the same features trained twice
    /// produce the same model. This did not hold before
    /// [`AveragedPerceptron::add_instance`] sorted its features - `HashSet`
    /// iteration order varies per set, and perceptron updates are
    /// order-sensitive, so two runs in one process disagreed.
    #[test]
    fn test_two_stage_training_is_reproducible() -> Result<()> {
        use crate::extractor::Extractor;
        use crate::language::Language;
        use crate::two_stage::TwoStageFeatureSet;

        let corpus = sample_pos_corpus();
        let (mut stage1, mut stage2, mut lexicon) = (Vec::new(), Vec::new(), Vec::new());
        Extractor::new(Language::Japanese).extract_two_stage_to_writers(
            &corpus,
            &mut stage1,
            &mut stage2,
            &mut lexicon,
            TwoStageFeatureSet::Fast,
        )?;
        let stage1 = String::from_utf8(stage1).expect("stage1 is UTF-8");
        let stage2 = String::from_utf8(stage2).expect("stage2 is UTF-8");
        let lexicon = String::from_utf8(lexicon).expect("lexicon is UTF-8");

        let running = AtomicBool::new(true);
        let mut models = Vec::new();
        for _ in 0..2 {
            let mut model = Vec::new();
            TwoStageTrainer::from_features(3, 0.99, &stage1, &stage2, &lexicon)?
                .train_to_writer(&running, &mut model)?;
            models.push(model);
        }

        assert_eq!(models[0], models[1], "two training runs disagreed");

        Ok(())
    }

    /// The same guarantee for the plain perceptron trainer, which the
    /// bundled segmentation models' collapse recipe goes through.
    #[test]
    fn test_perceptron_training_is_reproducible() -> Result<()> {
        let features = "B\tf1\tf2\tf3\nO\tf2\tf3\tf4\nB\tf1\tf4\tf5\nO\tf3\tf5\tf6\n";
        let running = AtomicBool::new(true);

        let mut models = Vec::new();
        for _ in 0..2 {
            let mut model = Vec::new();
            PerceptronTrainer::from_features(5, features)?.train_to_writer(&running, &mut model)?;
            models.push(model);
        }

        assert_eq!(models[0], models[1], "two training runs disagreed");

        Ok(())
    }
}
