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
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use rustc_hash::FxHashMap;

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::metrics::{BinaryMetrics, MulticlassMetrics};
use crate::perceptron::AveragedPerceptron;
use crate::two_stage::{TwoStageLearner, parse_lexicon, two_stage_paths};
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
    pub fn new(threshold: f64, num_iterations: usize, features_path: &Path) -> Result<Self> {
        let mut learner = AdaBoost::new(threshold, num_iterations);

        learner.initialize_features(features_path)?;
        learner.initialize_instances(features_path)?;

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
    pub fn train(&mut self, running: &AtomicBool, model_path: &Path) -> Result<BinaryMetrics> {
        self.learner.train(running);

        // Save the trained model to the specified file
        self.learner.save_model(model_path)?;

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
    pub fn new(num_epochs: usize, features_path: &Path) -> Result<Self> {
        Ok(PerceptronTrainer {
            learner: load_perceptron_instances(features_path)?,
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
    pub fn train(&mut self, running: &AtomicBool, model_path: &Path) -> Result<MulticlassMetrics> {
        self.learner.train(self.num_epochs, running);
        self.learner.save_model(model_path)?;
        Ok(self.learner.metrics())
    }
}

/// Loads training instances from a features file (`label\tfeature\t...`
/// rows) into a fresh [`AveragedPerceptron`]. Shared by
/// [`PerceptronTrainer::new`] and [`TwoStageTrainer::new`], which read the
/// same row format for different label spaces (boundary `B`/`O` vs. UPOS
/// tags — the perceptron treats labels as opaque strings either way).
fn load_perceptron_instances(features_path: &Path) -> Result<AveragedPerceptron> {
    let mut learner = AveragedPerceptron::new();

    let file = File::open(features_path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        // lines() already strips the trailing newline/CRLF; trimming
        // further would strip Unicode whitespace (e.g. a trailing U+3000)
        // off the last feature and desync training from inference (#99).
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let label = parts
            .next()
            .ok_or_else(|| LitseaError::InvalidData("Missing label in feature line".to_string()))?;
        let features: HashSet<String> =
            parts.filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
        if features.is_empty() {
            continue;
        }
        learner.add_instance(features, label.to_string());
    }

    Ok(learner)
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
    pub fn new(num_epochs: usize, dominance: f64, features_prefix: &Path) -> Result<Self> {
        // Checked here (not just in from_parts at train() time) so an
        // out-of-range value fails before training runs, not after.
        if !(dominance > 0.5 && dominance <= 1.0) {
            return Err(LitseaError::InvalidInput(format!(
                "dominance must be in (0.5, 1.0], got {}",
                dominance
            )));
        }
        let (stage1_path, stage2_path, lexicon_path) = two_stage_paths(features_prefix);

        let stage1 = load_perceptron_instances(&stage1_path)?;
        if stage1.class_names().iter().any(|c| c != "B" && c != "O") {
            return Err(LitseaError::InvalidData(format!(
                "stage-1 features file has a non-boundary label; expected only 'B'/'O', found {:?}",
                stage1.class_names()
            )));
        }
        let stage2 = load_perceptron_instances(&stage2_path)?;

        let lexicon_file = File::open(&lexicon_path)?;
        let lines: io::Result<Vec<String>> = io::BufReader::new(lexicon_file).lines().collect();
        let lexicon = parse_lexicon(&lines?)?;

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
    pub fn train(mut self, running: &AtomicBool, model_path: &Path) -> Result<TwoStageMetrics> {
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
        learner.save_model(model_path)?;

        Ok(TwoStageMetrics {
            stage1: stage1_metrics,
            stage2: stage2_metrics,
        })
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
}
