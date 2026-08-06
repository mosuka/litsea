use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::atomic::AtomicBool;

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::metrics::{BinaryMetrics, MulticlassMetrics};
use crate::perceptron::AveragedPerceptron;

/// Trainer struct for managing the AdaBoost training process.
/// It initializes the AdaBoost learner with the specified parameters,
/// loads the model from a file, and provides methods to train the model
/// and save the trained model.
#[derive(Debug)]
pub struct Trainer {
    learner: AdaBoost,
}

/// Trainer for the POS tagging model.
/// Manages multiclass classification training with the Averaged Perceptron.
#[derive(Debug)]
pub struct PosTrainer {
    learner: AveragedPerceptron,
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
    /// Returns a Result indicating success or failure.
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

impl PosTrainer {
    /// Creates a PosTrainer from a POS-tagged features file.
    ///
    /// Features file format: each line is "label\tfeature1\tfeature2\t...".
    /// Labels are SegmentLabel strings such as "B-NOUN" or "O".
    ///
    /// # Arguments
    /// * `num_epochs` - The number of training epochs
    /// * `features_path` - The path to the features file
    pub fn new(num_epochs: usize, features_path: &Path) -> Result<Self> {
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
            let label = parts.next().ok_or_else(|| {
                LitseaError::InvalidData("Missing label in feature line".to_string())
            })?;
            let features: HashSet<String> =
                parts.filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            if features.is_empty() {
                continue;
            }
            learner.add_instance(features, label.to_string());
        }

        Ok(PosTrainer {
            learner,
            num_epochs,
        })
    }

    /// Loads an existing model from a URI.
    pub async fn load_model(&mut self, model_uri: &str) -> Result<()> {
        self.learner.load_model(model_uri).await
    }

    /// Trains the model and saves it.
    ///
    /// # Arguments
    /// * `running` - A flag for interrupting the training
    /// * `model_path` - The path to save the model to
    pub fn train(&mut self, running: &AtomicBool, model_path: &Path) -> Result<MulticlassMetrics> {
        self.learner.train(self.num_epochs, running);
        self.learner.save_model(model_path)?;
        Ok(self.learner.metrics())
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

    // --- PosTrainer tests ---

    fn create_dummy_pos_features_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        // POS-tagged features file (SegmentLabel-style labels)
        writeln!(file, "B-NOUN\tUW4:猫\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:は\tUC4:I").expect("write");
        writeln!(file, "B-VERB\tUW4:食\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:べ\tUC4:I").expect("write");
        file
    }

    #[test]
    fn test_pos_trainer_new() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let trainer = PosTrainer::new(5, features_file.path())?;
        assert_eq!(trainer.num_epochs, 5);
        Ok(())
    }

    #[test]
    fn test_pos_trainer_train() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PosTrainer::new(5, features_file.path())?;

        let model_out = NamedTempFile::new()?;
        let running = AtomicBool::new(true);

        let metrics = trainer.train(&running, model_out.path())?;
        assert!(metrics.accuracy >= 0.0);
        assert_eq!(metrics.num_instances, 4);
        Ok(())
    }

    #[test]
    fn test_pos_trainer_train_immediate_stop() -> Result<()> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PosTrainer::new(5, features_file.path())?;

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
        let pos_trainer = PosTrainer::new(3, pos_features.path())?;
        assert!(!format!("{:?}", pos_trainer).is_empty());
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
    fn test_pos_trainer_preserves_trailing_unicode_whitespace() -> Result<()> {
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

        let mut trainer = PosTrainer::new(5, file.path())?;
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
}
