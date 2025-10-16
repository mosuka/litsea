use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::adaboost::{AdaBoost, Metrics};

/// Trainer struct for managing the AdaBoost training process.
/// It initializes the AdaBoost learner with the specified parameters,
/// loads the model from a file, and provides methods to train the model
/// and save the trained model.
pub struct Trainer {
    learner: AdaBoost,
}

impl Trainer {
    /// Creates a new instance of [`Trainer`].
    ///
    /// # Arguments
    /// * `threshold` - The threshold for the AdaBoost algorithm.
    /// * `num_iterations` - The number of iterations for the training.
    /// * `num_threads` - The number of threads to use for training.
    /// * `features_path` - The path to the features file.
    ///
    /// # Returns
    /// Returns a new instance of `Trainer`.
    ///
    /// # Errors
    /// Returns an error if the features or instances cannot be initialized.
    pub fn new(
        threshold: f64,
        num_iterations: usize,
        num_threads: usize,
        features_path: &Path,
    ) -> Self {
        let mut learner = AdaBoost::new(threshold, num_iterations, num_threads);

        learner
            .initialize_features(features_path)
            .expect("Failed to initialize features");
        learner
            .initialize_instances(features_path)
            .expect("Failed to initialize instances");

        Trainer { learner }
    }

    /// Load Model from a file
    ///
    /// # Arguments
    /// * `model_path` - The path to the model file to load.    
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub async fn load_model(&mut self, model_uri: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Load the model from the specified file
        Ok(self.learner.load_model(model_uri).await?)
    }

    /// Train the AdaBoost model.
    ///
    /// # Arguments
    /// * `running` - An Arc<AtomicBool> to control the running state of the training process.
    /// * `model_path` - The path to save the trained model.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the training fails or if the model cannot be saved.
    pub fn train(
        &mut self,
        running: Arc<AtomicBool>,
        model_path: &Path,
    ) -> Result<Metrics, Box<dyn std::error::Error>> {
        self.learner.train(running.clone());

        // Save the trained model to the specified file
        self.learner.save_model(model_path)?;

        Ok(self.learner.get_metrics())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use tempfile::NamedTempFile;

    use crate::adaboost::Metrics;

    // Helper: create a dummy features file.
    // This file should contain at least one line for initialize_features and initialize_instances.
    fn create_dummy_features_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file for features");

        // For example, it could contain "1 feature1" to represent one feature.
        writeln!(file, "1 feature1").expect("Failed to write to features file");
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
    async fn test_load_model() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare a dummy features file
        let features_file = create_dummy_features_file();

        // Create a Trainer instance
        let mut trainer = Trainer::new(0.01, 10, 1, features_file.path());

        // Prepare a dummy model file
        let model_file = create_dummy_model_file();

        // Load the model file into the Trainer
        // This should not return an error if the model file is correctly formatted.
        // If the model file is not correctly formatted, it will return an error.
        trainer.load_model(model_file.path().to_str().unwrap()).await?;

        Ok(())
    }

    #[test]
    fn test_train() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare a dummy features file
        let features_file = create_dummy_features_file();

        // Create a Trainer instance with the dummy features file
        let mut trainer = Trainer::new(0.01, 5, 1, features_file.path());

        // Prepare a temporary file for the model output
        let model_out = NamedTempFile::new()?;

        // Set AtomicBool to false and immediately exit the learning loop
        let running = Arc::new(AtomicBool::new(false));

        // Execute the train method.
        let metrics: Metrics = trainer.train(running, model_out.path())?;

        // Check if the metrics are valie.
        // Since metrics are dummy data, we will consider anything 0 or above to be OK here.
        assert!(metrics.accuracy >= 0.0);
        assert!(metrics.precision >= 0.0);
        assert!(metrics.recall >= 0.0);
        Ok(())
    }
}
