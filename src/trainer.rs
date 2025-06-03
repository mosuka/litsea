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
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Load the model from the specified file
        Ok(self.learner.load_model(model_path)?)
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
