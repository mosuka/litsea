use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::adaboost::{AdaBoost, Metrics};
use crate::perceptron::{self, AveragedPerceptron};

/// Trainer struct for managing the AdaBoost training process.
/// It initializes the AdaBoost learner with the specified parameters,
/// loads the model from a file, and provides methods to train the model
/// and save the trained model.
pub struct Trainer {
    learner: AdaBoost,
}

/// 品詞推定モデル用のトレーナー。
/// Averaged Perceptronによる多クラス分類の学習を管理する。
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
    pub fn new(
        threshold: f64,
        num_iterations: usize,
        features_path: &Path,
    ) -> std::io::Result<Self> {
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
    pub async fn load_model(&mut self, model_uri: &str) -> std::io::Result<()> {
        self.learner.load_model(model_uri).await
    }

    /// Train the AdaBoost model.
    ///
    /// # Arguments
    /// * `running` - An `Arc<AtomicBool>` to control the running state of the training process.
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
        self.learner.train(running);

        // Save the trained model to the specified file
        self.learner.save_model(model_path)?;

        Ok(self.learner.get_metrics())
    }
}

impl PosTrainer {
    /// 品詞付き特徴量ファイルからPosTrainerを作成する。
    ///
    /// 特徴量ファイルのフォーマット: 各行が "ラベル\t特徴1\t特徴2\t..." 形式。
    /// ラベルは "B-NOUN", "O" 等のSegmentLabel文字列。
    ///
    /// # Arguments
    /// * `num_epochs` - 学習エポック数
    /// * `features_path` - 特徴量ファイルのパス
    pub fn new(num_epochs: usize, features_path: &Path) -> io::Result<Self> {
        let mut learner = AveragedPerceptron::new();

        let file = File::open(features_path)?;
        let reader = io::BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let label = parts.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Missing label in feature line")
            })?;
            let features: HashSet<String> = parts.map(|s| s.to_string()).collect();
            if features.is_empty() {
                continue;
            }
            learner.add_instance(features, label.to_string());
        }

        Ok(PosTrainer { learner, num_epochs })
    }

    /// 既存モデルをURIから読み込む。
    pub async fn load_model(&mut self, model_uri: &str) -> io::Result<()> {
        self.learner.load_model(model_uri).await
    }

    /// モデルを学習して保存する。
    ///
    /// # Arguments
    /// * `running` - 学習中断フラグ
    /// * `model_path` - モデルの保存先パス
    pub fn train(
        &mut self,
        running: Arc<AtomicBool>,
        model_path: &Path,
    ) -> Result<perceptron::Metrics, Box<dyn std::error::Error>> {
        self.learner.train(self.num_epochs, running);
        self.learner.save_model(model_path)?;
        Ok(self.learner.get_metrics())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

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
    fn test_train_immediate_stop() -> Result<(), Box<dyn std::error::Error>> {
        // Prepare a dummy features file
        let features_file = create_dummy_features_file();

        // Create a Trainer instance with the dummy features file
        let mut trainer = Trainer::new(0.01, 5, features_file.path())?;

        // Prepare a temporary file for the model output
        let model_out = NamedTempFile::new()?;

        // Set AtomicBool to false and immediately exit the learning loop
        let running = Arc::new(AtomicBool::new(false));

        // Execute the train method.
        let metrics: Metrics = trainer.train(running, model_out.path())?;

        // Check if the metrics are valid.
        // Since metrics are dummy data, we will consider anything 0 or above to be OK here.
        assert!(metrics.accuracy >= 0.0);
        assert!(metrics.precision >= 0.0);
        assert!(metrics.recall >= 0.0);
        Ok(())
    }

    // --- PosTrainer テスト ---

    fn create_dummy_pos_features_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        // 品詞付き特徴量ファイル（SegmentLabel形式のラベル）
        writeln!(file, "B-NOUN\tUW4:猫\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:は\tUC4:I").expect("write");
        writeln!(file, "B-VERB\tUW4:食\tUC4:H").expect("write");
        writeln!(file, "O\tUW4:べ\tUC4:I").expect("write");
        file
    }

    #[test]
    fn test_pos_trainer_new() -> Result<(), Box<dyn std::error::Error>> {
        let features_file = create_dummy_pos_features_file();
        let trainer = PosTrainer::new(5, features_file.path())?;
        assert_eq!(trainer.num_epochs, 5);
        Ok(())
    }

    #[test]
    fn test_pos_trainer_train() -> Result<(), Box<dyn std::error::Error>> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PosTrainer::new(5, features_file.path())?;

        let model_out = NamedTempFile::new()?;
        let running = Arc::new(AtomicBool::new(true));

        let metrics = trainer.train(running, model_out.path())?;
        assert!(metrics.accuracy >= 0.0);
        assert_eq!(metrics.num_instances, 4);
        Ok(())
    }

    #[test]
    fn test_pos_trainer_train_immediate_stop() -> Result<(), Box<dyn std::error::Error>> {
        let features_file = create_dummy_pos_features_file();
        let mut trainer = PosTrainer::new(5, features_file.path())?;

        let model_out = NamedTempFile::new()?;
        let running = Arc::new(AtomicBool::new(false));

        let metrics = trainer.train(running, model_out.path())?;
        assert_eq!(metrics.num_instances, 4);
        Ok(())
    }
}
