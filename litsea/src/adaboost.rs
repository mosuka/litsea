use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{LitseaError, Result};
use crate::metrics::BinaryMetrics;

type Label = i8;

/// AdaBoost implementation for binary classification
/// This implementation uses a simple feature extraction method
/// and is designed for educational purposes.
/// It is not optimized for performance or large datasets.
#[derive(Debug)]
pub struct AdaBoost {
    pub threshold: f64,
    pub num_iterations: usize,
    instance_weights: Vec<f64>,
    model: Vec<f64>,
    features: Vec<String>,
    feature_index: HashMap<String, usize>,
    labels: Vec<Label>,
    instances_buf: Vec<usize>,
    instances: Vec<(usize, usize)>, // (start, end) index in instances_buf
    num_instances: usize,
}

impl AdaBoost {
    /// Creates a new instance of [`AdaBoost`].
    /// This method initializes the AdaBoost parameters such as threshold
    /// and number of iterations.
    ///
    /// # Arguments
    /// * `threshold`: The threshold for stopping the training.
    /// * `num_iterations`: The maximum number of iterations for training.
    ///
    /// # Returns: A new instance of [`AdaBoost`].
    pub fn new(threshold: f64, num_iterations: usize) -> Self {
        AdaBoost {
            threshold,
            num_iterations,
            instance_weights: vec![],
            model: vec![],
            features: vec![],
            feature_index: HashMap::new(),
            labels: vec![],
            instances_buf: vec![],
            instances: vec![],
            num_instances: 0,
        }
    }

    /// Initializes the features from a file.
    /// The file should contain lines with a label followed by space-separated features.
    ///
    /// # Arguments
    /// * `filename`: The path to the file containing the features.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the file cannot be opened or read.
    ///
    /// This method reads the file line by line, extracts features,
    /// and initializes the model with the features and their corresponding weights.
    /// It also counts the number of instances and reserves space in the vectors for efficient memory usage.
    ///
    /// # Note: The features are stored in a `BTreeMap` to preserve the order of insertion.
    /// The last feature is an empty string, which is used as a bias term.
    /// The model is initialized with zeros for each feature.
    /// The number of instances is counted to ensure that the model can handle the data efficiently.
    pub fn initialize_features(&mut self, filename: &Path) -> Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut map = BTreeMap::new(); // preserve order

        let mut buf_size = 0;
        self.num_instances = 0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            // Skip empty lines (no label token).
            let Some(_label) = parts.next() else {
                continue;
            };

            for h in parts {
                map.entry(h.to_string()).or_insert(0.0);
                buf_size += 1;
            }

            self.num_instances += 1;
        }

        // The bias term (empty string key) is always present.
        map.insert("".to_string(), 0.0);

        // A map with only the bias term means no actual features were extracted.
        if map.len() == 1 {
            return Err(LitseaError::InvalidData(
                "No features found in the training data (only bias term present)".to_string(),
            ));
        }

        self.features = map.keys().cloned().collect();
        self.model = map.values().cloned().collect();
        self.feature_index =
            self.features.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();

        self.instance_weights.reserve(self.num_instances);
        self.labels.reserve(self.num_instances);
        self.instances.reserve(self.num_instances);
        self.instances_buf.reserve(buf_size);

        Ok(())
    }

    /// Initializes the instances from a file.
    /// The file should contain lines with a label followed by space-separated features.
    ///
    /// Must be called after [`initialize_features`](Self::initialize_features) on the same file,
    /// because it depends on the feature index built by that method.
    ///
    /// # Arguments
    /// * `filename`: The path to the file containing the instances.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the file cannot be opened or read.
    ///
    /// This method reads the file line by line, extracts the label and features,
    /// and initializes the instances with their corresponding weights.
    /// It calculates the score for each instance based on the features and updates the model accordingly.
    /// The instance weights are initialized based on the label and score.
    pub fn initialize_instances(&mut self, filename: &Path) -> Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let bias = self.bias();

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let label: Label = parts
                .next()
                .ok_or_else(|| {
                    LitseaError::InvalidData("Missing label in instance line".to_string())
                })?
                .parse()
                .map_err(|e| LitseaError::InvalidData(format!("Invalid label: {}", e)))?;
            self.labels.push(label);

            let start = self.instances_buf.len();
            let mut score = bias;

            for h in parts {
                if let Some(&pos) = self.feature_index.get(h) {
                    self.instances_buf.push(pos);
                    score += self.model[pos];
                }
            }

            let end = self.instances_buf.len();
            // Sort feature indices so that binary_search in train() works correctly.
            self.instances_buf[start..end].sort_unstable();
            self.instances.push((start, end));
            self.instance_weights.push((-2.0 * label as f64 * score).exp());
        }

        Ok(())
    }

    /// Trains the AdaBoost model.
    /// This method iteratively updates the model based on the training data.
    ///
    /// # Arguments
    /// * `running`: An `Arc<AtomicBool>` to control the running state of the training process.
    ///
    /// # Returns: This method does not return a value.
    ///
    /// # Errors: This method does not return an error, but it will stop training if `running` is set to false.
    ///
    /// This method performs the following steps:
    /// 1. Initializes the error vector and sums of weights.
    /// 2. Iterates through the training data for a specified number of iterations.
    /// 3. For each instance, calculates the error based on the current model.
    /// 4. Finds the best hypothesis based on the error rates.
    /// 5. Updates the model with the best hypothesis and calculates the alpha value.
    /// 6. Updates the instance weights based on the predictions.
    /// 7. Normalizes the instance weights to ensure they sum to 1.
    pub fn train(&mut self, running: Arc<AtomicBool>) {
        let num_features = self.features.len();

        for _t in 0..self.num_iterations {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            let mut errors = vec![0.0f64; num_features];
            let mut instance_weight_sum = 0.0;
            let mut positive_weight_sum = 0.0;

            // Calculate errors and sum of weights
            for i in 0..self.num_instances {
                let d = self.instance_weights[i];
                let label = self.labels[i];
                instance_weight_sum += d;
                if label > 0 {
                    positive_weight_sum += d;
                }
                let delta = d * label as f64;
                let (start, end) = self.instances[i];
                for &h in &self.instances_buf[start..end] {
                    errors[h] -= delta;
                }
            }

            // Find the best hypothesis.
            // Initialize h_best to 0 (the bias bucket, i.e., the empty-string feature "").
            // The initial best_error_rate corresponds to a hypothetical weak learner that
            // predicts all instances as negative (label = -1), whose error rate equals the
            // fraction of positive instances.  Any real feature (index >= 1) must beat this
            // baseline to be selected.  If none does, h_best stays 0 and the bias bucket
            // is updated, which is equivalent to adding a constant "all-negative" weak learner.
            let mut h_best = 0;
            let mut best_error_rate = positive_weight_sum / instance_weight_sum;
            for (h, _) in errors.iter().enumerate().skip(1) {
                let mut e = errors[h] + positive_weight_sum;
                e /= instance_weight_sum;
                if (0.5 - e).abs() > (0.5 - best_error_rate).abs() {
                    h_best = h;
                    best_error_rate = e;
                }
            }

            if (0.5 - best_error_rate).abs() < self.threshold {
                break;
            }

            // Calculate alpha (weight for the weak learner)
            let alpha =
                0.5 * ((1.0 - best_error_rate).max(1e-10) / best_error_rate.max(1e-10)).ln();
            let alpha_exp = alpha.exp();
            self.model[h_best] += alpha;

            // Update model
            for i in 0..self.num_instances {
                let label = self.labels[i];
                let (start, end) = self.instances[i];
                let hs = &self.instances_buf[start..end];
                let prediction = if hs.binary_search(&h_best).is_ok() { 1 } else { -1 };
                if label * prediction < 0 {
                    self.instance_weights[i] *= alpha_exp;
                } else {
                    self.instance_weights[i] /= alpha_exp;
                }
            }

            // Normalize instance weights (guard against zero sum to prevent NaN).
            let sum_w: f64 = self.instance_weights.iter().sum();
            if sum_w > 0.0 {
                for d in &mut self.instance_weights {
                    *d /= sum_w;
                }
            }
        }
    }

    /// Saves the trained model to a file.
    /// The model is saved in a format where each line contains a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// # Arguments
    /// * `filename`: The path to the file where the model will be saved.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the file cannot be created or written to.
    ///
    /// This method writes the model to a file in a tab-separated format,
    /// where each line contains a feature and its corresponding weight.
    /// The last line contains the bias term, which is calculated as the negative sum of the model weights divided by 2.
    pub fn save_model(&self, filename: &Path) -> Result<()> {
        if self.model.is_empty() {
            return Err(LitseaError::InvalidInput("Cannot save an empty model".to_string()));
        }
        let mut file = File::create(filename)?;
        let mut bias = -self.model[0];
        for (h, &w) in self.features.iter().zip(self.model.iter()).skip(1) {
            if w != 0.0 {
                writeln!(file, "{}\t{}", h, w)?;
                bias -= w;
            }
        }
        writeln!(file, "{}", bias / 2.0)?;
        Ok(())
    }

    /// Loads a model from a URI.
    /// The URI can be a file path, a `file://` path, or an `http(s)://` URL
    /// (the latter requires the `remote_model` feature).
    /// The model should contain lines with a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// For local files, prefer the synchronous
    /// [`load_model_from_path`](Self::load_model_from_path).
    ///
    /// # Arguments
    /// * `uri`: The URI of the file containing the model.
    ///
    /// # Errors: Returns an error if the URI is invalid or the model cannot be read.
    pub async fn load_model(&mut self, uri: &str) -> Result<()> {
        let bytes = crate::model_io::read_model_bytes(uri).await?;
        self.load_model_from_reader(bytes.as_slice())
    }

    /// Loads a model from a local file path (synchronous).
    ///
    /// # Arguments
    /// * `path`: The path to the file containing the model.
    ///
    /// # Errors: Returns an error if the file cannot be read or parsed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_model_from_path(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        self.load_model_from_reader(BufReader::new(file))
    }

    /// Loads a model from a buffered reader (synchronous).
    ///
    /// If the learner already holds features (e.g. training data was loaded
    /// via [`initialize_features`](Self::initialize_features)), the loaded
    /// weights are merged into the existing feature index by feature name and
    /// unknown features are appended. This keeps previously built instance
    /// data valid for incremental training. Otherwise the model is loaded
    /// as-is.
    ///
    /// # Arguments
    /// * `reader`: A buffered reader containing the model data.
    ///
    /// # Errors: Returns an error if the content cannot be parsed.
    pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        let mut m: HashMap<String, f64> = HashMap::new();
        let mut bias = 0.0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let mut parts = line.split_whitespace();

            let h = parts.next().ok_or_else(|| {
                LitseaError::InvalidData(format!("Empty line at line {}", line_num + 1))
            })?;

            if let Some(v) = parts.next() {
                let value: f64 = v.parse().map_err(|e| {
                    LitseaError::InvalidData(format!(
                        "Invalid value at line {}: {}",
                        line_num + 1,
                        e
                    ))
                })?;
                m.insert(h.to_string(), value);
                bias += value;
            } else {
                let b: f64 = h.parse().map_err(|e| {
                    LitseaError::InvalidData(format!(
                        "Invalid bias at line {}: {}",
                        line_num + 1,
                        e
                    ))
                })?;
                m.insert("".to_string(), -b * 2.0 - bias);
            }
        }

        if self.features.is_empty() {
            // Fresh load: replace everything, features sorted by name.
            let sorted: BTreeMap<_, _> = m.into_iter().collect();
            self.features = sorted.keys().cloned().collect();
            self.model = sorted.values().cloned().collect();
            self.feature_index =
                self.features.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();
        } else {
            // Incremental load: merge weights by feature name so that indices
            // referenced by already-built instances stay valid; append
            // features that are not part of the training data.
            for (feature, weight) in m {
                if let Some(&idx) = self.feature_index.get(&feature) {
                    self.model[idx] = weight;
                } else {
                    let idx = self.features.len();
                    self.features.push(feature.clone());
                    self.model.push(weight);
                    self.feature_index.insert(feature, idx);
                }
            }
        }
        Ok(())
    }

    /// Adds a new instance to the model.
    /// The instance is represented by a set of attributes and a label.
    ///
    /// # Arguments
    /// * `attributes`: A `HashSet<String>` containing the attributes of the instance.
    /// * `label`: The label of the instance, represented as an `i8`.
    pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8) {
        let start = self.instances_buf.len();
        let attrs: Vec<String> = attributes.into_iter().collect();
        for attr in attrs.iter() {
            let idx = if let Some(&pos) = self.feature_index.get(attr) {
                pos
            } else {
                let pos = self.features.len();
                self.features.push(attr.clone());
                self.model.push(0.0);
                self.feature_index.insert(attr.clone(), pos);
                pos
            };
            self.instances_buf.push(idx);
        }
        let end = self.instances_buf.len();
        // Sort feature indices numerically so that binary_search in train() works correctly.
        self.instances_buf[start..end].sort_unstable();
        self.instances.push((start, end));
        self.labels.push(label);
        self.instance_weights.push(1.0);
        self.num_instances += 1;
    }

    /// Predicts the label for a given set of attributes.
    ///
    /// # Arguments
    /// * `attributes`: A `HashSet<String>` containing the attributes to predict.
    ///
    /// # Returns: The predicted label as an `i8`, where 1 indicates a positive prediction and -1 indicates a negative prediction.
    #[must_use]
    pub fn predict(&self, attributes: &HashSet<String>) -> i8 {
        let mut score = self.bias();
        for attr in attributes {
            if let Some(&idx) = self.feature_index.get(attr) {
                score += self.model[idx];
            }
        }
        if score >= 0.0 { 1 } else { -1 }
    }

    /// Gets the bias term of the model.
    /// The bias is calculated as the negative sum of the model weights divided by 2.
    ///
    /// # Returns: The bias term as a `f64`.
    #[must_use]
    pub fn bias(&self) -> f64 {
        -self.model.iter().sum::<f64>() / 2.0
    }

    /// Calculates and returns the performance metrics of the model on the training data.
    #[must_use]
    pub fn metrics(&self) -> BinaryMetrics {
        let bias = self.bias();
        let mut true_positives = 0; // true positives
        let mut false_positives = 0; // false positives
        let mut false_negatives = 0; // false negatives
        let mut true_negatives = 0; // true negatives

        for i in 0..self.num_instances {
            let label = self.labels[i];
            let (start, end) = self.instances[i];
            let mut score = bias;
            for &h in &self.instances_buf[start..end] {
                score += self.model[h];
            }
            if score >= 0.0 {
                if label > 0 {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            } else if label > 0 {
                false_negatives += 1;
            } else {
                true_negatives += 1;
            }
        }

        let accuracy =
            (true_positives + true_negatives) as f64 / self.num_instances.max(1) as f64 * 100.0;
        let precision =
            true_positives as f64 / (true_positives + false_positives).max(1) as f64 * 100.0;
        let recall =
            true_positives as f64 / (true_positives + false_negatives).max(1) as f64 * 100.0;

        BinaryMetrics {
            accuracy,
            precision,
            recall,
            num_instances: self.num_instances,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use tempfile::NamedTempFile;

    #[test]
    fn test_initialize_features() -> Result<()> {
        // Create a dummy features file
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        writeln!(features_file, "0 feat3")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 10);
        learner.initialize_features(features_file.path())?;

        // Features is an ordered set that should contain ""(empty string), "feat1", "feat2", "feat3"
        assert!(learner.features.contains(&"".to_string()));
        assert!(learner.features.contains(&"feat1".to_string()));
        assert!(learner.features.contains(&"feat2".to_string()));
        assert!(learner.features.contains(&"feat3".to_string()));
        Ok(())
    }

    #[test]
    fn test_initialize_instances() -> Result<()> {
        // First, initialize features in the feature file.
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 10);
        learner.initialize_features(features_file.path())?;

        // Create a dummy instance file
        let mut instance_file = NamedTempFile::new()?;
        // Example: "1 feat1" line. The learner will consider feat1 as a candidate if found by binary_search.
        writeln!(instance_file, "1 feat1")?;
        instance_file.as_file().sync_all()?;

        learner.initialize_instances(instance_file.path())?;

        // The number of instances should be 1, and the instance_weights, labels, and instances should be updated accordingly.
        assert_eq!(learner.num_instances, 1);
        assert_eq!(learner.labels.len(), 1);
        assert_eq!(learner.instance_weights.len(), 1);
        assert_eq!(learner.instances.len(), 1);

        Ok(())
    }

    #[test]
    fn test_train_immediate_stop() -> Result<()> {
        // Initialize features using a features file.
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 3);
        learner.initialize_features(features_file.path())?;

        // Create a dummy instance file with one instance.
        let mut instance_file = NamedTempFile::new()?;
        writeln!(instance_file, "1 feat1")?;
        instance_file.as_file().sync_all()?;
        learner.initialize_instances(instance_file.path())?;

        // Set running to false to immediately exit the learning loop.
        let running = Arc::new(AtomicBool::new(false));
        learner.train(running.clone());

        // If normalization of model or instance_weights is performed after learning, it should be OK.
        let weight_sum: f64 = learner.instance_weights.iter().sum();

        // weight_sum should be normalized to 1.0.
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // Model weights should remain at their initial state (all zeros) since
        // training was immediately stopped before any iteration could execute.
        assert!(
            learner.model.iter().all(|w| *w == 0.0),
            "Model weights should be all zeros after immediate stop"
        );

        Ok(())
    }

    #[test]
    fn test_save_and_load_model() -> Result<()> {
        // Prepare a dummy learner.
        let mut learner = AdaBoost::new(0.01, 10);

        // Set the features and weights in advance.
        learner.features = vec!["feat1".to_string(), "feat2".to_string()];
        learner.model = vec![0.5, -0.3];

        // Save the model to a temporary file.
        let temp_model = NamedTempFile::new()?;
        learner.save_model(temp_model.path())?;

        // Load the model with a new learner (synchronous path API).
        let mut learner2 = AdaBoost::new(0.01, 10);
        learner2.load_model_from_path(temp_model.path())?;

        // Check that the number of features and models match.
        assert_eq!(learner2.features.len(), learner.features.len());
        assert_eq!(learner2.model.len(), learner.model.len());

        Ok(())
    }

    #[tokio::test]
    async fn test_load_model_uri() -> Result<()> {
        let mut learner = AdaBoost::new(0.01, 10);
        learner.features = vec!["feat1".to_string()];
        learner.model = vec![0.5];

        let temp_model = NamedTempFile::new()?;
        learner.save_model(temp_model.path())?;

        // Load via the async URI API with a plain path.
        let mut learner2 = AdaBoost::new(0.01, 10);
        learner2.load_model(temp_model.path().to_str().unwrap()).await?;
        assert_eq!(learner2.features.len(), learner.features.len());
        Ok(())
    }

    #[test]
    fn test_load_model_merges_into_existing_features() -> Result<()> {
        // Regression test for the incremental-training path (bug B4):
        // loading a model after training data must keep existing feature
        // indices valid by merging weights by name.
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        writeln!(features_file, "-1 feat3")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 10);
        learner.initialize_features(features_file.path())?;
        learner.initialize_instances(features_file.path())?;

        let feat1_idx = learner.feature_index["feat1"];
        let feat3_idx = learner.feature_index["feat3"];

        // A model that knows feat1 (existing) and feat9 (new).
        let model_content = "feat1\t0.5\nfeat9\t-0.25\n0.0\n";
        learner.load_model_from_reader(model_content.as_bytes())?;

        // Existing indices are unchanged and weights are applied in place.
        assert_eq!(learner.feature_index["feat1"], feat1_idx);
        assert_eq!(learner.feature_index["feat3"], feat3_idx);
        assert!((learner.model[feat1_idx] - 0.5).abs() < 1e-9);
        // The unknown feature is appended, not substituted.
        assert!(learner.feature_index["feat9"] >= learner.features.len() - 2);
        assert_eq!(learner.features.len(), learner.model.len());
        Ok(())
    }

    #[test]
    fn test_add_instance_and_predict() {
        let mut learner = AdaBoost::new(0.01, 10);

        // Here, features and model are empty in the initial state. They are newly registered by add_instance.
        let mut attrs = HashSet::new();
        attrs.insert("A".to_string());
        learner.add_instance(attrs.clone(), 1);

        // When the same attribute is passed to predict, score returns 1 based on the initial model value (0.0) (because score>=0).
        let prediction = learner.predict(&attrs);
        assert_eq!(prediction, 1);
    }

    #[test]
    fn test_bias() {
        let mut learner = AdaBoost::new(0.01, 10);

        // Set model weights as an example.
        learner.model = vec![0.2, 0.3, -0.1];

        // bias = -sum(model)/2 = -(0.2+0.3-0.1)/2 = -0.4/2 = -0.2
        assert!((learner.bias() + 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_metrics() {
        let mut learner = AdaBoost::new(0.01, 10);

        // Set features and model for prediction
        learner.features = vec!["A".to_string(), "B".to_string()];
        learner.model = vec![0.5, -1.0];
        learner.feature_index =
            learner.features.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();

        // Instance 1: Attribute "A" → score = 0.25 + 0.5 = 0.75 (positive example)
        let mut attrs1 = HashSet::new();
        attrs1.insert("A".to_string());
        learner.add_instance(attrs1, 1);

        // Instance 2: Attribute "B" → score = 0.25 + (-1.0) = -0.75 (negative example)
        let mut attrs2 = HashSet::new();
        attrs2.insert("B".to_string());
        learner.add_instance(attrs2, -1);

        let metrics = learner.metrics();
        assert_eq!(metrics.true_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.num_instances, 2);

        // Since this is a simple case, the accuracy is 100%.
        assert!((metrics.accuracy - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_zero_instances() {
        // An empty AdaBoost with no instances should return zeroed metrics
        // without division-by-zero panics.
        let learner = AdaBoost::new(0.01, 10);
        let metrics = learner.metrics();
        assert_eq!(metrics.num_instances, 0);
        assert_eq!(metrics.true_positives, 0);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.true_negatives, 0);
        // .max(1) guard ensures 0/1 = 0.0, not NaN.
        assert!((metrics.accuracy - 0.0).abs() < f64::EPSILON);
        assert!((metrics.precision - 0.0).abs() < f64::EPSILON);
        assert!((metrics.recall - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_all_positive() {
        // All-positive instances: precision=100%, recall=100%, no false negatives.
        // Verifies the .max(1) guard handles zero denominators correctly.
        let mut learner = AdaBoost::new(0.01, 10);
        learner.features = vec!["".to_string(), "A".to_string()];
        learner.feature_index.insert("".to_string(), 0);
        learner.feature_index.insert("A".to_string(), 1);
        // model: weight for "" (bias bucket) = 0, weight for "A" = 1.0
        // bias = -(0.0 + 1.0) / 2.0 = -0.5
        // score for instance with "A": -0.5 + 1.0 = 0.5 >= 0 → positive prediction
        learner.model = vec![0.0, 1.0];

        let mut attrs = HashSet::new();
        attrs.insert("A".to_string());
        learner.add_instance(attrs.clone(), 1);
        learner.add_instance(attrs, 1);

        let metrics = learner.metrics();
        assert_eq!(metrics.num_instances, 2);
        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.true_negatives, 0);
        assert!((metrics.accuracy - 100.0).abs() < f64::EPSILON);
        assert!((metrics.precision - 100.0).abs() < f64::EPSILON);
        assert!((metrics.recall - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_model_from_reader_empty_input() {
        let mut learner = AdaBoost::new(0.01, 10);
        // Empty input should succeed with no features.
        let result = learner.load_model_from_reader("".as_bytes());
        assert!(result.is_ok());
        assert!(learner.features.is_empty());
    }

    #[test]
    fn test_load_model_from_reader_invalid_bias() {
        let mut learner = AdaBoost::new(0.01, 10);
        // A single non-numeric token (no tab separator) should fail as an invalid bias.
        let result = learner.load_model_from_reader("not_a_number".as_bytes());
        assert!(matches!(result, Err(LitseaError::InvalidData(_))));
    }

    #[test]
    fn test_load_model_from_reader_invalid_weight() {
        let mut learner = AdaBoost::new(0.01, 10);
        // A feature line with a non-numeric weight should fail.
        let result = learner.load_model_from_reader("feat1\tnot_a_number".as_bytes());
        assert!(matches!(result, Err(LitseaError::InvalidData(_))));
    }

    #[test]
    fn test_save_model_empty() {
        let learner = AdaBoost::new(0.01, 10);
        let temp = NamedTempFile::new().unwrap();
        let result = learner.save_model(temp.path());
        assert!(matches!(result, Err(LitseaError::InvalidInput(_))));
    }
}
