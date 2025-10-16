use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use reqwest::Client;

use crate::util::ModelScheme;

type Label = i8;

/// Structure to hold evaluation metrics.
pub struct Metrics {
    /// Accuracy in percentage (%)
    pub accuracy: f64,
    /// Precision in percentage (%)
    pub precision: f64,
    /// Recall in percentage (%)
    pub recall: f64,
    /// Number of instances in the dataset
    pub num_instances: usize,
    /// True Positives count
    pub true_positives: usize,
    /// False Positives count
    pub false_positives: usize,
    /// False Negatives count
    pub false_negatives: usize,
    /// True Negatives count
    pub true_negatives: usize,
}

/// AdaBoost implementation for binary classification
/// This implementation uses a simple feature extraction method
/// and is designed for educational purposes.
/// It is not optimized for performance or large datasets.
#[derive(Debug)]
pub struct AdaBoost {
    pub threshold: f64,
    pub num_iterations: usize,
    pub num_threads: usize,
    instance_weights: Vec<f64>,
    model: Vec<f64>,
    features: Vec<String>,
    labels: Vec<Label>,
    instances_buf: Vec<usize>,
    instances: Vec<(usize, usize)>, // (start, end) index in instances_buf
    num_instances: usize,
}

impl AdaBoost {
    /// Creates a new instance of [`AdaBoost`].
    /// This method initializes the AdaBoost parameters such as threshold,
    /// number of iterations, and number of threads.
    ///
    /// # Arguments
    /// * `threshold`: The threshold for stopping the training.
    /// * `num_iterations`: The maximum number of iterations for training.
    /// * `num_threads`: The number of threads to use for training (not used in this implementation).
    ///
    /// # Returns: A new instance of [`AdaBoost`].
    pub fn new(threshold: f64, num_iterations: usize, num_threads: usize) -> Self {
        AdaBoost {
            threshold,
            num_iterations,
            num_threads,
            instance_weights: vec![],
            model: vec![],
            features: vec![],
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
    pub fn initialize_features(&mut self, filename: &Path) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut map = BTreeMap::new(); // preserve order

        let mut buf_size = 0;
        self.num_instances = 0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let _label = parts.next();

            for h in parts {
                map.entry(h.to_string()).or_insert(0.0);
                buf_size += 1;
            }

            self.num_instances += 1;
            if self.num_instances.is_multiple_of(1000) {
                eprint!("\rfinding instances...: {} instances found", self.num_instances);
            }
        }

        eprintln!("\rfinding instances...: {} instances found", self.num_instances);

        map.insert("".to_string(), 0.0);

        self.features = map.keys().cloned().collect();
        self.model = map.values().cloned().collect();

        self.instance_weights.reserve(self.num_instances);
        self.labels.reserve(self.num_instances);
        self.instances.reserve(self.num_instances);
        self.instances_buf.reserve(buf_size);

        Ok(())
    }

    /// Initializes the instances from a file.
    /// The file should contain lines with a label followed by space-separated features.
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
    /// It also prints the progress of loading instances to the standard error output.
    pub fn initialize_instances(&mut self, filename: &Path) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let bias = self.get_bias();

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let label: Label = parts.next().unwrap().parse().unwrap();
            self.labels.push(label);

            let start = self.instances_buf.len();
            let mut score = bias;

            for h in parts {
                if let Ok(pos) = self.features.binary_search(&h.to_string()) {
                    self.instances_buf.push(pos);
                    score += self.model[pos];
                }
            }

            let end = self.instances_buf.len();
            self.instances.push((start, end));
            self.instance_weights.push((-2.0 * label as f64 * score).exp());
            if self.instance_weights.len().is_multiple_of(1000) {
                eprint!(
                    "\rloading instances...: {}/{} instances loaded",
                    self.instance_weights.len(),
                    self.num_instances
                );
            }
        }

        eprintln!(
            "\rloading instances...: {}/{} instances loaded",
            self.instance_weights.len(),
            self.num_instances
        );

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

        for t in 0..self.num_iterations {
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

            // Find the best hypothesis
            let mut h_best = 0;
            let mut best_error_rate = positive_weight_sum / instance_weight_sum;
            for (h, _) in errors.iter().enumerate().take(num_features).skip(1) {
                let mut e = errors[h] + positive_weight_sum;
                e /= instance_weight_sum;
                if (0.5 - e).abs() > (0.5 - best_error_rate).abs() {
                    h_best = h;
                    best_error_rate = e;
                }
            }

            eprint!("\rIteration {} - margin: {}", t, (0.5 - best_error_rate).abs());
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

            // Normalize instance weights
            let sum_w: f64 = self.instance_weights.iter().sum();
            for d in &mut self.instance_weights {
                *d /= sum_w;
            }
        }
        eprintln!();
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
    pub fn save_model(&self, filename: &Path) -> std::io::Result<()> {
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
    /// The URI can be a file path or a URL (http, https or file).
    /// The model should contain lines with a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// # Arguments
    /// * `uri`: The URI of the file containing the model.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the URI is invalid or the file cannot be read.
    pub async fn load_model(&mut self, uri: &str) -> std::io::Result<()> {
        if uri.contains("://") {
            let parts: Vec<&str> = uri.splitn(2, "://").collect();
            if parts.len() != 2 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Invalid URI: {}", uri),
                ));
            }
            let scheme = ModelScheme::from_str(parts[0]).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
            })?;
            match scheme {
                ModelScheme::Http | ModelScheme::Https => {
                    self.load_model_from_url(uri).await.map_err(|e| {
                        std::io::Error::other(format!("Failed to load model from URL: {}", e))
                    })
                }
                ModelScheme::File => {
                    #[cfg(target_arch = "wasm32")]
                    {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "file:// scheme is not supported in WASM environment. Use http:// or https:// URLs.",
                        ));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let path = Path::new(parts[1]);
                        self.load_model_from_file(path)
                    }
                }
            }
        } else {
            #[cfg(target_arch = "wasm32")]
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "Local file paths are not supported in WASM environment. Use http:// or https:// URLs.",
                ));
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let path = Path::new(uri);
                self.load_model_from_file(path)
            }
        }
    }

    /// Loads a model from a URL.
    /// The URL should point to a file containing lines with a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// # Arguments
    /// * `url`: The URL of the file containing the model.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the URL cannot be accessed or the file cannot be read.
    async fn load_model_from_url(&mut self, url: &str) -> std::io::Result<()> {
        // Create HTTP client with a custom user agent
        let client = Client::builder()
            .user_agent(format!("Litsea/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| std::io::Error::other(format!("Failed to create HTTP client: {}", e)))?;

        // Send GET request to the URL
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| std::io::Error::other(format!("Failed to download model: {}", e)))?;

        // Check if the response status is successful
        if !resp.status().is_success() {
            return Err(std::io::Error::other(format!(
                "Failed to download model: HTTP {}",
                resp.status()
            )));
        }

        // Read the response body
        let content = resp
            .bytes()
            .await
            .map_err(|e| std::io::Error::other(format!("Failed to read model content: {}", e)))?;

        let reader = BufReader::new(content.as_ref());
        self.parse_model_content(reader)
    }

    /// Parses model content from a buffered reader.
    /// This is a helper method used by both `load_model_from_file` and `load_model_from_url`.
    ///
    /// # Arguments
    /// * `reader`: A buffered reader containing the model data.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the content cannot be parsed.
    pub fn parse_model_content<R: BufRead>(&mut self, reader: R) -> std::io::Result<()> {
        let mut m: HashMap<String, f64> = HashMap::new();
        let mut bias = 0.0;

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let mut parts = line.split_whitespace();

            let h = parts.next().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Empty line at line {}", line_num + 1),
                )
            })?;

            if let Some(v) = parts.next() {
                let value: f64 = v.parse().map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid value at line {}: {}", line_num + 1, e),
                    )
                })?;
                m.insert(h.to_string(), value);
                bias += value;
            } else {
                let b: f64 = h.parse().map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid bias at line {}: {}", line_num + 1, e),
                    )
                })?;
                m.insert("".to_string(), -b * 2.0 - bias);
            }
        }

        let sorted: BTreeMap<_, _> = m.into_iter().collect();
        self.features = sorted.keys().cloned().collect();
        self.model = sorted.values().cloned().collect();
        Ok(())
    }

    /// Loads a model from a file.
    /// The file should contain lines with a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// # Arguments
    /// * `filename`: The path to the file containing the model.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the file cannot be read.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_model_from_file(&mut self, filename: &Path) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        self.parse_model_content(reader)
    }

    #[cfg(target_arch = "wasm32")]
    fn load_model_from_file(&mut self, _filename: &Path) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "File system access is not supported in WASM environment",
        ))
    }

    /// Adds a new instance to the model.
    /// The instance is represented by a set of attributes and a label.
    ///
    /// # Arguments
    /// * `attributes`: A `HashSet<String>` containing the attributes of the instance.
    /// * `label`: The label of the instance, represented as an `i8`.
    pub fn add_instance(&mut self, attributes: HashSet<String>, label: i8) {
        let start = self.instances_buf.len();
        let mut attrs: Vec<String> = attributes.into_iter().collect();
        attrs.sort();
        for attr in attrs.iter() {
            let feature_index = if let Some(pos) = self.features.iter().position(|f| f == attr) {
                pos
            } else {
                self.features.push(attr.clone());
                self.model.push(0.0);
                self.features.len() - 1
            };
            self.instances_buf.push(feature_index);
        }
        let end = self.instances_buf.len();
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
    pub fn predict(&self, attributes: HashSet<String>) -> i8 {
        let mut score = 0.0;
        for attr in attributes {
            if let Some(idx) = self.features.iter().position(|f| f == &attr) {
                score += self.model[idx];
            }
        }
        if score >= 0.0 {
            1
        } else {
            -1
        }
    }

    /// Gets the bias term of the model.
    /// The bias is calculated as the negative sum of the model weights divided by 2.
    ///
    /// # Returns:The bias term as a `f64`.
    pub fn get_bias(&self) -> f64 {
        -self.model.iter().sum::<f64>() / 2.0
    }

    /// Calculates and returns the performance metrics of the model on the training data.
    pub fn get_metrics(&self) -> Metrics {
        let bias = self.get_bias();
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

        let accuracy = (true_positives + true_negatives) as f64 / self.num_instances as f64 * 100.0;
        let precision =
            true_positives as f64 / (true_positives + false_positives).max(1) as f64 * 100.0;
        let recall =
            true_positives as f64 / (true_positives + false_negatives).max(1) as f64 * 100.0;

        Metrics {
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
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use tempfile::NamedTempFile;

    #[test]
    fn test_initialize_features() -> std::io::Result<()> {
        // Create a dummy features file
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        writeln!(features_file, "0 feat3")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 10, 1);
        learner.initialize_features(features_file.path())?;

        // Features is an ordered set that should contain ""(empty string), "feat1", "feat2", "feat3"
        assert!(learner.features.contains(&"".to_string()));
        assert!(learner.features.contains(&"feat1".to_string()));
        assert!(learner.features.contains(&"feat2".to_string()));
        assert!(learner.features.contains(&"feat3".to_string()));
        Ok(())
    }

    #[test]
    fn test_initialize_instances() -> std::io::Result<()> {
        // First, initialize features in the feature file.
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 10, 1);
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
    fn test_train() -> std::io::Result<()> {
        // Initialize features using a features file.
        let mut features_file = NamedTempFile::new()?;
        writeln!(features_file, "1 feat1 feat2")?;
        features_file.as_file().sync_all()?;

        let mut learner = AdaBoost::new(0.01, 3, 1);
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

        Ok(())
    }

    #[tokio::test]
    async fn test_save_and_load_model() -> std::io::Result<()> {
        // Prepare a dummy learner.
        let mut learner = AdaBoost::new(0.01, 10, 1);

        // Set the features and weights in advance.
        learner.features = vec!["feat1".to_string(), "feat2".to_string()];
        learner.model = vec![0.5, -0.3];

        // Save the model to a temporary file.
        let temp_model = NamedTempFile::new()?;
        learner.save_model(temp_model.path())?;

        // Load the model with a new learner.
        let mut learner2 = AdaBoost::new(0.01, 10, 1);
        learner2.load_model(temp_model.path().to_str().unwrap()).await?;

        // Check that the number of features and models match.
        assert_eq!(learner2.features.len(), learner.features.len());
        assert_eq!(learner2.model.len(), learner.model.len());

        Ok(())
    }

    #[test]
    fn test_add_instance_and_predict() {
        let mut learner = AdaBoost::new(0.01, 10, 1);

        // Here, features and model are empty in the initial state. They are newly registered by add_instance.
        let mut attrs = HashSet::new();
        attrs.insert("A".to_string());
        learner.add_instance(attrs.clone(), 1);

        // When the same attribute is passed to predict, score returns 1 based on the initial model value (0.0) (because score>=0).
        let prediction = learner.predict(attrs);
        assert_eq!(prediction, 1);
    }

    #[test]
    fn test_get_bias() {
        let mut learner = AdaBoost::new(0.01, 10, 1);

        // Set model weights as an example.
        learner.model = vec![0.2, 0.3, -0.1];

        // bias = -sum(model)/2 = -(0.2+0.3-0.1)/2 = -0.4/2 = -0.2
        assert!((learner.get_bias() + 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_get_metrics() {
        let mut learner = AdaBoost::new(0.01, 10, 1);

        // Set features and model for prediction
        learner.features = vec!["A".to_string(), "B".to_string()];
        learner.model = vec![0.5, -1.0];

        // Instance 1: Attribute “A” → score = 0.25 + 0.5 = 0.75 (positive example)
        let mut attrs1 = HashSet::new();
        attrs1.insert("A".to_string());
        learner.add_instance(attrs1, 1);

        // Instance 2: Attribute “B” → score = 0.25 + (-1.0) = -0.75 (negative example)
        let mut attrs2 = HashSet::new();
        attrs2.insert("B".to_string());
        learner.add_instance(attrs2, -1);

        let metrics = learner.get_metrics();
        assert_eq!(metrics.true_positives, 1);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.num_instances, 2);

        // Since this is a simple case, the accuracy is 100%.
        assert!((metrics.accuracy - 100.0).abs() < 1e-6);
    }
}
