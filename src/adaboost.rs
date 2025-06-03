use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

type Label = i8;

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
            if self.num_instances % 1000 == 0 {
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

            if self.instance_weights.len() % 1000 == 0 {
                eprint!(
                    "\rloading instances...: {}/{} instances loaded",
                    self.instance_weights.len(),
                    self.num_instances
                );
            }
        }
        eprintln!();
        Ok(())
    }

    /// Trains the AdaBoost model.
    /// This method iteratively updates the model based on the training data.
    ///
    /// # Arguments
    /// * `running`: An `Arc<AtomicBool>` to control the running state of the training process.
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

    /// Loads a model from a file.
    /// The file should contain lines with a feature and its weight,
    /// with the last line containing the bias term.
    ///
    /// # Arguments
    /// * `filename`: The path to the file containing the model.
    ///
    /// # Returns: A result indicating success or failure.
    ///
    /// # Errors: Returns an error if the file cannot be opened or read.
    pub fn load_model(&mut self, filename: &Path) -> std::io::Result<()> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut m: HashMap<String, f64> = HashMap::new();
        let mut bias = 0.0;

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let h = parts.next().unwrap();
            if let Some(v) = parts.next() {
                let value: f64 = v.parse().unwrap();
                m.insert(h.to_string(), value);
                bias += value;
            } else {
                let b: f64 = h.parse().unwrap();
                m.insert("".to_string(), -b * 2.0 - bias);
            }
        }

        let sorted: BTreeMap<_, _> = m.into_iter().collect();
        self.features = sorted.keys().cloned().collect();
        self.model = sorted.values().cloned().collect();
        Ok(())
    }

    /// Gets the bias term of the model.
    /// The bias is calculated as the negative sum of the model weights divided by 2.
    ///
    /// # Returns:The bias term as a `f64`.
    pub fn get_bias(&self) -> f64 {
        -self.model.iter().sum::<f64>() / 2.0
    }

    /// Displays the result of the model's performance on the training data.
    /// It calculates accuracy, precision, recall, and confusion matrix.
    pub fn show_result(&self) {
        let bias = self.get_bias();
        let mut pp = 0;
        let mut pn = 0;
        let mut np = 0;
        let mut nn = 0;

        for i in 0..self.num_instances {
            let label = self.labels[i];
            let (start, end) = self.instances[i];
            let mut score = bias;
            for &h in &self.instances_buf[start..end] {
                score += self.model[h];
            }

            if score >= 0.0 {
                if label > 0 {
                    pp += 1
                } else {
                    pn += 1
                }
            } else if label > 0 {
                np += 1
            } else {
                nn += 1
            }
        }

        let acc = (pp + nn) as f64 / self.num_instances as f64 * 100.0;
        let prec = pp as f64 / (pp + pn).max(1) as f64 * 100.0;
        let recall = pp as f64 / (pp + np).max(1) as f64 * 100.0;

        eprintln!("Result:");
        eprintln!("Accuracy: {:.2}% ({} / {})", acc, pp + nn, self.num_instances);
        eprintln!("Precision: {:.2}% ({} / {})", prec, pp, pp + pn);
        eprintln!("Recall: {:.2}% ({} / {})", recall, pp, pp + np);
        eprintln!("Confusion Matrix: TP: {}, FP: {}, FN: {}, TN: {}", pp, pn, np, nn);
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
}
