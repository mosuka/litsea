use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{LitseaError, Result};
use crate::metrics::MulticlassMetrics;

/// Multiclass Averaged Perceptron classifier.
///
/// Performs multiclass classification over sparse binary features.
/// During training it keeps a running average of the weights to reduce
/// overfitting.
///
/// Weights are stored in a feature -> per-class vector layout. This reduces
/// the number of hash lookups per prediction from features x classes to
/// features, which makes the inference hot path significantly faster.
#[derive(Debug)]
pub struct AveragedPerceptron {
    /// Per-feature weight vectors: weights\[feature\]\[class_index\] = weight
    weights: HashMap<String, Vec<f64>>,
    /// Accumulated weights used for averaging
    accumulated: HashMap<String, Vec<f64>>,
    /// Last-updated timestamp of each weight
    timestamps: HashMap<String, Vec<usize>>,
    /// Current step count (total across all instances)
    step: usize,
    /// Known classes (always kept sorted)
    classes: Vec<String>,
    /// Training instances: (feature set, gold label)
    instances: Vec<(Vec<String>, String)>,
}

impl Default for AveragedPerceptron {
    fn default() -> Self {
        Self::new()
    }
}

impl AveragedPerceptron {
    /// Creates a new Averaged Perceptron instance.
    pub fn new() -> Self {
        AveragedPerceptron {
            weights: HashMap::new(),
            accumulated: HashMap::new(),
            timestamps: HashMap::new(),
            step: 0,
            classes: Vec::new(),
            instances: Vec::new(),
        }
    }

    /// Registers a class and returns its index.
    /// New classes are inserted in sorted order, and a matching column is
    /// inserted into every existing weight vector.
    fn ensure_class(&mut self, label: &str) -> usize {
        match self.classes.binary_search_by(|c| c.as_str().cmp(label)) {
            Ok(i) => i,
            Err(i) => {
                self.classes.insert(i, label.to_string());
                for v in self.weights.values_mut() {
                    v.insert(i, 0.0);
                }
                for v in self.accumulated.values_mut() {
                    v.insert(i, 0.0);
                }
                for v in self.timestamps.values_mut() {
                    v.insert(i, 0);
                }
                i
            }
        }
    }

    /// Adds a training instance.
    ///
    /// # Arguments
    /// * `features` - The feature set
    /// * `label` - The gold label
    pub fn add_instance(&mut self, features: HashSet<String>, label: String) {
        self.ensure_class(&label);
        let feats: Vec<String> = features.into_iter().collect();
        self.instances.push((feats, label));
    }

    /// Returns the index of the highest-scoring class for the features.
    /// Returns None if no classes are registered.
    fn predict_idx<I>(&self, features: I) -> Option<usize>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        if self.classes.is_empty() {
            return None;
        }
        let mut scores = vec![0.0f64; self.classes.len()];
        for feat in features {
            if let Some(ws) = self.weights.get(feat.as_ref()) {
                for (s, w) in scores.iter_mut().zip(ws.iter()) {
                    *s += *w;
                }
            }
        }
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, s) in scores.iter().enumerate() {
            if *s > best_score {
                best_score = *s;
                best = i;
            }
        }
        Some(best)
    }

    /// Predicts the label for a feature set.
    ///
    /// Computes the score of each class and returns the one with the highest
    /// score. Returns an empty string if no classes are registered.
    #[must_use]
    pub fn predict(&self, features: &HashSet<String>) -> String {
        match self.predict_idx(features.iter()) {
            Some(i) => self.classes[i].clone(),
            None => String::new(),
        }
    }

    /// Predicts the label from a slice (internal allocation-avoiding API).
    pub(crate) fn predict_slice(&self, features: &[String]) -> &str {
        match self.predict_idx(features.iter()) {
            Some(i) => &self.classes[i],
            None => "",
        }
    }

    /// Updates the weight of a single (feature, class) pair.
    /// Catches the accumulated weight up to the current step before adding
    /// `delta`.
    fn update_single(&mut self, feat: &str, class_idx: usize, delta: f64) {
        let n = self.classes.len();
        let ws = self.weights.entry(feat.to_string()).or_insert_with(|| vec![0.0; n]);
        let ts = self.timestamps.entry(feat.to_string()).or_insert_with(|| vec![0; n]);
        let acc = self.accumulated.entry(feat.to_string()).or_insert_with(|| vec![0.0; n]);

        let elapsed = self.step - ts[class_idx];
        if elapsed > 0 {
            acc[class_idx] += ws[class_idx] * elapsed as f64;
        }
        ts[class_idx] = self.step;
        ws[class_idx] += delta;
    }

    /// Updates the weights for one instance.
    ///
    /// When the prediction differs from the gold label:
    /// - the gold class weights are incremented by 1
    /// - the predicted class weights are decremented by 1
    fn update(&mut self, truth_idx: usize, guess_idx: usize, features: &[String]) {
        for feat in features {
            self.update_single(feat, truth_idx, 1.0);
            self.update_single(feat, guess_idx, -1.0);
        }
    }

    /// Writes the averaged weights into the final model.
    fn average_weights(&mut self) {
        let n = self.classes.len();
        let step = self.step.max(1) as f64;
        let feats: Vec<String> = self.weights.keys().cloned().collect();
        for feat in feats {
            for class_idx in 0..n {
                // An update with delta 0 catches the accumulated weight up to
                // the current step
                self.update_single(&feat, class_idx, 0.0);
                let acc = self.accumulated[&feat][class_idx];
                if let Some(ws) = self.weights.get_mut(&feat) {
                    ws[class_idx] = acc / step;
                }
            }
        }
    }

    /// Trains the model.
    ///
    /// # Arguments
    /// * `num_epochs` - The number of epochs
    /// * `running` - A flag for interrupting the training
    pub fn train(&mut self, num_epochs: usize, running: Arc<AtomicBool>) {
        if self.instances.is_empty() {
            return;
        }

        // Temporarily move the instances out to avoid double borrows during
        // training (previously every instance was cloned).
        let instances = std::mem::take(&mut self.instances);

        for _epoch in 0..num_epochs {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            for (features, truth) in &instances {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                let guess_idx = self.predict_idx(features.iter()).expect("classes registered");
                let truth_idx = self
                    .classes
                    .binary_search_by(|c| c.as_str().cmp(truth))
                    .expect("truth class registered by add_instance");
                if guess_idx != truth_idx {
                    self.update(truth_idx, guess_idx, features);
                }
                self.step += 1;
            }
        }

        self.instances = instances;

        // Write the averaged weights into the final model
        self.average_weights();
    }

    /// Saves the model to a file as text (class header + TSV).
    ///
    /// Format:
    /// ```text
    /// <number of classes>
    /// <class name 1>
    /// <class name 2>
    /// ...
    /// <feature>\t<class>\t<weight>
    /// <feature>\t<class>\t<weight>
    /// ...
    /// ```
    pub fn save_model(&self, path: &Path) -> Result<()> {
        if self.classes.is_empty() {
            return Err(LitseaError::InvalidInput("Cannot save an empty model".to_string()));
        }

        let mut file = File::create(path)?;

        // Header: the number of classes and the class names
        writeln!(file, "{}", self.classes.len())?;
        for class in &self.classes {
            writeln!(file, "{}", class)?;
        }

        // Weights: only non-zero weights are saved
        for (feat, ws) in &self.weights {
            for (class_idx, &w) in ws.iter().enumerate() {
                if w != 0.0 {
                    writeln!(file, "{}\t{}\t{}", feat, self.classes[class_idx], w)?;
                }
            }
        }

        Ok(())
    }

    /// Loads a model from a URI.
    ///
    /// The URI can be a file path, a `file://` path, or an `http(s)://` URL
    /// (the latter requires the `remote_model` feature).
    /// For local files, prefer the synchronous
    /// [`load_model_from_path`](Self::load_model_from_path).
    pub async fn load_model(&mut self, uri: &str) -> Result<()> {
        let bytes = crate::model_io::read_model_bytes(uri).await?;
        self.load_model_from_reader(bytes.as_slice())
    }

    /// Loads a model from a local file path (synchronous).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_model_from_path(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        self.load_model_from_reader(BufReader::new(file))
    }

    /// Loads a model from a buffered reader (synchronous).
    ///
    /// If classes are already registered from training instances, the classes
    /// in the model file are merged into the existing class list so that gold
    /// label classes are not lost during incremental training.
    pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        let mut lines = reader.lines();

        // Read the number of classes
        let num_classes: usize = lines
            .next()
            .ok_or_else(|| LitseaError::InvalidData("Empty model file".to_string()))?
            .map_err(|e| LitseaError::InvalidData(format!("Read error: {}", e)))?
            .trim()
            .parse()
            .map_err(|e| LitseaError::InvalidData(format!("Invalid class count: {}", e)))?;

        // Read the class names (merging with existing classes)
        for _ in 0..num_classes {
            let class = lines
                .next()
                .ok_or_else(|| {
                    LitseaError::InvalidData(
                        "Unexpected end of model file while reading classes".to_string(),
                    )
                })?
                .map_err(|e| LitseaError::InvalidData(format!("Read error: {}", e)))?;
            self.ensure_class(class.trim());
        }

        // Read the weights
        self.weights.clear();
        let n = self.classes.len();
        for line in lines {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                return Err(LitseaError::InvalidData(format!("Invalid weight line: '{}'", line)));
            }
            let feat = parts[0];
            let class_idx =
                self.classes.binary_search_by(|c| c.as_str().cmp(parts[1])).map_err(|_| {
                    LitseaError::InvalidData(format!("Unknown class in weight line: '{}'", line))
                })?;
            let weight: f64 = parts[2]
                .parse()
                .map_err(|e| LitseaError::InvalidData(format!("Invalid weight value: {}", e)))?;
            self.weights.entry(feat.to_string()).or_insert_with(|| vec![0.0; n])[class_idx] =
                weight;
        }

        Ok(())
    }

    /// Computes evaluation metrics on the training data.
    #[must_use]
    pub fn metrics(&self) -> MulticlassMetrics {
        let mut correct_per_class: HashMap<String, usize> = HashMap::new();
        let mut predicted_per_class: HashMap<String, usize> = HashMap::new();
        let mut gold_per_class: HashMap<String, usize> = HashMap::new();
        let mut total_correct = 0usize;

        for (features, truth) in &self.instances {
            let guess = match self.predict_idx(features.iter()) {
                Some(i) => self.classes[i].as_str(),
                None => "",
            };

            *gold_per_class.entry(truth.clone()).or_insert(0) += 1;
            *predicted_per_class.entry(guess.to_string()).or_insert(0) += 1;

            if guess == truth {
                total_correct += 1;
                *correct_per_class.entry(truth.clone()).or_insert(0) += 1;
            }
        }

        let num_instances = self.instances.len();
        let accuracy = total_correct as f64 / num_instances.max(1) as f64 * 100.0;

        // Macro-averaged precision and recall
        let mut sum_precision = 0.0;
        let mut sum_recall = 0.0;
        let num_classes = self.classes.len().max(1);

        for class in &self.classes {
            let correct = correct_per_class.get(class).copied().unwrap_or(0) as f64;
            let predicted = predicted_per_class.get(class).copied().unwrap_or(0) as f64;
            let gold = gold_per_class.get(class).copied().unwrap_or(0) as f64;

            if predicted > 0.0 {
                sum_precision += correct / predicted;
            }
            if gold > 0.0 {
                sum_recall += correct / gold;
            }
        }

        MulticlassMetrics {
            accuracy,
            macro_precision: sum_precision / num_classes as f64 * 100.0,
            macro_recall: sum_recall / num_classes as f64 * 100.0,
            num_instances,
            correct_per_class,
            predicted_per_class,
            gold_per_class,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use tempfile::NamedTempFile;

    #[test]
    fn test_new() {
        let p = AveragedPerceptron::new();
        assert!(p.classes.is_empty());
        assert!(p.weights.is_empty());
        assert_eq!(p.step, 0);
    }

    #[test]
    fn test_add_instance() {
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "A".to_string());
        assert_eq!(p.instances.len(), 1);
        assert_eq!(p.classes, vec!["A"]);
    }

    #[test]
    fn test_classes_stay_sorted() {
        let mut p = AveragedPerceptron::new();
        for label in ["C", "A", "B", "A"] {
            let mut feats = HashSet::new();
            feats.insert(format!("f_{}", label));
            p.add_instance(feats, label.to_string());
        }
        assert_eq!(p.classes, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_predict_empty() {
        let p = AveragedPerceptron::new();
        let feats = HashSet::new();
        assert_eq!(p.predict(&feats), "");
    }

    #[test]
    fn test_train_simple() {
        let mut p = AveragedPerceptron::new();

        // Class A features: f1, f2
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        feats_a.insert("f2".to_string());
        p.add_instance(feats_a.clone(), "A".to_string());

        // Class B features: f3, f4
        let mut feats_b = HashSet::new();
        feats_b.insert("f3".to_string());
        feats_b.insert("f4".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(10, running);

        // After training, instances are classified correctly
        assert_eq!(p.predict(&feats_a), "A");
        assert_eq!(p.predict(&feats_b), "B");
    }

    #[test]
    fn test_train_immediate_stop() {
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "A".to_string());

        let running = Arc::new(AtomicBool::new(false));
        p.train(10, running);

        // Stopping immediately must not panic
        assert_eq!(p.step, 0);
        // Instances are not lost
        assert_eq!(p.instances.len(), 1);
    }

    #[test]
    fn test_train_multiclass() {
        let mut p = AveragedPerceptron::new();

        // Training data for three classes
        for _ in 0..5 {
            let mut fa = HashSet::new();
            fa.insert("feat_a".to_string());
            fa.insert("shared".to_string());
            p.add_instance(fa, "CLASS_A".to_string());

            let mut fb = HashSet::new();
            fb.insert("feat_b".to_string());
            fb.insert("shared".to_string());
            p.add_instance(fb, "CLASS_B".to_string());

            let mut fc = HashSet::new();
            fc.insert("feat_c".to_string());
            fc.insert("shared".to_string());
            p.add_instance(fc, "CLASS_C".to_string());
        }

        let running = Arc::new(AtomicBool::new(true));
        p.train(20, running);

        // Distinctive features classify correctly
        let mut test_a = HashSet::new();
        test_a.insert("feat_a".to_string());
        test_a.insert("shared".to_string());
        assert_eq!(p.predict(&test_a), "CLASS_A");

        let mut test_b = HashSet::new();
        test_b.insert("feat_b".to_string());
        test_b.insert("shared".to_string());
        assert_eq!(p.predict(&test_b), "CLASS_B");
    }

    #[test]
    fn test_predict_slice_matches_predict() {
        let mut p = AveragedPerceptron::new();
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        p.add_instance(feats_a.clone(), "A".to_string());
        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(10, running);

        let slice_a: Vec<String> = feats_a.iter().cloned().collect();
        assert_eq!(p.predict_slice(&slice_a), p.predict(&feats_a));
        let slice_b: Vec<String> = feats_b.iter().cloned().collect();
        assert_eq!(p.predict_slice(&slice_b), p.predict(&feats_b));
    }

    #[test]
    fn test_save_and_load_model() -> Result<()> {
        let mut p = AveragedPerceptron::new();
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        p.add_instance(feats_a.clone(), "A".to_string());

        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(5, running);

        // Save
        let temp = NamedTempFile::new()?;
        p.save_model(temp.path())?;

        // Load (synchronous path API)
        let mut p2 = AveragedPerceptron::new();
        p2.load_model_from_path(temp.path())?;

        // The same classes are restored
        assert_eq!(p2.classes.len(), p.classes.len());

        // The same predictions are produced
        assert_eq!(p.predict(&feats_a), p2.predict(&feats_a));
        assert_eq!(p.predict(&feats_b), p2.predict(&feats_b));

        Ok(())
    }

    #[tokio::test]
    async fn test_load_model_uri() -> Result<()> {
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "A".to_string());
        let running = Arc::new(AtomicBool::new(true));
        p.train(5, running);

        let temp = NamedTempFile::new()?;
        p.save_model(temp.path())?;

        let mut p2 = AveragedPerceptron::new();
        p2.load_model(temp.path().to_str().unwrap()).await?;
        assert_eq!(p2.classes.len(), p.classes.len());
        Ok(())
    }

    #[test]
    fn test_load_model_merges_classes() -> Result<()> {
        // Incremental training: classes already present in the training data
        // must survive a model load.
        let mut p = AveragedPerceptron::new();
        let mut feats = HashSet::new();
        feats.insert("f1".to_string());
        p.add_instance(feats, "NEW_CLASS".to_string());

        // Load a model that only contains class A
        let model_content = "1\nA\nf1\tA\t0.5\n";
        p.load_model_from_reader(model_content.as_bytes())?;

        assert!(p.classes.contains(&"A".to_string()));
        assert!(p.classes.contains(&"NEW_CLASS".to_string()));
        Ok(())
    }

    #[test]
    fn test_save_model_empty() {
        let p = AveragedPerceptron::new();
        let temp = NamedTempFile::new().unwrap();
        let result = p.save_model(temp.path());
        assert!(matches!(result, Err(LitseaError::InvalidInput(_))));
    }

    #[test]
    fn test_metrics() {
        let mut p = AveragedPerceptron::new();

        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        p.add_instance(feats_a, "A".to_string());

        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());
        p.add_instance(feats_b, "B".to_string());

        let running = Arc::new(AtomicBool::new(true));
        p.train(10, running);

        let metrics = p.metrics();
        assert_eq!(metrics.num_instances, 2);
        assert!(metrics.accuracy > 0.0);
    }

    #[test]
    fn test_metrics_empty() {
        let p = AveragedPerceptron::new();
        let metrics = p.metrics();
        assert_eq!(metrics.num_instances, 0);
        assert!((metrics.accuracy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_model_from_reader_invalid() {
        let mut p = AveragedPerceptron::new();
        // Invalid class count
        let result = p.load_model_from_reader("not_a_number".as_bytes());
        assert!(matches!(result, Err(LitseaError::InvalidData(_))));
    }

    #[test]
    fn test_load_model_from_reader_empty() {
        let mut p = AveragedPerceptron::new();
        let result = p.load_model_from_reader("".as_bytes());
        assert!(matches!(result, Err(LitseaError::InvalidData(_))));
    }
}
