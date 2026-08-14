//! Multiclass Averaged Perceptron for joint segmentation + POS tagging.
//!
//! Defines [`AveragedPerceptron`]: training with weight averaging,
//! text-format model I/O, and training-set metrics. Its weights back
//! [`Segmenter::segment_with_pos`](crate::segmenter::Segmenter::segment_with_pos)
//! through the packed POS tables compiled in [`crate::packed_pos_model`].

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

// Internal weight maps use FxHashMap: keys are internally generated feature
// strings (no HashDoS exposure). Since the packed POS scorer (issue #143),
// segment_with_pos() inference performs no per-key string probing (that
// survives only in the test-only reference path); the map is hashed during
// training, model I/O, and packed-table compilation, where FxHash beats the
// default SipHash.
use rustc_hash::FxHashMap;

use crate::error::{LitseaError, Result};
use crate::metrics::MulticlassMetrics;

/// Per-feature training state, one entry per class: live weights (`w`),
/// averaging accumulators (`acc`), and the step at which each weight was
/// last updated (`ts`).
///
/// Keeping the three vectors in one slot means a weight update needs a
/// single hashed lookup instead of one per map (issue #104). `acc`/`ts`
/// stay empty until training first touches the feature — model loading for
/// inference-only use must not pay for averaging state it never needs
/// (materializing them eagerly doubled the loaded-model memory footprint).
#[derive(Debug)]
struct FeatureSlot {
    w: Vec<f64>,
    acc: Vec<f64>,
    ts: Vec<usize>,
}

impl FeatureSlot {
    /// Creates a slot with zeroed weights for `n` classes; the averaging
    /// state stays empty until [`ensure_averaging`](Self::ensure_averaging).
    fn new(n: usize) -> Self {
        FeatureSlot {
            w: vec![0.0; n],
            acc: Vec::new(),
            ts: Vec::new(),
        }
    }

    /// Materializes the averaging state (all zeros, equivalent to "never
    /// updated") the first time training touches this slot.
    fn ensure_averaging(&mut self) {
        if self.acc.is_empty() {
            self.acc.resize(self.w.len(), 0.0);
            self.ts.resize(self.w.len(), 0);
        }
    }
}

/// Multiclass Averaged Perceptron classifier.
///
/// Performs multiclass classification over sparse binary features.
/// During training it keeps a running average of the weights to reduce
/// overfitting.
///
/// Weights are stored in a feature -> per-class vector layout, so one hashed
/// lookup yields the weights of every class at once. Production inference
/// goes through the packed POS tables compiled from these weights
/// (`crate::packed_pos_model`); the string-keyed prediction here serves
/// training and the test-only reference path.
#[derive(Debug)]
pub struct AveragedPerceptron {
    /// Per-feature training state: slots\[feature\] holds the live weights,
    /// averaging accumulators, and update timestamps for every class.
    slots: FxHashMap<String, FeatureSlot>,
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
    ///
    /// # Returns
    /// A new [`AveragedPerceptron`] with no classes, weights, or training
    /// instances.
    pub fn new() -> Self {
        AveragedPerceptron {
            slots: FxHashMap::default(),
            step: 0,
            classes: Vec::new(),
            instances: Vec::new(),
        }
    }

    /// Registers a class and returns its index.
    /// New classes are inserted in sorted order, and a matching column is
    /// inserted into every existing feature slot.
    fn ensure_class(&mut self, label: &str) -> usize {
        match self.classes.binary_search_by(|c| c.as_str().cmp(label)) {
            Ok(i) => i,
            Err(i) => {
                self.classes.insert(i, label.to_string());
                for slot in self.slots.values_mut() {
                    slot.w.insert(i, 0.0);
                    // acc/ts are lazily materialized; only shift them when
                    // they exist.
                    if !slot.acc.is_empty() {
                        slot.acc.insert(i, 0.0);
                        slot.ts.insert(i, 0);
                    }
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

    /// Returns the index of the highest-scoring class for the features,
    /// reusing `scores` as a scratch buffer (cleared and resized here) to
    /// avoid one heap allocation per prediction on the inference hot path.
    /// Returns None if no classes are registered.
    fn predict_idx_into<I>(&self, features: I, scores: &mut Vec<f64>) -> Option<usize>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        if self.classes.is_empty() {
            return None;
        }
        scores.clear();
        scores.resize(self.classes.len(), 0.0);
        for feat in features {
            if let Some(slot) = self.slots.get(feat.as_ref()) {
                for (s, w) in scores.iter_mut().zip(slot.w.iter()) {
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
        let mut scores = Vec::new();
        match self.predict_idx_into(features.iter(), &mut scores) {
            Some(i) => self.classes[i].clone(),
            None => String::new(),
        }
    }

    /// Predicts the label from a slice, reusing the caller's scratch buffer
    /// (internal allocation-avoiding API).
    ///
    /// Test-only since the packed POS scorer (issue #143): kept as part of
    /// the string-keyed reference path that the differential tests pin
    /// against.
    #[cfg(test)]
    pub(crate) fn predict_slice(&self, features: &[String], scores: &mut Vec<f64>) -> &str {
        match self.predict_idx_into(features.iter(), scores) {
            Some(i) => &self.classes[i],
            None => "",
        }
    }

    /// Iterates over every known feature and its per-class weight row (row
    /// entries are in class-index order, parallel to
    /// [`class_names`](Self::class_names)). Internal accessor used to compile
    /// the packed POS scoring tables
    /// ([`crate::packed_pos_model::PackedPosModel`]).
    pub(crate) fn feature_class_weights(&self) -> impl Iterator<Item = (&str, &[f64])> + '_ {
        self.slots.iter().map(|(feat, slot)| (feat.as_str(), slot.w.as_slice()))
    }

    /// Returns the known class names in sorted order (the class-index order
    /// used by every per-class weight row and by prediction tie-breaking).
    /// Internal accessor for the packed POS scoring tables.
    pub(crate) fn class_names(&self) -> &[String] {
        &self.classes
    }

    /// Updates the weight of a single (feature, class) pair.
    /// Catches the accumulated weight up to the current step before adding
    /// `delta`. One hashed lookup, and no allocation when the feature is
    /// already known (get-then-insert instead of the owned-key entry API).
    fn update_single(&mut self, feat: &str, class_idx: usize, delta: f64) {
        let slot = match self.slots.get_mut(feat) {
            Some(slot) => slot,
            None => {
                let n = self.classes.len();
                self.slots.entry(feat.to_string()).or_insert_with(|| FeatureSlot::new(n))
            }
        };
        slot.ensure_averaging();

        let elapsed = self.step - slot.ts[class_idx];
        if elapsed > 0 {
            slot.acc[class_idx] += slot.w[class_idx] * elapsed as f64;
        }
        slot.ts[class_idx] = self.step;
        slot.w[class_idx] += delta;
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
    ///
    /// A single pass over the slots: each (feature, class) accumulator is
    /// caught up to the current step and the live weight is replaced by the
    /// average. Pairs are independent, so map iteration order cannot affect
    /// the result (same math as the previous per-key update_single loop,
    /// without cloning every key and re-looking each one up).
    fn average_weights(&mut self) {
        let step_now = self.step;
        let step = self.step.max(1) as f64;
        for slot in self.slots.values_mut() {
            slot.ensure_averaging();
            for class_idx in 0..slot.w.len() {
                let elapsed = step_now - slot.ts[class_idx];
                if elapsed > 0 {
                    slot.acc[class_idx] += slot.w[class_idx] * elapsed as f64;
                }
                slot.ts[class_idx] = step_now;
                slot.w[class_idx] = slot.acc[class_idx] / step;
            }
        }
    }

    /// Trains the model.
    ///
    /// # Arguments
    /// * `num_epochs` - The number of epochs
    /// * `running` - A flag for interrupting the training
    pub fn train(&mut self, num_epochs: usize, running: &AtomicBool) {
        if self.instances.is_empty() {
            return;
        }

        // Temporarily move the instances out to avoid double borrows during
        // training (previously every instance was cloned).
        let instances = std::mem::take(&mut self.instances);
        // Scratch buffer reused across every prediction in the epoch loop.
        let mut scores: Vec<f64> = Vec::new();

        for _epoch in 0..num_epochs {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            for (features, truth) in &instances {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                // Invariant: instances are non-empty here, and add_instance
                // registers a class for every instance, so classes cannot be
                // empty; degrade gracefully instead of panicking if the
                // invariant is ever broken.
                let Some(guess_idx) = self.predict_idx_into(features.iter(), &mut scores) else {
                    break;
                };
                // Invariant: add_instance registered the gold class.
                let Ok(truth_idx) = self.classes.binary_search_by(|c| c.as_str().cmp(truth)) else {
                    continue;
                };
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
    /// Weight lines are written in sorted feature order, so saving the same
    /// model always produces byte-identical files (reproducible and
    /// diffable); loading does not depend on the order. Output is buffered
    /// (one syscall per line would mean ~half a million syscalls for the
    /// shipped POS models).
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
    ///
    /// # Arguments
    /// * `path` - The path of the file to write the model to.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if no classes are registered
    /// (an empty model), or an I/O error if the file cannot be created or
    /// written.
    pub fn save_model(&self, path: &Path) -> Result<()> {
        let mut file = io::BufWriter::new(File::create(path)?);
        self.save_model_to_writer(&mut file)?;
        file.flush()?;
        Ok(())
    }

    /// Writes the model to an arbitrary writer in the same text format as
    /// [`save_model`](Self::save_model).
    ///
    /// This is the format-producing core of `save_model`; it exists so the
    /// model can be embedded as a section of a larger file (the two-stage
    /// model format). The writer is not flushed.
    ///
    /// # Arguments
    /// * `writer` - The writer receiving the model text.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if no classes are registered
    /// (an empty model), or an I/O error if writing fails.
    pub fn save_model_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        if self.classes.is_empty() {
            return Err(LitseaError::InvalidInput("Cannot save an empty model".to_string()));
        }

        // Header: the number of classes and the class names
        writeln!(writer, "{}", self.classes.len())?;
        for class in &self.classes {
            writeln!(writer, "{}", class)?;
        }

        // Weights: only non-zero weights are saved, in sorted feature order.
        let mut feats: Vec<&String> = self.slots.keys().collect();
        feats.sort_unstable();
        for feat in feats {
            let slot = &self.slots[feat];
            for (class_idx, &w) in slot.w.iter().enumerate() {
                if w != 0.0 {
                    writeln!(writer, "{}\t{}\t{}", feat, self.classes[class_idx], w)?;
                }
            }
        }

        Ok(())
    }

    /// Returns the registered class names in their sorted storage order
    /// (the order used for weight-vector indexing and argmax tie-breaking).
    ///
    /// # Returns
    /// A slice of class-name strings; empty if the model holds no classes.
    #[must_use]
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Loads a model from a URI.
    ///
    /// The URI can be a file path, a `file://` path, or an `http(s)://` URL
    /// (the latter requires the `remote_model` feature).
    /// For local files, prefer the synchronous
    /// [`load_model_from_path`](Self::load_model_from_path).
    ///
    /// # Arguments
    /// * `uri` - The URI of the model to load.
    ///
    /// # Errors
    /// Returns an error if the model bytes cannot be fetched from the URI
    /// (unsupported scheme, missing file, or network failure) or if the
    /// model content is malformed (see
    /// [`load_model_from_reader`](Self::load_model_from_reader)).
    pub async fn load_model(&mut self, uri: &str) -> Result<()> {
        let bytes = crate::model_io::read_model_bytes(uri).await?;
        self.load_model_from_reader(bytes.as_slice())
    }

    /// Loads a model from a local file path (synchronous).
    ///
    /// # Arguments
    /// * `path` - The path of the model file to load.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be opened, or a parse error
    /// if the model content is malformed (see
    /// [`load_model_from_reader`](Self::load_model_from_reader)).
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
    ///
    /// # Arguments
    /// * `reader` - The buffered reader providing the model content (the
    ///   text format written by [`save_model`](Self::save_model)).
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidData`] if the model file is empty, the
    /// class count is not a valid number, the file ends while reading the
    /// class names, a weight line does not have exactly three tab-separated
    /// fields, a weight line names an unknown class, or a weight value is
    /// unparsable or non-finite. I/O errors from the reader are also
    /// propagated.
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

        // Read the weights. Reset all learned state: the weights are replaced
        // by the file's, and the averaging accumulators must not survive into
        // a later train() call (they would combine stale timestamps with the
        // loaded weights and corrupt the averaged model). Fresh slots carry
        // zeroed accumulators, so replacing the map does exactly that.
        self.slots.clear();
        self.step = 0;
        let n = self.classes.len();
        for line in lines {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Parse "feature\tclass\tweight" without collecting into a Vec.
            let Some((feat, rest)) = line.split_once('\t') else {
                return Err(LitseaError::InvalidData(format!("Invalid weight line: '{}'", line)));
            };
            let Some((class, weight_str)) = rest.split_once('\t') else {
                return Err(LitseaError::InvalidData(format!("Invalid weight line: '{}'", line)));
            };
            if weight_str.contains('\t') {
                return Err(LitseaError::InvalidData(format!("Invalid weight line: '{}'", line)));
            }
            let class_idx =
                self.classes.binary_search_by(|c| c.as_str().cmp(class)).map_err(|_| {
                    LitseaError::InvalidData(format!("Unknown class in weight line: '{}'", line))
                })?;
            let weight: f64 = weight_str
                .parse()
                .map_err(|e| LitseaError::InvalidData(format!("Invalid weight value: {}", e)))?;
            if !weight.is_finite() {
                return Err(LitseaError::InvalidData(format!(
                    "Non-finite weight in line: '{}'",
                    line
                )));
            }
            // Avoid the owned-key allocation when the feature already exists
            // (each feature appears once per non-zero class in the file).
            let slot = match self.slots.get_mut(feat) {
                Some(slot) => slot,
                None => self.slots.entry(feat.to_string()).or_insert_with(|| FeatureSlot::new(n)),
            };
            slot.w[class_idx] = weight;
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
        let mut scores: Vec<f64> = Vec::new();

        for (features, truth) in &self.instances {
            let guess = match self.predict_idx_into(features.iter(), &mut scores) {
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

    use std::sync::atomic::AtomicBool;

    use tempfile::NamedTempFile;

    #[test]
    fn test_new() {
        let p = AveragedPerceptron::new();
        assert!(p.classes.is_empty());
        assert!(p.slots.is_empty());
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

        let running = AtomicBool::new(true);
        p.train(10, &running);

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

        let running = AtomicBool::new(false);
        p.train(10, &running);

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

        let running = AtomicBool::new(true);
        p.train(20, &running);

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

        let running = AtomicBool::new(true);
        p.train(10, &running);

        let mut scores: Vec<f64> = Vec::new();
        let slice_a: Vec<String> = feats_a.iter().cloned().collect();
        assert_eq!(p.predict_slice(&slice_a, &mut scores), p.predict(&feats_a));
        let slice_b: Vec<String> = feats_b.iter().cloned().collect();
        assert_eq!(p.predict_slice(&slice_b, &mut scores), p.predict(&feats_b));
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

        let running = AtomicBool::new(true);
        p.train(5, &running);

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
        let running = AtomicBool::new(true);
        p.train(5, &running);

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
    fn test_save_model_sorted_order() -> Result<()> {
        // #104: weight lines are written in sorted feature order so that
        // saving the same model always produces byte-identical, diffable
        // files (map iteration order previously leaked into the output).
        let mut p = AveragedPerceptron::new();
        for name in ["zeta", "alpha", "mid", "beta", "omega", "kappa", "echo"] {
            let mut feats = HashSet::new();
            feats.insert(name.to_string());
            let label = if name.len() % 2 == 0 { "A" } else { "B" };
            p.add_instance(feats, label.to_string());
        }
        p.train(5, &AtomicBool::new(true));

        let temp = NamedTempFile::new()?;
        p.save_model(temp.path())?;

        let content = std::fs::read_to_string(temp.path())?;
        // Skip the class header (count + class names), then check ordering.
        let feats: Vec<&str> = content
            .lines()
            .skip(1 + p.classes.len())
            .filter_map(|l| l.split('\t').next())
            .collect();
        let mut sorted = feats.clone();
        sorted.sort_unstable();
        assert_eq!(feats, sorted, "weight lines must be in sorted feature order");
        assert!(feats.len() > 1, "test needs multiple weight lines");
        Ok(())
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

        let running = AtomicBool::new(true);
        p.train(10, &running);

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
    fn test_load_model_rejects_nonfinite_weight() {
        // #101: NaN/inf weights must be rejected at load time.
        for content in ["1\nA\nf1\tA\tNaN\n", "1\nA\nf1\tA\tinf\n"] {
            let mut p = AveragedPerceptron::new();
            let result = p.load_model_from_reader(content.as_bytes());
            assert!(
                matches!(result, Err(LitseaError::InvalidData(_))),
                "expected InvalidData for {:?}",
                content
            );
        }
    }

    #[test]
    fn test_load_model_resets_averaging_state() -> Result<()> {
        // #101: loading a model must reset the averaging accumulators so a
        // subsequent train() does not mix stale state with loaded weights.
        let mut feats_a = HashSet::new();
        feats_a.insert("f1".to_string());
        let mut feats_b = HashSet::new();
        feats_b.insert("f2".to_string());

        let mut p = AveragedPerceptron::new();
        p.add_instance(feats_a.clone(), "A".to_string());
        p.add_instance(feats_b.clone(), "B".to_string());
        p.train(5, &AtomicBool::new(true));
        assert!(p.step > 0);

        let model_content = "2\nA\nB\nf1\tA\t0.5\nf2\tB\t0.5\n";
        p.load_model_from_reader(model_content.as_bytes())?;
        assert_eq!(p.step, 0);
        // Fresh slots after a load carry zeroed averaging state.
        assert!(p.slots.values().all(|s| s.acc.iter().all(|&a| a == 0.0)));
        assert!(p.slots.values().all(|s| s.ts.iter().all(|&t| t == 0)));

        // Behavioral check: train-after-load on the recycled instance matches
        // load-then-train on a fresh perceptron with the same instances.
        p.train(5, &AtomicBool::new(true));

        let mut q = AveragedPerceptron::new();
        q.add_instance(feats_a.clone(), "A".to_string());
        q.add_instance(feats_b.clone(), "B".to_string());
        q.load_model_from_reader(model_content.as_bytes())?;
        q.train(5, &AtomicBool::new(true));

        assert_eq!(p.predict(&feats_a), q.predict(&feats_a));
        assert_eq!(p.predict(&feats_b), q.predict(&feats_b));
        Ok(())
    }

    #[test]
    fn test_load_model_from_reader_empty() {
        let mut p = AveragedPerceptron::new();
        let result = p.load_model_from_reader("".as_bytes());
        assert!(matches!(result, Err(LitseaError::InvalidData(_))));
    }
}
