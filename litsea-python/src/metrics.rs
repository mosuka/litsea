//! Training metric classes.

use std::collections::HashMap;

use litsea::{BinaryMetrics, MulticlassMetrics, TwoStageMetrics};
use pyo3::prelude::*;

/// Metrics from training a binary (segmentation) model.
///
/// All percentages are 0-100.
#[pyclass(name = "BinaryMetrics", frozen, skip_from_py_object, module = "litsea")]
#[derive(Debug, Clone)]
pub struct PyBinaryMetrics {
    /// Accuracy, as a percentage.
    #[pyo3(get)]
    accuracy: f64,
    /// Precision, as a percentage.
    #[pyo3(get)]
    precision: f64,
    /// Recall, as a percentage.
    #[pyo3(get)]
    recall: f64,
    /// Number of training instances.
    #[pyo3(get)]
    num_instances: usize,
    /// True positives.
    #[pyo3(get)]
    true_positives: usize,
    /// False positives.
    #[pyo3(get)]
    false_positives: usize,
    /// False negatives.
    #[pyo3(get)]
    false_negatives: usize,
    /// True negatives.
    #[pyo3(get)]
    true_negatives: usize,
}

#[pymethods]
impl PyBinaryMetrics {
    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `BinaryMetrics(accuracy=99.12%, instances=1234)`.
    fn __repr__(&self) -> String {
        format!(
            "BinaryMetrics(accuracy={:.2}%, precision={:.2}%, recall={:.2}%, instances={})",
            self.accuracy, self.precision, self.recall, self.num_instances
        )
    }
}

impl From<BinaryMetrics> for PyBinaryMetrics {
    /// Converts `litsea`'s metrics into the Python-facing class.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PyBinaryMetrics`].
    fn from(metrics: BinaryMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            num_instances: metrics.num_instances,
            true_positives: metrics.true_positives,
            false_positives: metrics.false_positives,
            false_negatives: metrics.false_negatives,
            true_negatives: metrics.true_negatives,
        }
    }
}

/// Metrics from training a multiclass model.
///
/// All percentages are 0-100. The per-class counts are dictionaries keyed by
/// the label name.
#[pyclass(
    name = "MulticlassMetrics",
    frozen,
    skip_from_py_object,
    module = "litsea"
)]
#[derive(Debug, Clone)]
pub struct PyMulticlassMetrics {
    /// Accuracy, as a percentage.
    #[pyo3(get)]
    accuracy: f64,
    /// Macro-averaged precision, as a percentage.
    #[pyo3(get)]
    macro_precision: f64,
    /// Macro-averaged recall, as a percentage.
    #[pyo3(get)]
    macro_recall: f64,
    /// Number of training instances.
    #[pyo3(get)]
    num_instances: usize,
    /// Correct predictions per class.
    #[pyo3(get)]
    correct_per_class: HashMap<String, usize>,
    /// Predictions made per class.
    #[pyo3(get)]
    predicted_per_class: HashMap<String, usize>,
    /// Gold instances per class.
    #[pyo3(get)]
    gold_per_class: HashMap<String, usize>,
}

#[pymethods]
impl PyMulticlassMetrics {
    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `MulticlassMetrics(accuracy=96.78%, instances=1234)`.
    fn __repr__(&self) -> String {
        format!(
            "MulticlassMetrics(accuracy={:.2}%, macro_precision={:.2}%, macro_recall={:.2}%, instances={})",
            self.accuracy, self.macro_precision, self.macro_recall, self.num_instances
        )
    }
}

impl From<MulticlassMetrics> for PyMulticlassMetrics {
    /// Converts `litsea`'s metrics into the Python-facing class.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PyMulticlassMetrics`].
    fn from(metrics: MulticlassMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            macro_precision: metrics.macro_precision,
            macro_recall: metrics.macro_recall,
            num_instances: metrics.num_instances,
            correct_per_class: metrics.correct_per_class,
            predicted_per_class: metrics.predicted_per_class,
            gold_per_class: metrics.gold_per_class,
        }
    }
}

/// Metrics from training a two-stage model: one set per stage.
#[pyclass(
    name = "TwoStageMetrics",
    frozen,
    skip_from_py_object,
    module = "litsea"
)]
#[derive(Debug, Clone)]
pub struct PyTwoStageMetrics {
    /// Boundary-classifier (stage 1) metrics.
    #[pyo3(get)]
    stage1: PyMulticlassMetrics,
    /// Word-level tagger (stage 2) metrics.
    #[pyo3(get)]
    stage2: PyMulticlassMetrics,
}

#[pymethods]
impl PyTwoStageMetrics {
    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example
    /// `TwoStageMetrics(stage1=99.12%, stage2=96.78%)`.
    fn __repr__(&self) -> String {
        format!(
            "TwoStageMetrics(stage1={:.2}%, stage2={:.2}%)",
            self.stage1.accuracy, self.stage2.accuracy
        )
    }
}

impl From<TwoStageMetrics> for PyTwoStageMetrics {
    /// Converts `litsea`'s metrics into the Python-facing class.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PyTwoStageMetrics`].
    fn from(metrics: TwoStageMetrics) -> Self {
        Self {
            stage1: PyMulticlassMetrics::from(metrics.stage1),
            stage2: PyMulticlassMetrics::from(metrics.stage2),
        }
    }
}
