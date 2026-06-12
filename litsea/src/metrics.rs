//! Evaluation metrics for the learners.

use std::collections::HashMap;

/// Evaluation metrics for binary classification ([`crate::adaboost::AdaBoost`]).
#[derive(Debug, Clone)]
pub struct BinaryMetrics {
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

/// Evaluation metrics for multiclass classification
/// ([`crate::perceptron::AveragedPerceptron`]), using macro averages.
#[derive(Debug, Clone)]
pub struct MulticlassMetrics {
    /// Accuracy in percentage (%)
    pub accuracy: f64,
    /// Macro-averaged precision in percentage (%)
    pub macro_precision: f64,
    /// Macro-averaged recall in percentage (%)
    pub macro_recall: f64,
    /// Number of instances in the dataset
    pub num_instances: usize,
    /// Correct predictions per class
    pub correct_per_class: HashMap<String, usize>,
    /// Predicted count per class
    pub predicted_per_class: HashMap<String, usize>,
    /// Gold (true) label count per class
    pub gold_per_class: HashMap<String, usize>,
}
