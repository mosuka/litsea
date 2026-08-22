//! Training metric classes.

use ext_php_rs::prelude::*;
use litsea::{BinaryMetrics, MulticlassMetrics, TwoStageMetrics};

/// Metrics from training a binary (segmentation) model.
///
/// All percentages are 0-100.
#[php_class]
#[php(name = "Litsea\\BinaryMetrics")]
#[derive(Default)]
pub struct PhpBinaryMetrics {
    /// Accuracy, as a percentage.
    #[php(prop)]
    pub accuracy: f64,
    /// Precision, as a percentage.
    #[php(prop)]
    pub precision: f64,
    /// Recall, as a percentage.
    #[php(prop)]
    pub recall: f64,
    /// Number of training instances.
    #[php(prop)]
    pub num_instances: u64,
    /// True positives.
    #[php(prop)]
    pub true_positives: u64,
    /// False positives.
    #[php(prop)]
    pub false_positives: u64,
    /// False negatives.
    #[php(prop)]
    pub false_negatives: u64,
    /// True negatives.
    #[php(prop)]
    pub true_negatives: u64,
}

impl From<BinaryMetrics> for PhpBinaryMetrics {
    /// Converts `litsea`'s metrics into the PHP-facing class.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PhpBinaryMetrics`].
    fn from(metrics: BinaryMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            num_instances: metrics.num_instances as u64,
            true_positives: metrics.true_positives as u64,
            false_positives: metrics.false_positives as u64,
            false_negatives: metrics.false_negatives as u64,
            true_negatives: metrics.true_negatives as u64,
        }
    }
}

/// Metrics from training a multiclass model.
///
/// All percentages are 0-100.
#[php_class]
#[php(name = "Litsea\\MulticlassMetrics")]
#[derive(Default)]
pub struct PhpMulticlassMetrics {
    /// Accuracy, as a percentage.
    #[php(prop)]
    pub accuracy: f64,
    /// Macro-averaged precision, as a percentage.
    #[php(prop)]
    pub macro_precision: f64,
    /// Macro-averaged recall, as a percentage.
    #[php(prop)]
    pub macro_recall: f64,
    /// Number of training instances.
    #[php(prop)]
    pub num_instances: u64,
}

impl From<MulticlassMetrics> for PhpMulticlassMetrics {
    /// Converts `litsea`'s metrics into the PHP-facing class.
    ///
    /// The per-class count maps are not exposed: PHP properties must be a
    /// fixed type, and the maps are only useful for detailed analysis, which
    /// the CLI already prints.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PhpMulticlassMetrics`].
    fn from(metrics: MulticlassMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            macro_precision: metrics.macro_precision,
            macro_recall: metrics.macro_recall,
            num_instances: metrics.num_instances as u64,
        }
    }
}

/// Metrics from training a two-stage model: one set per stage.
#[php_class]
#[php(name = "Litsea\\TwoStageMetrics")]
#[derive(Default)]
pub struct PhpTwoStageMetrics {
    /// Stage-1 (boundary classifier) accuracy, as a percentage.
    #[php(prop)]
    pub stage1_accuracy: f64,
    /// Stage-1 training instances.
    #[php(prop)]
    pub stage1_num_instances: u64,
    /// Stage-2 (word tagger) accuracy, as a percentage.
    #[php(prop)]
    pub stage2_accuracy: f64,
    /// Stage-2 training instances.
    #[php(prop)]
    pub stage2_num_instances: u64,
}

impl From<TwoStageMetrics> for PhpTwoStageMetrics {
    /// Converts `litsea`'s metrics into the PHP-facing class.
    ///
    /// The two stages are flattened into scalar properties rather than
    /// nested objects, which keeps the class a plain data holder.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`PhpTwoStageMetrics`].
    fn from(metrics: TwoStageMetrics) -> Self {
        Self {
            stage1_accuracy: metrics.stage1.accuracy,
            stage1_num_instances: metrics.stage1.num_instances as u64,
            stage2_accuracy: metrics.stage2.accuracy,
            stage2_num_instances: metrics.stage2.num_instances as u64,
        }
    }
}
