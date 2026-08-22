//! Training metric objects.

use std::collections::HashMap;

use litsea::{BinaryMetrics, MulticlassMetrics, TwoStageMetrics};

/// Metrics from training a binary (segmentation) model.
///
/// All percentages are 0-100.
#[napi(object)]
pub struct JsBinaryMetrics {
    /// Accuracy, as a percentage.
    pub accuracy: f64,
    /// Precision, as a percentage.
    pub precision: f64,
    /// Recall, as a percentage.
    pub recall: f64,
    /// Number of training instances.
    pub num_instances: u32,
    /// True positives.
    pub true_positives: u32,
    /// False positives.
    pub false_positives: u32,
    /// False negatives.
    pub false_negatives: u32,
    /// True negatives.
    pub true_negatives: u32,
}

impl From<BinaryMetrics> for JsBinaryMetrics {
    /// Converts `litsea`'s metrics into the JavaScript-facing object.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`JsBinaryMetrics`].
    fn from(metrics: BinaryMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            num_instances: metrics.num_instances as u32,
            true_positives: metrics.true_positives as u32,
            false_positives: metrics.false_positives as u32,
            false_negatives: metrics.false_negatives as u32,
            true_negatives: metrics.true_negatives as u32,
        }
    }
}

/// Metrics from training a multiclass model.
///
/// All percentages are 0-100. The per-class counts are objects keyed by the
/// label name.
#[napi(object)]
pub struct JsMulticlassMetrics {
    /// Accuracy, as a percentage.
    pub accuracy: f64,
    /// Macro-averaged precision, as a percentage.
    pub macro_precision: f64,
    /// Macro-averaged recall, as a percentage.
    pub macro_recall: f64,
    /// Number of training instances.
    pub num_instances: u32,
    /// Correct predictions per class.
    pub correct_per_class: HashMap<String, u32>,
    /// Predictions made per class.
    pub predicted_per_class: HashMap<String, u32>,
    /// Gold instances per class.
    pub gold_per_class: HashMap<String, u32>,
}

impl From<MulticlassMetrics> for JsMulticlassMetrics {
    /// Converts `litsea`'s metrics into the JavaScript-facing object.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`JsMulticlassMetrics`].
    fn from(metrics: MulticlassMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            macro_precision: metrics.macro_precision,
            macro_recall: metrics.macro_recall,
            num_instances: metrics.num_instances as u32,
            correct_per_class: to_u32_map(metrics.correct_per_class),
            predicted_per_class: to_u32_map(metrics.predicted_per_class),
            gold_per_class: to_u32_map(metrics.gold_per_class),
        }
    }
}

/// Narrows a per-class count map to the width N-API exposes.
///
/// # Arguments
/// * `map` - The per-class counts.
///
/// # Returns
/// The same map with `u32` values.
fn to_u32_map(map: HashMap<String, usize>) -> HashMap<String, u32> {
    map.into_iter().map(|(class, count)| (class, count as u32)).collect()
}

/// Metrics from training a two-stage model: one set per stage.
#[napi(object)]
pub struct JsTwoStageMetrics {
    /// Boundary-classifier (stage 1) metrics.
    pub stage1: JsMulticlassMetrics,
    /// Word-level tagger (stage 2) metrics.
    pub stage2: JsMulticlassMetrics,
}

impl From<TwoStageMetrics> for JsTwoStageMetrics {
    /// Converts `litsea`'s metrics into the JavaScript-facing object.
    ///
    /// # Arguments
    /// * `metrics` - The metrics to convert.
    ///
    /// # Returns
    /// The corresponding [`JsTwoStageMetrics`].
    fn from(metrics: TwoStageMetrics) -> Self {
        Self {
            stage1: JsMulticlassMetrics::from(metrics.stage1),
            stage2: JsMulticlassMetrics::from(metrics.stage2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_u32_map() {
        let mut map = HashMap::new();
        map.insert("NOUN".to_string(), 42usize);
        let converted = to_u32_map(map);
        assert_eq!(converted.get("NOUN"), Some(&42u32));
    }
}
