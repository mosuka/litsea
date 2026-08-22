//! Training metric classes.

use litsea::{BinaryMetrics, MulticlassMetrics, TwoStageMetrics};
use magnus::{Module, RModule, Ruby, error::Error};

/// Metrics from training a binary (segmentation) model.
///
/// All percentages are 0-100.
#[magnus::wrap(class = "Litsea::BinaryMetrics", free_immediately, size)]
pub struct RbBinaryMetrics {
    /// The wrapped metrics.
    inner: BinaryMetrics,
}

impl RbBinaryMetrics {
    /// Accuracy, as a percentage.
    fn accuracy(&self) -> f64 {
        self.inner.accuracy
    }

    /// Precision, as a percentage.
    fn precision(&self) -> f64 {
        self.inner.precision
    }

    /// Recall, as a percentage.
    fn recall(&self) -> f64 {
        self.inner.recall
    }

    /// Number of training instances.
    fn num_instances(&self) -> usize {
        self.inner.num_instances
    }

    /// True positives.
    fn true_positives(&self) -> usize {
        self.inner.true_positives
    }

    /// False positives.
    fn false_positives(&self) -> usize {
        self.inner.false_positives
    }

    /// False negatives.
    fn false_negatives(&self) -> usize {
        self.inner.false_negatives
    }

    /// True negatives.
    fn true_negatives(&self) -> usize {
        self.inner.true_negatives
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `#<Litsea::BinaryMetrics accuracy=99.12% instances=1234>`.
    fn inspect(&self) -> String {
        format!(
            "#<Litsea::BinaryMetrics accuracy={:.2}% instances={}>",
            self.inner.accuracy, self.inner.num_instances
        )
    }
}

impl From<BinaryMetrics> for RbBinaryMetrics {
    /// Wraps `litsea`'s metrics for Ruby.
    ///
    /// # Arguments
    /// * `inner` - The metrics to wrap.
    ///
    /// # Returns
    /// The corresponding [`RbBinaryMetrics`].
    fn from(inner: BinaryMetrics) -> Self {
        Self { inner }
    }
}

/// Metrics from training a multiclass model.
///
/// All percentages are 0-100.
#[magnus::wrap(class = "Litsea::MulticlassMetrics", free_immediately, size)]
pub struct RbMulticlassMetrics {
    /// The wrapped metrics.
    inner: MulticlassMetrics,
}

impl RbMulticlassMetrics {
    /// Accuracy, as a percentage.
    fn accuracy(&self) -> f64 {
        self.inner.accuracy
    }

    /// Macro-averaged precision, as a percentage.
    fn macro_precision(&self) -> f64 {
        self.inner.macro_precision
    }

    /// Macro-averaged recall, as a percentage.
    fn macro_recall(&self) -> f64 {
        self.inner.macro_recall
    }

    /// Number of training instances.
    fn num_instances(&self) -> usize {
        self.inner.num_instances
    }

    /// Gold instances per class, as a Hash keyed by the label name.
    fn gold_per_class(&self) -> std::collections::HashMap<String, usize> {
        self.inner.gold_per_class.clone()
    }

    /// Correct predictions per class.
    fn correct_per_class(&self) -> std::collections::HashMap<String, usize> {
        self.inner.correct_per_class.clone()
    }

    /// Predictions made per class.
    fn predicted_per_class(&self) -> std::collections::HashMap<String, usize> {
        self.inner.predicted_per_class.clone()
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `#<Litsea::MulticlassMetrics accuracy=97.30% instances=999>`.
    fn inspect(&self) -> String {
        format!(
            "#<Litsea::MulticlassMetrics accuracy={:.2}% instances={}>",
            self.inner.accuracy, self.inner.num_instances
        )
    }
}

impl From<MulticlassMetrics> for RbMulticlassMetrics {
    /// Wraps `litsea`'s metrics for Ruby.
    ///
    /// # Arguments
    /// * `inner` - The metrics to wrap.
    ///
    /// # Returns
    /// The corresponding [`RbMulticlassMetrics`].
    fn from(inner: MulticlassMetrics) -> Self {
        Self { inner }
    }
}

/// Metrics from training a two-stage model: one set per stage.
#[magnus::wrap(class = "Litsea::TwoStageMetrics", free_immediately, size)]
pub struct RbTwoStageMetrics {
    /// The wrapped metrics.
    inner: TwoStageMetrics,
}

impl RbTwoStageMetrics {
    /// Stage-1 (boundary classifier) metrics.
    fn stage1(&self) -> RbMulticlassMetrics {
        RbMulticlassMetrics::from(self.inner.stage1.clone())
    }

    /// Stage-2 (word tagger) metrics.
    fn stage2(&self) -> RbMulticlassMetrics {
        RbMulticlassMetrics::from(self.inner.stage2.clone())
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `#<Litsea::TwoStageMetrics stage1=99.10% stage2=95.40%>`.
    fn inspect(&self) -> String {
        format!(
            "#<Litsea::TwoStageMetrics stage1={:.2}% stage2={:.2}%>",
            self.inner.stage1.accuracy, self.inner.stage2.accuracy
        )
    }
}

impl From<TwoStageMetrics> for RbTwoStageMetrics {
    /// Wraps `litsea`'s metrics for Ruby.
    ///
    /// # Arguments
    /// * `inner` - The metrics to wrap.
    ///
    /// # Returns
    /// The corresponding [`RbTwoStageMetrics`].
    fn from(inner: TwoStageMetrics) -> Self {
        Self { inner }
    }
}

/// Defines the metric classes.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `module` - The `Litsea` module to define the classes on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a Ruby exception if a class cannot be defined.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let binary = module.define_class("BinaryMetrics", ruby.class_object())?;
    binary.define_method("accuracy", magnus::method!(RbBinaryMetrics::accuracy, 0))?;
    binary.define_method("precision", magnus::method!(RbBinaryMetrics::precision, 0))?;
    binary.define_method("recall", magnus::method!(RbBinaryMetrics::recall, 0))?;
    binary.define_method("num_instances", magnus::method!(RbBinaryMetrics::num_instances, 0))?;
    binary.define_method("true_positives", magnus::method!(RbBinaryMetrics::true_positives, 0))?;
    binary
        .define_method("false_positives", magnus::method!(RbBinaryMetrics::false_positives, 0))?;
    binary
        .define_method("false_negatives", magnus::method!(RbBinaryMetrics::false_negatives, 0))?;
    binary.define_method("true_negatives", magnus::method!(RbBinaryMetrics::true_negatives, 0))?;
    binary.define_method("inspect", magnus::method!(RbBinaryMetrics::inspect, 0))?;
    binary.define_method("to_s", magnus::method!(RbBinaryMetrics::inspect, 0))?;

    let multiclass = module.define_class("MulticlassMetrics", ruby.class_object())?;
    multiclass.define_method("accuracy", magnus::method!(RbMulticlassMetrics::accuracy, 0))?;
    multiclass.define_method(
        "macro_precision",
        magnus::method!(RbMulticlassMetrics::macro_precision, 0),
    )?;
    multiclass
        .define_method("macro_recall", magnus::method!(RbMulticlassMetrics::macro_recall, 0))?;
    multiclass
        .define_method("num_instances", magnus::method!(RbMulticlassMetrics::num_instances, 0))?;
    multiclass
        .define_method("gold_per_class", magnus::method!(RbMulticlassMetrics::gold_per_class, 0))?;
    multiclass.define_method(
        "correct_per_class",
        magnus::method!(RbMulticlassMetrics::correct_per_class, 0),
    )?;
    multiclass.define_method(
        "predicted_per_class",
        magnus::method!(RbMulticlassMetrics::predicted_per_class, 0),
    )?;
    multiclass.define_method("inspect", magnus::method!(RbMulticlassMetrics::inspect, 0))?;
    multiclass.define_method("to_s", magnus::method!(RbMulticlassMetrics::inspect, 0))?;

    let two_stage = module.define_class("TwoStageMetrics", ruby.class_object())?;
    two_stage.define_method("stage1", magnus::method!(RbTwoStageMetrics::stage1, 0))?;
    two_stage.define_method("stage2", magnus::method!(RbTwoStageMetrics::stage2, 0))?;
    two_stage.define_method("inspect", magnus::method!(RbTwoStageMetrics::inspect, 0))?;
    two_stage.define_method("to_s", magnus::method!(RbTwoStageMetrics::inspect, 0))?;

    Ok(())
}
