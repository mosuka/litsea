//! Litsea is an extremely compact word segmentation and POS tagging library implemented in Rust.
//!
//! It performs word segmentation using a compact pre-trained model based on AdaBoost
//! binary classification, inspired by TinySegmenter and TinySegmenterMaker.
//! It also supports joint word segmentation and POS (Part-of-Speech) tagging
//! using an Averaged Perceptron with Universal POS (UPOS) tags.
//!
//! # Supported Languages
//! - Japanese
//! - Chinese (Simplified and Traditional)
//! - Korean

pub mod adaboost;
pub mod error;
pub mod evaluation;
pub mod extractor;
pub mod language;
pub mod metrics;
mod model_io;
mod packed_model;
mod packed_pos_model;
mod packed_two_stage;
pub mod perceptron;
pub mod segmenter;
pub mod trainer;
pub mod two_stage;
pub mod upos;
mod word_features;

pub use adaboost::AdaBoost;
pub use error::{LitseaError, Result};
pub use evaluation::{PosMetrics, SegmentationMetrics};
pub use extractor::Extractor;
pub use language::{Language, ParseLanguageError};
pub use metrics::{BinaryMetrics, MulticlassMetrics};
pub use perceptron::AveragedPerceptron;
pub use segmenter::Segmenter;
pub use trainer::{PosTrainer, Trainer, TwoStageMetrics, TwoStageTrainer};
pub use two_stage::{
    AnyPosModel, ModelKind, ParseTwoStageFeatureSetError, TwoStageFeatureSet, TwoStageLearner,
};
pub use upos::{ParseSegmentLabelError, ParseUposError, SegmentLabel, Upos};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the version of the litsea crate (the `CARGO_PKG_VERSION` it was
/// built with), e.g. `"0.6.0"`.
#[must_use]
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
        assert_eq!(v, env!("CARGO_PKG_VERSION"));
    }
}
