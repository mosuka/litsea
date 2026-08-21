//! Litsea is an extremely compact word segmentation and POS tagging library implemented in Rust.
//!
//! It performs word segmentation using a compact pre-trained model in the
//! [`AdaBoost`] text format, inspired by TinySegmenter and
//! TinySegmenterMaker; the bundled segmentation models are trained via a
//! lossless Averaged Perceptron collapse rather than AdaBoost boosting (see
//! the `litsea::trainer` module docs), but the format and inference path are
//! unchanged. It also supports word segmentation and POS (Part-of-Speech)
//! tagging with Universal POS (UPOS) tags through a two-stage architecture:
//! a binary boundary classifier plus a word-level tagger (see the
//! [`two_stage`] module).
//!
//! # Supported Languages
//! - Japanese
//! - Chinese (Simplified and Traditional)
//! - Korean
//! - English

pub mod adaboost;
pub mod error;
pub mod evaluation;
pub mod extractor;
pub mod language;
pub mod metrics;
pub mod model_io;
mod packed_model;
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
pub use segmenter::{SegmentBuffer, Segmenter};
pub use trainer::{PerceptronTrainer, Trainer, TwoStageMetrics, TwoStageTrainer};
pub use two_stage::{ModelKind, ParseTwoStageFeatureSetError, TwoStageFeatureSet, TwoStageLearner};
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
