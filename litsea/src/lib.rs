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
pub mod extractor;
pub mod language;
pub mod perceptron;
pub mod segmenter;
pub mod trainer;
pub mod upos;
pub mod util;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
