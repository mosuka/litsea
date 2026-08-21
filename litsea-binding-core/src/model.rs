//! Model loading and model-kind dispatch.
//!
//! The CLI decides between a segmentation model and a two-stage POS model
//! from its `--pos` flag (`litsea-cli/src/main.rs`), which forces the caller
//! to know what kind of file they hold. Bindings can do better: this module
//! reads the model bytes once, dispatches on
//! [`ModelKind::detect`](litsea::ModelKind::detect), and builds the matching
//! [`Segmenter`]. That is what `ModelKind` is public for, and it means no
//! binding has to expose a "this is a POS model" flag.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use litsea::{AdaBoost, Language, ModelKind, Segmenter, TwoStageLearner};

use crate::error::{CoreError, CoreResult, ErrorKind};

/// A [`Segmenter`] built from model bytes, together with what the model can do.
#[derive(Debug)]
pub struct BuiltSegmenter {
    /// The segmenter, ready to use.
    pub segmenter: Segmenter,
    /// Whether the model supports POS tagging (a two-stage model).
    pub has_pos: bool,
}

/// Builds a [`Segmenter`] from raw model bytes, detecting the model kind.
///
/// A two-stage model produces a POS-capable segmenter; an AdaBoost-format
/// model produces a segmentation-only one. Legacy joint POS models (removed
/// in #190) are rejected with an actionable message.
///
/// # Arguments
/// * `language` - The language the model was trained for. Character type
///   codes are language-specific, so this must match the training language.
/// * `bytes` - The raw model file contents.
///
/// # Returns
/// The built segmenter and whether it supports POS tagging.
///
/// # Errors
/// Returns an [`ErrorKind::Parse`] error if the bytes are not valid UTF-8 or
/// the model content is malformed, or an [`ErrorKind::Model`] error if the
/// file is a legacy joint POS model.
pub fn build_segmenter(language: Language, bytes: &[u8]) -> CoreResult<BuiltSegmenter> {
    // `ModelKind::detect` only inspects the first line, so decoding just
    // that avoids validating megabytes of UTF-8 twice (the learner's parser
    // does its own decoding).
    let head_end = bytes.iter().position(|&b| b == b'\n').unwrap_or(bytes.len());
    let head = std::str::from_utf8(&bytes[..head_end]).map_err(|e| {
        CoreError::new(ErrorKind::Parse, format!("model is not valid UTF-8: {}", e))
    })?;

    match ModelKind::detect(head) {
        ModelKind::TwoStage => {
            let mut learner = TwoStageLearner::new();
            learner.load_model_from_reader(bytes)?;
            Ok(BuiltSegmenter {
                segmenter: Segmenter::with_two_stage_learner(language, learner),
                has_pos: true,
            })
        }
        ModelKind::AdaBoost => {
            let mut learner = AdaBoost::default();
            learner.load_model_from_reader(bytes)?;
            Ok(BuiltSegmenter {
                segmenter: Segmenter::with_learner(language, learner),
                has_pos: false,
            })
        }
        ModelKind::AveragedPerceptron => Err(CoreError::model(
            "this file is a joint POS (Averaged Perceptron) model; joint POS models are no \
             longer supported - retrain with `litsea train --pos`",
        )),
    }
}

/// Reads a model file from the local filesystem.
///
/// # Arguments
/// * `path` - Path to the model file.
///
/// # Returns
/// The raw model bytes.
///
/// # Errors
/// Returns an [`ErrorKind::Io`] error if the file cannot be read.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_model_file(path: &Path) -> CoreResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| {
        CoreError::new(
            ErrorKind::Io,
            format!("failed to read model file {}: {}", path.display(), e),
        )
    })
}

/// Reads model bytes from a URI.
///
/// Delegates to [`litsea::model_io::read_model_bytes`], so the accepted
/// forms are a filesystem path, `file://<path>`, and - with the
/// `remote_model` feature - `http(s)://` URLs. Reading the bytes here rather
/// than calling a learner's `load_model(uri)` means a remote model is
/// fetched once and then dispatched on its detected kind.
///
/// # Arguments
/// * `uri` - The model URI.
///
/// # Returns
/// The raw model bytes.
///
/// # Errors
/// Returns the mapped [`litsea::LitseaError`]: an unknown scheme becomes
/// [`ErrorKind::InvalidArgument`], an unavailable scheme
/// [`ErrorKind::Unsupported`], a failed download [`ErrorKind::Model`], and a
/// filesystem failure [`ErrorKind::Io`].
pub async fn read_model_uri(uri: &str) -> CoreResult<Vec<u8>> {
    litsea::model_io::read_model_bytes(uri).await.map_err(CoreError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Repository-root-relative path to a bundled model.
    fn model_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("models")
            .join(name)
    }

    #[test]
    fn test_build_segmentation_model() {
        let bytes = read_model_file(&model_path("japanese.model")).unwrap();
        let built = build_segmenter(Language::Japanese, &bytes).unwrap();
        assert!(!built.has_pos);
        assert_eq!(built.segmenter.language(), Language::Japanese);
        assert!(built.segmenter.segment_with_pos("test").is_err());
    }

    #[test]
    fn test_build_two_stage_model() {
        let bytes = read_model_file(&model_path("japanese_pos.model")).unwrap();
        let built = build_segmenter(Language::Japanese, &bytes).unwrap();
        assert!(built.has_pos);
        assert!(built.segmenter.segment_with_pos("すもも").is_ok());
    }

    #[test]
    fn test_joint_pos_model_is_rejected() {
        // A bare integer on the first line is the joint (Averaged
        // Perceptron) class-count header.
        let error = build_segmenter(Language::Japanese, b"17\nfoo\t1.0\n").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Model);
        assert!(
            error.message().contains("no longer supported"),
            "unexpected message: {}",
            error.message()
        );
    }

    #[test]
    fn test_invalid_utf8_is_a_parse_error() {
        let error = build_segmenter(Language::Japanese, &[0xff, 0xfe, b'\n']).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Parse);
    }

    #[test]
    fn test_empty_model_is_rejected() {
        let error = build_segmenter(Language::Japanese, b"").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Parse);
    }

    #[test]
    fn test_read_model_file_missing() {
        let error = read_model_file(std::path::Path::new("/nonexistent/model")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Io);
    }
}
