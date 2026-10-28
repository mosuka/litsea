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

use litsea::{AdaBoost, Language, ModelKind, Segmenter, TwoStageLearner, Upos, UserDictionary};

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
    build_segmenter_with(language, bytes, None)
}

/// Builds a [`Segmenter`] from raw model bytes with an optional user
/// dictionary (issue #263); [`build_segmenter`] is this with `None`.
///
/// # Arguments
/// * `language` - The language the model was trained for.
/// * `bytes` - The raw model file contents.
/// * `dictionary` - Words to emit as single tokens (see
///   [`build_user_dictionary`]), or `None` to segment with the model alone.
///
/// # Returns
/// The built segmenter, with the dictionary attached, and whether it
/// supports POS tagging.
///
/// # Errors
/// The same errors as [`build_segmenter`].
pub fn build_segmenter_with(
    language: Language,
    bytes: &[u8],
    dictionary: Option<UserDictionary>,
) -> CoreResult<BuiltSegmenter> {
    // `ModelKind::detect` only inspects the first line, so decoding just
    // that avoids validating megabytes of UTF-8 twice (the learner's parser
    // does its own decoding).
    let head_end = bytes.iter().position(|&b| b == b'\n').unwrap_or(bytes.len());
    let head = std::str::from_utf8(&bytes[..head_end]).map_err(|e| {
        CoreError::new(ErrorKind::Parse, format!("model is not valid UTF-8: {}", e))
    })?;

    let (mut segmenter, has_pos) = match ModelKind::detect(head) {
        ModelKind::TwoStage => {
            let mut learner = TwoStageLearner::new();
            learner.load_model_from_reader(bytes)?;
            (Segmenter::with_two_stage_learner(language, learner), true)
        }
        ModelKind::AdaBoost => {
            let mut learner = AdaBoost::default();
            learner.load_model_from_reader(bytes)?;
            (Segmenter::with_learner(language, learner), false)
        }
        ModelKind::AveragedPerceptron => {
            return Err(CoreError::model(
                "this file is a joint POS (Averaged Perceptron) model; joint POS models are no \
                 longer supported - retrain with `litsea train --pos`",
            ));
        }
    };
    segmenter.set_user_dictionary(dictionary);
    Ok(BuiltSegmenter { segmenter, has_pos })
}

/// Builds the user dictionary a binding received as a word list and an
/// optional gate (issue #263); [`build_user_dictionary_with_tags`] is this
/// with tagged words as well.
///
/// # Arguments
/// * `words` - The words to register; words of one character or less are
///   ignored (see [`UserDictionary::add`]).
/// * `gate` - The flip-ratio gate in `0.0..=1.0`, or `None` to apply every
///   occurrence (see [`UserDictionary::set_gate`]).
///
/// # Returns
/// `Ok(None)` when `words` is empty (nothing to apply, so the segmenter's
/// hot path stays untouched); otherwise the dictionary with the gate set.
///
/// # Errors
/// Returns an [`ErrorKind::InvalidArgument`] error if `gate` is outside
/// `0.0..=1.0` or NaN - also when `words` is empty, so a bad gate is never
/// silently ignored.
pub fn build_user_dictionary<S: AsRef<str>>(
    words: &[S],
    gate: Option<f64>,
) -> CoreResult<Option<UserDictionary>> {
    build_user_dictionary_with_tags::<S, &str, &str>(words, &[], gate)
}

/// Builds the user dictionary a binding received as a word list, a
/// word-to-tag mapping and an optional gate (issues #263, #264, #276) -
/// the one place the five bindings share.
///
/// # Arguments
/// * `words` - The untagged words to register; words of one character or
///   less are ignored (see [`UserDictionary::add`]).
/// * `word_tags` - `(word, UPOS tag)` pairs to register with their tag
///   (see [`UserDictionary::add_with_pos`]); a word that is also in
///   `words` keeps the tag. A word of one character becomes a tag-only
///   entry (issue #276) that changes no boundary and retags only the
///   tokens that are exactly that character. The pairs are applied in
///   sorted order, so neither the result nor the reported error depends
///   on the order a host's map iterates in.
/// * `gate` - The flip-ratio gate in `0.0..=1.0`, or `None` to apply every
///   occurrence (see [`UserDictionary::set_gate`]).
///
/// # Returns
/// `Ok(None)` when both `words` and `word_tags` are empty (nothing to
/// apply, so the segmenter's hot path stays untouched); otherwise the
/// dictionary with the gate set.
///
/// # Errors
/// Returns an [`ErrorKind::InvalidArgument`] error if a tag is not one of
/// the 17 UPOS tags, or if
/// `gate` is outside `0.0..=1.0` or NaN - also when no word is given, so a
/// bad gate is never silently ignored.
pub fn build_user_dictionary_with_tags<W, K, T>(
    words: &[W],
    word_tags: &[(K, T)],
    gate: Option<f64>,
) -> CoreResult<Option<UserDictionary>>
where
    W: AsRef<str>,
    K: AsRef<str>,
    T: AsRef<str>,
{
    let mut dictionary = UserDictionary::from_words(words);
    let mut tagged: Vec<(&str, &str)> =
        word_tags.iter().map(|(word, tag)| (word.as_ref(), tag.as_ref())).collect();
    tagged.sort_unstable();
    for (word, tag) in tagged {
        let pos: Upos = tag.parse().map_err(|_| {
            let expected: Vec<String> = Upos::ALL.iter().map(Upos::to_string).collect();
            CoreError::invalid_argument(format!(
                "unknown UPOS tag {tag:?} for the user dictionary word {word:?} (expected one \
                 of {})",
                expected.join(", ")
            ))
        })?;
        dictionary.add_with_pos(word, pos);
    }
    dictionary.set_gate(gate)?;
    Ok(if words.is_empty() && word_tags.is_empty() { None } else { Some(dictionary) })
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

    #[test]
    fn test_build_user_dictionary() {
        // #263: an empty word list is "no dictionary".
        assert!(build_user_dictionary::<&str>(&[], None).unwrap().is_none());
        let dictionary = build_user_dictionary(&["雪だるま", "あ"], Some(0.5)).unwrap().unwrap();
        assert_eq!(dictionary.len(), 1, "one-character words are dropped");
        assert_eq!(dictionary.gate(), Some(0.5));
        // A bad gate is an InvalidArgument error, with or without words.
        for words in [vec![], vec!["雪だるま"]] {
            let error = build_user_dictionary(&words, Some(1.5)).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument, "{words:?}");
            assert!(error.message().contains("0.0..=1.0"), "{}", error.message());
        }
    }

    #[test]
    fn test_build_user_dictionary_with_tags() {
        // #264: tagged words alone make a dictionary; nothing at all is
        // still "no dictionary".
        let none: [(&str, &str); 0] = [];
        assert!(
            build_user_dictionary_with_tags::<&str, _, _>(&[], &none, None)
                .unwrap()
                .is_none()
        );
        let dictionary = build_user_dictionary_with_tags::<&str, _, _>(
            &[],
            &[("チャットGPT", "PROPN")],
            Some(0.5),
        )
        .unwrap()
        .unwrap();
        assert_eq!(dictionary.pos("チャットGPT"), Some(litsea::Upos::PROPN));
        assert_eq!(dictionary.gate(), Some(0.5));
        // A word in both lists keeps its tag; a tagged one-character word
        // is a tag-only entry (#276), an untagged one is dropped.
        let dictionary = build_user_dictionary_with_tags(
            &["雪だるま", "新幹線", "犬"],
            &[("雪だるま", "NOUN"), ("猫", "NOUN")],
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(dictionary.len(), 3);
        assert_eq!(dictionary.pos("雪だるま"), Some(litsea::Upos::NOUN));
        assert_eq!(dictionary.pos("新幹線"), None);
        assert!(dictionary.contains("猫"));
        assert_eq!(dictionary.pos("猫"), Some(litsea::Upos::NOUN));
        assert!(!dictionary.contains("犬"));
    }

    #[test]
    fn test_build_user_dictionary_with_tags_errors() {
        // A tag that is not UPOS is an InvalidArgument error naming the tag
        // and the word - also for a one-character word. The pairs are
        // checked in sorted order, so the first bad pair in that order is
        // reported whatever order the caller passed them in.
        for word_tags in [
            vec![("ポケモン", "Noun"), ("猫", "CAT")],
            vec![("猫", "CAT"), ("ポケモン", "Noun")],
        ] {
            let error =
                build_user_dictionary_with_tags::<&str, _, _>(&[], &word_tags, None).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidArgument);
            assert!(
                error.message().contains("unknown UPOS tag \"Noun\"")
                    && error.message().contains("\"ポケモン\"")
                    && error.message().contains("expected one of ADJ, ADP"),
                "{}",
                error.message()
            );
        }
        let error =
            build_user_dictionary_with_tags::<&str, _, _>(&[], &[("猫", "CAT")], None).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert!(error.message().contains("\"CAT\""), "{}", error.message());
        // A bad gate is still an error when only tagged words are given.
        let error = build_user_dictionary_with_tags::<&str, _, _>(
            &[],
            &[("チャットGPT", "PROPN")],
            Some(1.5),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert!(error.message().contains("0.0..=1.0"), "{}", error.message());
    }

    #[test]
    fn test_build_segmenter_with_attaches_the_dictionary() {
        let bytes = read_model_file(&model_path("japanese.model")).unwrap();
        let dictionary = build_user_dictionary(&["雪だるま"], None).unwrap();
        let built = build_segmenter_with(Language::Japanese, &bytes, dictionary).unwrap();
        assert!(!built.has_pos);
        assert!(built.segmenter.user_dictionary().is_some());
        assert_eq!(built.segmenter.segment("雪だるまを作る")[0], "雪だるま");
        // The plain builder attaches nothing.
        let plain = build_segmenter(Language::Japanese, &bytes).unwrap();
        assert!(plain.segmenter.user_dictionary().is_none());
        assert_eq!(plain.segmenter.segment("雪だるまを作る")[0], "雪");
        // The two-stage arm attaches it too.
        let bytes = read_model_file(&model_path("japanese_pos.model")).unwrap();
        let dictionary = build_user_dictionary(&["雪だるま"], None).unwrap();
        let built = build_segmenter_with(Language::Japanese, &bytes, dictionary).unwrap();
        assert!(built.has_pos);
        assert_eq!(built.segmenter.segment_with_pos("雪だるまを作る").unwrap()[0].0, "雪だるま");
    }
}
