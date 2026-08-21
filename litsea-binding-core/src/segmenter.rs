//! The segmenter facade the bindings wrap.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use litsea::{Language, SegmentBuffer, Segmenter};

use crate::error::CoreResult;
use crate::model::{BuiltSegmenter, build_segmenter, read_model_uri};
use crate::token::TokenView;

/// A ready-to-use segmenter with a reusable scratch buffer.
///
/// Holds `Arc<Segmenter>` rather than a `Mutex<Segmenter>`: `Segmenter` is
/// `Send + Sync` (pinned by a compile-time assertion in `litsea`) and a
/// segmenter built from a loaded model has its packed tables already
/// compiled, so concurrent `segment` calls only take an internal read lock.
/// The one piece that genuinely needs exclusive access is the
/// [`SegmentBuffer`], so that alone sits behind a `Mutex` - which also keeps
/// this type `Send + Sync`, as PyO3's `#[pyclass]` and napi both require.
///
/// Reusing one instance across calls is the intended usage: the buffer
/// reaches a steady state where segmentation allocates only the output
/// strings.
#[derive(Debug)]
pub struct CoreSegmenter {
    /// The underlying segmenter, shareable across threads.
    segmenter: Arc<Segmenter>,
    /// Scratch storage reused by [`Segmenter::segment_into`].
    buffer: Mutex<SegmentBuffer>,
    /// Whether the loaded model supports POS tagging.
    has_pos: bool,
}

// `#[pyclass]` (PyO3) and napi class wrappers require `Send + Sync`; keep
// that a compile error rather than a surprise at the binding boundary.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CoreSegmenter>();
};

impl CoreSegmenter {
    /// Builds a segmenter from raw model bytes, detecting the model kind.
    ///
    /// # Arguments
    /// * `language` - The language the model was trained for.
    /// * `bytes` - The raw model file contents.
    ///
    /// # Returns
    /// The new [`CoreSegmenter`].
    ///
    /// # Errors
    /// Returns the error from [`build_segmenter`] if the bytes are not a
    /// supported model.
    pub fn from_bytes(language: Language, bytes: &[u8]) -> CoreResult<Self> {
        Ok(Self::from_built(build_segmenter(language, bytes)?))
    }

    /// Builds a segmenter from a model file on the local filesystem.
    ///
    /// # Arguments
    /// * `language` - The language the model was trained for.
    /// * `path` - Path to the model file.
    ///
    /// # Returns
    /// The new [`CoreSegmenter`].
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be read, or the error from
    /// [`build_segmenter`] if its contents are not a supported model.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_path(language: Language, path: &Path) -> CoreResult<Self> {
        let bytes = crate::model::read_model_file(path)?;
        Self::from_bytes(language, &bytes)
    }

    /// Builds a segmenter from a model URI.
    ///
    /// Accepts a filesystem path, `file://<path>`, or - with the
    /// `remote_model` feature - an `http(s)://` URL. The bytes are fetched
    /// once and then dispatched on the detected model kind.
    ///
    /// # Arguments
    /// * `language` - The language the model was trained for.
    /// * `uri` - The model URI.
    ///
    /// # Returns
    /// The new [`CoreSegmenter`].
    ///
    /// # Errors
    /// Returns an error if the URI cannot be resolved or its contents are
    /// not a supported model.
    pub async fn from_uri(language: Language, uri: &str) -> CoreResult<Self> {
        let bytes = read_model_uri(uri).await?;
        Self::from_bytes(language, &bytes)
    }

    /// Builds a segmenter from a model URI, blocking until it is loaded.
    ///
    /// For host languages without an event loop. Bindings that have one
    /// (Node.js, WASM) should await [`CoreSegmenter::from_uri`] instead.
    ///
    /// # Arguments
    /// * `language` - The language the model was trained for.
    /// * `uri` - The model URI.
    ///
    /// # Returns
    /// The new [`CoreSegmenter`].
    ///
    /// # Errors
    /// Returns an error if called from inside an async runtime, if the URI
    /// cannot be resolved, or if its contents are not a supported model.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_uri_blocking(language: Language, uri: &str) -> CoreResult<Self> {
        crate::runtime::block_on(Self::from_uri(language, uri))?
    }

    /// Wraps an already-built segmenter.
    ///
    /// # Arguments
    /// * `built` - The segmenter and its POS capability.
    ///
    /// # Returns
    /// The new [`CoreSegmenter`].
    fn from_built(built: BuiltSegmenter) -> Self {
        Self {
            segmenter: Arc::new(built.segmenter),
            buffer: Mutex::new(SegmentBuffer::new()),
            has_pos: built.has_pos,
        }
    }

    /// Returns the language this segmenter was built for.
    ///
    /// # Returns
    /// The [`Language`] passed at construction time.
    #[must_use]
    pub fn language(&self) -> Language {
        self.segmenter.language()
    }

    /// Returns whether POS tagging is available.
    ///
    /// # Returns
    /// `true` when the loaded model is a two-stage POS model.
    #[must_use]
    pub fn has_pos(&self) -> bool {
        self.has_pos
    }

    /// Returns the underlying segmenter.
    ///
    /// Bindings need this for the parts of the `litsea` API this facade does
    /// not wrap, such as `litsea::evaluation`.
    ///
    /// # Returns
    /// A shared reference to the wrapped [`Segmenter`].
    #[must_use]
    pub fn segmenter(&self) -> &Arc<Segmenter> {
        &self.segmenter
    }

    /// Splits a sentence into tokens.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, in order. An empty input yields an empty vector.
    #[must_use]
    pub fn segment(&self, text: &str) -> Vec<String> {
        self.with_buffer(|segmenter, buffer| {
            segmenter
                .segment_into(text, buffer)
                .iter()
                .map(|&(start, end)| text[start..end].to_string())
                .collect()
        })
    }

    /// Splits several sentences into tokens, reusing one scratch buffer.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// One token vector per input sentence, in input order.
    #[must_use]
    pub fn segment_batch<S: AsRef<str>>(&self, texts: &[S]) -> Vec<Vec<String>> {
        self.with_buffer(|segmenter, buffer| {
            texts
                .iter()
                .map(|text| {
                    let text = text.as_ref();
                    segmenter
                        .segment_into(text, buffer)
                        .iter()
                        .map(|&(start, end)| text[start..end].to_string())
                        .collect()
                })
                .collect()
        })
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens with `pos` set to `None`.
    #[must_use]
    pub fn segment_tokens(&self, text: &str) -> Vec<TokenView> {
        self.with_buffer(|segmenter, buffer| {
            segmenter
                .segment_into(text, buffer)
                .iter()
                .map(|&(start, end)| TokenView::new(&text[start..end], start, end, None))
                .collect()
        })
    }

    /// Splits a sentence into tokens and tags each with a UPOS tag.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment and tag.
    ///
    /// # Returns
    /// The tagged tokens, with byte offsets into `text`.
    ///
    /// # Errors
    /// Returns an [`crate::ErrorKind::PosUnavailable`] error when the
    /// segmenter was built from a segmentation-only model.
    pub fn segment_with_pos(&self, text: &str) -> CoreResult<Vec<TokenView>> {
        let tagged = self.segmenter.segment_with_pos(text)?;
        Ok(Self::attach_offsets(tagged))
    }

    /// Splits and tags several sentences.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment and tag.
    ///
    /// # Returns
    /// One tagged-token vector per input sentence, in input order.
    ///
    /// # Errors
    /// Returns an [`crate::ErrorKind::PosUnavailable`] error when the
    /// segmenter was built from a segmentation-only model.
    pub fn segment_with_pos_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> CoreResult<Vec<Vec<TokenView>>> {
        texts.iter().map(|text| self.segment_with_pos(text.as_ref())).collect()
    }

    /// Converts `litsea`'s tagged tokens into [`TokenView`]s with offsets.
    ///
    /// The tokens tile the input exactly - `segment_with_pos` tags the
    /// output of `segment`, whose ranges cover the sentence without gaps or
    /// overlaps - so offsets are the running sum of the surface lengths.
    /// `test_pos_offsets_slice_the_input` pins that invariant.
    ///
    /// # Arguments
    /// * `tagged` - Surface/tag pairs in input order.
    ///
    /// # Returns
    /// The tokens with byte offsets filled in.
    fn attach_offsets(tagged: Vec<(String, litsea::Upos)>) -> Vec<TokenView> {
        let mut offset = 0;
        tagged
            .into_iter()
            .map(|(surface, pos)| {
                let start = offset;
                offset += surface.len();
                TokenView::new(surface, start, offset, Some(pos))
            })
            .collect()
    }

    /// Runs `f` with the segmenter and the locked scratch buffer.
    ///
    /// A panic in a previous call may have poisoned the mutex; the buffer
    /// holds only scratch data that every call clears before use, so
    /// recovering from poisoning is safe.
    ///
    /// # Arguments
    /// * `f` - The closure to run.
    ///
    /// # Returns
    /// Whatever `f` returns.
    fn with_buffer<R>(&self, f: impl FnOnce(&Segmenter, &mut SegmentBuffer) -> R) -> R {
        let mut guard = self.buffer.lock().unwrap_or_else(PoisonError::into_inner);
        f(&self.segmenter, &mut guard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to a bundled model in the repository's `models/` directory.
    fn model_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .join("models")
            .join(name)
    }

    /// The bundled segmentation models and a sentence for each.
    fn segmentation_cases() -> Vec<(Language, &'static str, &'static str)> {
        vec![
            (Language::Japanese, "japanese.model", "すもももももももものうち"),
            (Language::Chinese, "chinese.model", "我喜欢吃中国菜。"),
            (Language::Korean, "korean.model", "안녕하세요 반갑습니다"),
            (
                Language::English,
                "english.model",
                "The quick brown fox jumps over the lazy dog.",
            ),
        ]
    }

    #[test]
    fn test_segment_matches_litsea_for_every_language() {
        for (language, model, sentence) in segmentation_cases() {
            let core = CoreSegmenter::from_path(language, &model_path(model)).unwrap();
            let expected = core.segmenter().segment(sentence);
            assert_eq!(
                core.segment(sentence),
                expected,
                "{language} output diverged from Segmenter::segment"
            );
            assert!(!core.has_pos(), "{model} is a segmentation-only model");
            assert_eq!(core.language(), language);
        }
    }

    #[test]
    fn test_segment_batch_matches_single_calls() {
        let core =
            CoreSegmenter::from_path(Language::Japanese, &model_path("japanese.model")).unwrap();
        let sentences = [
            "すもももももももものうち",
            "隣の客はよく柿食う客だ",
            "",
            "東京都から神奈川県へ引っ越した",
        ];

        let batched = core.segment_batch(&sentences);
        let one_by_one: Vec<Vec<String>> = sentences.iter().map(|s| core.segment(s)).collect();

        assert_eq!(batched, one_by_one);
        assert!(batched[2].is_empty(), "an empty sentence yields no tokens");
    }

    #[test]
    fn test_segment_tokens_offsets_slice_the_input() {
        for (language, model, sentence) in segmentation_cases() {
            let core = CoreSegmenter::from_path(language, &model_path(model)).unwrap();
            let tokens = core.segment_tokens(sentence);

            let mut expected_start = 0;
            for token in &tokens {
                assert_eq!(
                    token.byte_start, expected_start,
                    "{language}: tokens must tile the input without gaps"
                );
                assert_eq!(
                    &sentence[token.byte_start..token.byte_end],
                    token.surface,
                    "{language}: offsets must slice back to the surface"
                );
                expected_start = token.byte_end;
            }
            assert_eq!(
                expected_start,
                sentence.len(),
                "{language}: tokens must cover the whole sentence"
            );
        }
    }

    #[test]
    fn test_pos_offsets_slice_the_input() {
        // Korean is the interesting case: its corpus preserves spaces, so a
        // drifting offset would show up here first.
        for (language, model, sentence) in [
            (Language::Japanese, "japanese_pos.model", "すもももももももものうち"),
            (Language::Korean, "korean_pos.model", "안녕하세요 반갑습니다"),
        ] {
            let core = CoreSegmenter::from_path(language, &model_path(model)).unwrap();
            assert!(core.has_pos());

            let tokens = core.segment_with_pos(sentence).unwrap();
            assert!(!tokens.is_empty());

            let mut expected_start = 0;
            for token in &tokens {
                assert_eq!(token.byte_start, expected_start, "{language}: gap in offsets");
                assert_eq!(
                    &sentence[token.byte_start..token.byte_end],
                    token.surface,
                    "{language}: offsets must slice back to the surface"
                );
                assert!(token.pos.is_some(), "{language}: every token must be tagged");
                expected_start = token.byte_end;
            }
            assert_eq!(expected_start, sentence.len());
        }
    }

    #[test]
    fn test_pos_output_matches_litsea() {
        let core = CoreSegmenter::from_path(Language::Japanese, &model_path("japanese_pos.model"))
            .unwrap();
        let sentence = "すもももももももものうち";

        let expected = core.segmenter().segment_with_pos(sentence).unwrap();
        let actual = core.segment_with_pos(sentence).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (token, (surface, pos)) in actual.iter().zip(expected.iter()) {
            assert_eq!(&token.surface, surface);
            assert_eq!(token.pos, Some(*pos));
        }
    }

    #[test]
    fn test_pos_on_segmentation_model_is_a_typed_error() {
        let core =
            CoreSegmenter::from_path(Language::Japanese, &model_path("japanese.model")).unwrap();
        let error = core.segment_with_pos("すもも").unwrap_err();
        assert_eq!(error.kind(), crate::ErrorKind::PosUnavailable);
    }

    #[test]
    fn test_pos_batch_matches_single_calls() {
        let core = CoreSegmenter::from_path(Language::Japanese, &model_path("japanese_pos.model"))
            .unwrap();
        let sentences = ["すもももももももものうち", "隣の客はよく柿食う客だ"];

        let batched = core.segment_with_pos_batch(&sentences).unwrap();
        let one_by_one: Vec<Vec<TokenView>> =
            sentences.iter().map(|s| core.segment_with_pos(s).unwrap()).collect();

        assert_eq!(batched, one_by_one);
    }

    #[test]
    fn test_from_bytes_and_from_uri_agree_with_from_path() {
        let path = model_path("japanese.model");
        let sentence = "すもももももももものうち";

        let from_path = CoreSegmenter::from_path(Language::Japanese, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let from_bytes = CoreSegmenter::from_bytes(Language::Japanese, &bytes).unwrap();
        let from_uri =
            CoreSegmenter::from_uri_blocking(Language::Japanese, &path.display().to_string())
                .unwrap();

        assert_eq!(from_path.segment(sentence), from_bytes.segment(sentence));
        assert_eq!(from_path.segment(sentence), from_uri.segment(sentence));
    }

    #[test]
    fn test_shared_across_threads() {
        let core = Arc::new(
            CoreSegmenter::from_path(Language::Japanese, &model_path("japanese.model")).unwrap(),
        );
        let sentence = "すもももももももものうち";
        let expected = core.segment(sentence);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let core = Arc::clone(&core);
                std::thread::spawn(move || core.segment(sentence))
            })
            .collect();

        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    }
}
