//! The `Litsea\Segmenter` class.

use std::collections::HashMap;
use std::path::Path;

use ext_php_rs::prelude::*;
use litsea::UserDictionary;
use litsea_binding_core::{CoreSegmenter, build_user_dictionary_with_tags, parse_language};

use crate::error::map_err;
use crate::token::Token;

/// Builds the user dictionary from the constructors' optional trailing
/// arguments (issues #263, #264).
///
/// # Arguments
/// * `user_words` - The words to emit as single tokens, or `null`.
/// * `user_words_gate` - The flip-ratio gate in `0.0..=1.0`, or `null`.
/// * `user_word_tags` - Words to emit as single tokens with the UPOS tag
///   `segmentWithPos` gives them, or `null`.
///
/// # Returns
/// `None` when no words were given, otherwise the dictionary.
///
/// # Errors
/// Throws `Litsea\InvalidArgumentException` if a tag is not one of the 17
/// UPOS tags, or if the gate is outside `0.0..=1.0`, also when no words were
/// given.
fn user_dictionary(
    user_words: Option<Vec<String>>,
    user_words_gate: Option<f64>,
    user_word_tags: Option<HashMap<String, String>>,
) -> PhpResult<Option<UserDictionary>> {
    let word_tags: Vec<(String, String)> = user_word_tags.unwrap_or_default().into_iter().collect();
    map_err(build_user_dictionary_with_tags(
        user_words.as_deref().unwrap_or_default(),
        &word_tags,
        user_words_gate,
    ))
}

/// A word segmenter, optionally with POS tagging.
///
/// Build one with `Segmenter::open`, `Segmenter::fromBytes`, or
/// `Segmenter::fromUri`. The kind of model is detected from the file itself,
/// so `hasPos()` describes what was loaded rather than something the caller
/// declares. Every constructor takes three optional trailing arguments:
/// `$user_words`, an array of words to emit as single tokens;
/// `$user_words_gate`, a flip ratio in `0.0..=1.0` that applies a word only
/// where the model's margin against it is small enough; and
/// `$user_word_tags`, an array mapping further words to the UPOS tag
/// `segmentWithPos()` gives them.
#[php_class]
#[php(name = "Litsea\\Segmenter")]
pub struct Segmenter {
    /// The wrapped core segmenter.
    inner: CoreSegmenter,
}

#[php_impl]
impl Segmenter {
    /// Loads a model from a filesystem path.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code (`"japanese"`, `"ja"`).
    /// * `path` - Path to the model file.
    /// * `user_words` - Optional array of words to emit as single tokens.
    /// * `user_words_gate` - Optional flip-ratio gate for `$user_words`, in
    ///   `0.0..=1.0`.
    /// * `user_word_tags` - Optional array mapping words to emit as single
    ///   tokens to the UPOS tag `segmentWithPos()` gives them.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` (also for a tag that is not a
    /// UPOS tag or a gate outside `0.0..=1.0`), `Litsea\IoException`,
    /// `Litsea\ParseException`, or `Litsea\ModelException`.
    // A trailing `Option` argument is already optional with a `null`
    // default, so no `defaults` attribute is needed.
    pub fn open(
        language: String,
        path: String,
        user_words: Option<Vec<String>>,
        user_words_gate: Option<f64>,
        user_word_tags: Option<HashMap<String, String>>,
    ) -> PhpResult<Self> {
        let language = map_err(parse_language(&language))?;
        let dictionary = user_dictionary(user_words, user_words_gate, user_word_tags)?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_path_with(language, Path::new(&path), dictionary))?,
        })
    }

    /// Loads a model from a model file's contents.
    ///
    /// Model files are UTF-8 text, so the argument is taken as a PHP string
    /// rather than a binary buffer; ext-php-rs validates the encoding on the
    /// way in.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    /// * `data` - The model file contents.
    /// * `user_words` - Optional array of words to emit as single tokens.
    /// * `user_words_gate` - Optional flip-ratio gate for `$user_words`, in
    ///   `0.0..=1.0`.
    /// * `user_word_tags` - Optional array mapping words to emit as single
    ///   tokens to the UPOS tag `segmentWithPos()` gives them.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Throws `Litsea\InvalidArgumentException` (also for a tag that is not a
    /// UPOS tag or a gate outside `0.0..=1.0`), `Litsea\ParseException`, or
    /// `Litsea\ModelException`.
    pub fn from_bytes(
        language: String,
        data: String,
        user_words: Option<Vec<String>>,
        user_words_gate: Option<f64>,
        user_word_tags: Option<HashMap<String, String>>,
    ) -> PhpResult<Self> {
        let language = map_err(parse_language(&language))?;
        let dictionary = user_dictionary(user_words, user_words_gate, user_word_tags)?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes_with(language, data.as_bytes(), dictionary))?,
        })
    }

    /// Loads a model from a URI, blocking until it is available.
    ///
    /// Accepts a filesystem path, a `file://` path, or an `http(s)://` URL.
    /// The call blocks the request: PHP has no way to do this work in the
    /// background.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code.
    /// * `uri` - The model URI.
    /// * `user_words` - Optional array of words to emit as single tokens.
    /// * `user_words_gate` - Optional flip-ratio gate for `$user_words`, in
    ///   `0.0..=1.0`.
    /// * `user_word_tags` - Optional array mapping words to emit as single
    ///   tokens to the UPOS tag `segmentWithPos()` gives them.
    ///
    /// # Returns
    /// The new `Segmenter`.
    ///
    /// # Errors
    /// Throws `Litsea\ModelException` if the download fails, plus the same
    /// exceptions as `open`.
    pub fn from_uri(
        language: String,
        uri: String,
        user_words: Option<Vec<String>>,
        user_words_gate: Option<f64>,
        user_word_tags: Option<HashMap<String, String>>,
    ) -> PhpResult<Self> {
        let language = map_err(parse_language(&language))?;
        let dictionary = user_dictionary(user_words, user_words_gate, user_word_tags)?;
        Ok(Self {
            inner: map_err(CoreSegmenter::from_uri_blocking_with(language, &uri, dictionary))?,
        })
    }

    /// Returns the language this segmenter was built for.
    ///
    /// # Returns
    /// The canonical language name, for example `"japanese"`.
    pub fn language(&self) -> String {
        self.inner.language().to_string()
    }

    /// Returns whether this segmenter can tag parts of speech.
    ///
    /// # Returns
    /// `true` when a two-stage POS model was loaded.
    pub fn has_pos(&self) -> bool {
        self.inner.has_pos()
    }

    /// Splits a sentence into tokens.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, in order.
    pub fn segment(&self, text: String) -> Vec<String> {
        self.inner.segment(&text)
    }

    /// Splits several sentences into tokens, reusing one scratch buffer.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// One token array per input sentence, in input order.
    pub fn segment_batch(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        self.inner.segment_batch(&texts)
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, with `pos` set to `null`.
    pub fn segment_tokens(&self, text: String) -> Vec<Token> {
        self.inner.segment_tokens(&text).into_iter().map(Token::from).collect()
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
    /// Throws `Litsea\PosUnavailableException` when this segmenter was built
    /// from a segmentation-only model.
    pub fn segment_with_pos(&self, text: String) -> PhpResult<Vec<Token>> {
        Ok(map_err(self.inner.segment_with_pos(&text))?
            .into_iter()
            .map(Token::from)
            .collect())
    }

    /// Splits and tags several sentences.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment and tag.
    ///
    /// # Returns
    /// One tagged-token array per input sentence, in input order.
    ///
    /// # Errors
    /// Throws `Litsea\PosUnavailableException` when this segmenter was built
    /// from a segmentation-only model.
    pub fn segment_with_pos_batch(&self, texts: Vec<String>) -> PhpResult<Vec<Vec<Token>>> {
        Ok(map_err(self.inner.segment_with_pos_batch(&texts))?
            .into_iter()
            .map(|tokens| tokens.into_iter().map(Token::from).collect())
            .collect())
    }
}
