//! The `Litsea::Segmenter` class.

use std::path::Path;

use litsea_binding_core::{CoreSegmenter, TokenView};
use magnus::{Module, Object, RArray, RModule, RString, Ruby, Value, error::Error};

use crate::error::map_err;
use crate::gvl::without_gvl;
use crate::language::language_from_value;
use crate::token::Token;

/// A word segmenter, optionally with POS tagging.
///
/// Build one with `Segmenter.open`, `Segmenter.from_bytes`, or
/// `Segmenter.from_uri`. The kind of model is detected from the file itself,
/// so `has_pos?` describes what was loaded rather than something the caller
/// declares.
#[magnus::wrap(class = "Litsea::Segmenter", free_immediately, size)]
pub struct Segmenter {
    /// The wrapped core segmenter.
    inner: CoreSegmenter,
}

impl Segmenter {
    /// Loads a model from a filesystem path.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code, as a String or Symbol.
    /// * `path` - Path to the model file.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError`, `Litsea::IoError`,
    /// `Litsea::ParseError`, or `Litsea::ModelError`.
    fn open(language: Value, path: String) -> Result<Self, Error> {
        let language = language_from_value(language)?;
        // Reading and compiling a model is measurable work; let other Ruby
        // threads run while it happens.
        let inner = without_gvl(|| CoreSegmenter::from_path(language, Path::new(&path)));
        Ok(Self {
            inner: map_err(inner)?,
        })
    }

    /// Loads a model from a raw byte string.
    ///
    /// Takes the bytes as they are, so a String read with `File.binread`
    /// (ASCII-8BIT) works as well as one read as UTF-8.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code, as a String or Symbol.
    /// * `data` - The model file contents.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError`, `Litsea::ParseError`, or
    /// `Litsea::ModelError`.
    fn from_bytes(language: Value, data: RString) -> Result<Self, Error> {
        let language = language_from_value(language)?;
        // SAFETY: the slice is used only within this call, and nothing here
        // runs Ruby code that could move or free the string in the meantime
        // (`from_bytes` parses into owned Rust structures).
        let bytes = unsafe { data.as_slice() };
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes(language, bytes))?,
        })
    }

    /// Loads a model from a URI.
    ///
    /// Accepts a filesystem path, a `file://` path, or an `http(s)://` URL.
    /// The GVL is released while the model is fetched.
    ///
    /// # Arguments
    /// * `language` - A language name or ISO 639-1 code, as a String or Symbol.
    /// * `uri` - The model URI.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::ModelError` if the download fails, plus the same
    /// errors as `open`.
    fn from_uri(language: Value, uri: String) -> Result<Self, Error> {
        let language = language_from_value(language)?;
        let inner = without_gvl(|| CoreSegmenter::from_uri_blocking(language, &uri));
        Ok(Self {
            inner: map_err(inner)?,
        })
    }

    /// Returns the language this segmenter was built for.
    ///
    /// # Returns
    /// The canonical language name, for example `"japanese"`.
    fn language(&self) -> String {
        self.inner.language().to_string()
    }

    /// Returns whether this segmenter can tag parts of speech.
    ///
    /// # Returns
    /// `true` when a two-stage POS model was loaded.
    fn has_pos(&self) -> bool {
        self.inner.has_pos()
    }

    /// Splits a sentence into tokens.
    ///
    /// # Arguments
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, in order.
    fn segment(&self, text: String) -> Vec<String> {
        self.inner.segment(&text)
    }

    /// Splits several sentences into tokens, releasing the GVL.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment.
    ///
    /// # Returns
    /// One token array per input sentence, in input order.
    fn segment_batch(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        without_gvl(|| self.inner.segment_batch(&texts))
    }

    /// Splits a sentence into tokens carrying byte offsets.
    ///
    /// # Arguments
    /// * `ruby` - The Ruby handle for the current thread.
    /// * `text` - The sentence to segment.
    ///
    /// # Returns
    /// The tokens, with `pos` set to `nil`.
    ///
    /// # Errors
    /// Returns a Ruby exception if the result array cannot be built.
    fn segment_tokens(ruby: &Ruby, rb_self: &Self, text: String) -> Result<RArray, Error> {
        tokens_to_array(ruby, rb_self.inner.segment_tokens(&text))
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
    /// Raises `Litsea::PosUnavailableError` when this segmenter was built
    /// from a segmentation-only model.
    fn segment_with_pos(ruby: &Ruby, rb_self: &Self, text: String) -> Result<RArray, Error> {
        let tokens = map_err(rb_self.inner.segment_with_pos(&text))?;
        tokens_to_array(ruby, tokens)
    }

    /// Splits and tags several sentences, releasing the GVL.
    ///
    /// # Arguments
    /// * `texts` - The sentences to segment and tag.
    ///
    /// # Returns
    /// One tagged-token array per input sentence, in input order.
    ///
    /// # Errors
    /// Raises `Litsea::PosUnavailableError` when this segmenter was built
    /// from a segmentation-only model.
    fn segment_with_pos_batch(
        ruby: &Ruby,
        rb_self: &Self,
        texts: Vec<String>,
    ) -> Result<RArray, Error> {
        let batches = without_gvl(|| rb_self.inner.segment_with_pos_batch(&texts));
        let outer = ruby.ary_new_capa(texts.len());
        for tokens in map_err(batches)? {
            outer.push(tokens_to_array(ruby, tokens)?)?;
        }
        Ok(outer)
    }
}

/// Builds a Ruby array of [`Token`] objects.
///
/// Wrapped types satisfy `IntoValue` but not `IntoValueFromNative`, which is
/// what `Vec<T>` conversion requires, so the array is built element by
/// element.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `tokens` - The token views to wrap.
///
/// # Returns
/// A Ruby array of `Litsea::Token` objects.
///
/// # Errors
/// Returns a Ruby exception if an element cannot be pushed.
fn tokens_to_array(ruby: &Ruby, tokens: Vec<TokenView>) -> Result<RArray, Error> {
    let array = ruby.ary_new_capa(tokens.len());
    for view in tokens {
        array.push(Token::from(view))?;
    }
    Ok(array)
}

/// Defines `Litsea::Segmenter`.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `module` - The `Litsea` module to define the class on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a Ruby exception if the class cannot be defined.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("Segmenter", ruby.class_object())?;
    class.define_singleton_method("open", magnus::function!(Segmenter::open, 2))?;
    class.define_singleton_method("from_bytes", magnus::function!(Segmenter::from_bytes, 2))?;
    class.define_singleton_method("from_uri", magnus::function!(Segmenter::from_uri, 2))?;
    class.define_method("language", magnus::method!(Segmenter::language, 0))?;
    class.define_method("has_pos?", magnus::method!(Segmenter::has_pos, 0))?;
    class.define_method("segment", magnus::method!(Segmenter::segment, 1))?;
    class.define_method("segment_batch", magnus::method!(Segmenter::segment_batch, 1))?;
    class.define_method("segment_tokens", magnus::method!(Segmenter::segment_tokens, 1))?;
    class.define_method("segment_with_pos", magnus::method!(Segmenter::segment_with_pos, 1))?;
    class.define_method(
        "segment_with_pos_batch",
        magnus::method!(Segmenter::segment_with_pos_batch, 1),
    )?;
    Ok(())
}
