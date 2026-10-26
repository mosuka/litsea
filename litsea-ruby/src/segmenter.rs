//! The `Litsea::Segmenter` class.

use std::collections::HashMap;
use std::path::Path;

use litsea::UserDictionary;
use litsea_binding_core::{CoreSegmenter, TokenView, build_user_dictionary_with_tags};
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Module, Object, RArray, RModule, RString, Ruby, TryConvert, Value, error::Error};

use crate::error::map_err;
use crate::gvl::without_gvl;
use crate::language::language_from_value;
use crate::token::Token;

/// The keyword arguments every constructor accepts (issues #263, #264): the
/// optional `user_words:`, `user_words_gate:` and `user_word_tags:`, each of
/// which may also be passed as `nil`.
struct UserWordsArgs {
    /// The words to emit as single tokens.
    user_words: Option<Vec<String>>,
    /// The flip-ratio gate in `0.0..=1.0`.
    user_words_gate: Option<f64>,
    /// Words to emit as single tokens with the UPOS tag `segment_with_pos`
    /// gives them.
    user_word_tags: Option<HashMap<String, String>>,
}

impl UserWordsArgs {
    /// Builds the user dictionary these arguments describe.
    ///
    /// # Returns
    /// `None` when no words were given, otherwise the dictionary.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` if a tag is not one of the 17
    /// UPOS tags, or if the gate is outside `0.0..=1.0`, also when no words
    /// were given.
    fn dictionary(self) -> Result<Option<UserDictionary>, Error> {
        let word_tags: Vec<(String, String)> =
            self.user_word_tags.unwrap_or_default().into_iter().collect();
        map_err(build_user_dictionary_with_tags(
            self.user_words.as_deref().unwrap_or_default(),
            &word_tags,
            self.user_words_gate,
        ))
    }
}

/// Splits the arguments of `open`, `from_bytes`, and `from_uri`: the
/// language and the model source as required positionals, then the
/// `user_words:` / `user_words_gate:` / `user_word_tags:` keywords.
///
/// magnus cannot express optional keyword arguments in a `function!`
/// arity, so the constructors take their arguments through `scan_args`,
/// as the trainers do.
///
/// # Arguments
/// * `args` - The raw Ruby arguments.
///
/// # Returns
/// The language value, the model source, and the keyword arguments.
///
/// # Errors
/// Returns a Ruby `ArgumentError` for a wrong argument count or an unknown
/// keyword, or a `TypeError` if an argument has the wrong type.
fn scan_constructor_args<T: TryConvert>(
    args: &[Value],
) -> Result<(Value, T, UserWordsArgs), Error> {
    let args = scan_args::<(Value, T), (), (), (), _, ()>(args)?;
    let (language, source) = args.required;
    let kwargs = get_kwargs::<
        _,
        (),
        (
            Option<Option<Vec<String>>>,
            Option<Option<f64>>,
            Option<Option<HashMap<String, String>>>,
        ),
        (),
    >(args.keywords, &[], &["user_words", "user_words_gate", "user_word_tags"])?;
    let (user_words, user_words_gate, user_word_tags) = kwargs.optional;
    Ok((
        language,
        source,
        UserWordsArgs {
            user_words: user_words.flatten(),
            user_words_gate: user_words_gate.flatten(),
            user_word_tags: user_word_tags.flatten(),
        },
    ))
}

/// A word segmenter, optionally with POS tagging.
///
/// Build one with `Segmenter.open`, `Segmenter.from_bytes`, or
/// `Segmenter.from_uri`. The kind of model is detected from the file itself,
/// so `has_pos?` describes what was loaded rather than something the caller
/// declares. Every constructor accepts the keyword arguments `user_words:`,
/// an array of words to emit as single tokens; `user_words_gate:`, a flip
/// ratio in `0.0..=1.0` that applies a word only where the model's margin
/// against it is small enough; and `user_word_tags:`, a Hash mapping further
/// words to the UPOS tag `segment_with_pos` gives them.
#[magnus::wrap(class = "Litsea::Segmenter", free_immediately, size)]
pub struct Segmenter {
    /// The wrapped core segmenter.
    inner: CoreSegmenter,
}

impl Segmenter {
    /// Loads a model from a filesystem path.
    ///
    /// # Arguments
    /// * `args` - `language` (a language name or ISO 639-1 code, as a String
    ///   or Symbol), `path` (the model file), and the optional
    ///   `user_words:` / `user_words_gate:` / `user_word_tags:` keywords.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` (also for a tag that is not a
    /// UPOS tag or a gate outside `0.0..=1.0`), `Litsea::IoError`,
    /// `Litsea::ParseError`, or `Litsea::ModelError`.
    fn open(args: &[Value]) -> Result<Self, Error> {
        let (language, path, user_words): (_, String, _) = scan_constructor_args(args)?;
        let language = language_from_value(language)?;
        let dictionary = user_words.dictionary()?;
        // Reading and compiling a model is measurable work; let other Ruby
        // threads run while it happens.
        let inner =
            without_gvl(|| CoreSegmenter::from_path_with(language, Path::new(&path), dictionary));
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
    /// * `args` - `language` (a language name or ISO 639-1 code, as a String
    ///   or Symbol), `data` (the model file contents), and the optional
    ///   `user_words:` / `user_words_gate:` / `user_word_tags:` keywords.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::InvalidArgumentError` (also for a tag that is not a
    /// UPOS tag or a gate outside `0.0..=1.0`), `Litsea::ParseError`, or
    /// `Litsea::ModelError`.
    fn from_bytes(args: &[Value]) -> Result<Self, Error> {
        let (language, data, user_words): (_, RString, _) = scan_constructor_args(args)?;
        let language = language_from_value(language)?;
        let dictionary = user_words.dictionary()?;
        // SAFETY: the slice is used only within this call, and nothing here
        // runs Ruby code that could move or free the string in the meantime
        // (`from_bytes_with` parses into owned Rust structures).
        let bytes = unsafe { data.as_slice() };
        Ok(Self {
            inner: map_err(CoreSegmenter::from_bytes_with(language, bytes, dictionary))?,
        })
    }

    /// Loads a model from a URI.
    ///
    /// Accepts a filesystem path, a `file://` path, or an `http(s)://` URL.
    /// The GVL is released while the model is fetched.
    ///
    /// # Arguments
    /// * `args` - `language` (a language name or ISO 639-1 code, as a String
    ///   or Symbol), `uri` (the model URI), and the optional `user_words:` /
    ///   `user_words_gate:` / `user_word_tags:` keywords.
    ///
    /// # Returns
    /// The new segmenter.
    ///
    /// # Errors
    /// Raises `Litsea::ModelError` if the download fails, plus the same
    /// errors as `open`.
    fn from_uri(args: &[Value]) -> Result<Self, Error> {
        let (language, uri, user_words): (_, String, _) = scan_constructor_args(args)?;
        let language = language_from_value(language)?;
        let dictionary = user_words.dictionary()?;
        let inner =
            without_gvl(|| CoreSegmenter::from_uri_blocking_with(language, &uri, dictionary));
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
    class.define_singleton_method("open", magnus::function!(Segmenter::open, -1))?;
    class.define_singleton_method("from_bytes", magnus::function!(Segmenter::from_bytes, -1))?;
    class.define_singleton_method("from_uri", magnus::function!(Segmenter::from_uri, -1))?;
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
