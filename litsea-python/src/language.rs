//! The `Language` enum and string coercion.

use litsea::Language;
use litsea_binding_core::{SUPPORTED_LANGUAGES, parse_language};
use pyo3::prelude::*;

use crate::error::map_err;

/// A language supported by Litsea's models.
///
/// Exposed to Python as `litsea.Language`, with the members `JAPANESE`,
/// `CHINESE`, `KOREAN`, and `ENGLISH`. Anywhere a `Language` is accepted,
/// the equivalent string works too (`"japanese"` or `"ja"`, case-insensitive).
// `eq_int` is deliberately not enabled: it would make `Language.JAPANESE == 0`
// true, which surprises callers who never see the discriminants.
#[pyclass(name = "Language", eq, frozen, hash, from_py_object, module = "litsea")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyLanguage {
    /// Japanese.
    #[pyo3(name = "JAPANESE")]
    Japanese,
    /// Chinese (Simplified and Traditional).
    #[pyo3(name = "CHINESE")]
    Chinese,
    /// Korean.
    #[pyo3(name = "KOREAN")]
    Korean,
    /// English.
    #[pyo3(name = "ENGLISH")]
    English,
}

#[pymethods]
impl PyLanguage {
    /// Returns the canonical lowercase name, for example `"japanese"`.
    ///
    /// # Returns
    /// The canonical name of this language.
    #[getter]
    fn name(&self) -> String {
        Language::from(*self).to_string()
    }

    /// Parses a language name or ISO 639-1 code.
    ///
    /// # Arguments
    /// * `name` - `"japanese"` or `"ja"` (case-insensitive), and likewise
    ///   for the other languages.
    ///
    /// # Returns
    /// The matching [`PyLanguage`].
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` if the name is not recognized.
    #[staticmethod]
    fn parse(name: &str) -> PyResult<Self> {
        Ok(Self::from(map_err(parse_language(name))?))
    }

    /// Returns every supported language.
    ///
    /// # Returns
    /// The supported languages, in documentation order.
    #[staticmethod]
    fn all() -> Vec<Self> {
        SUPPORTED_LANGUAGES.into_iter().map(Self::from).collect()
    }

    /// Returns the canonical lowercase name.
    ///
    /// # Returns
    /// The same value as the `name` property.
    fn __str__(&self) -> String {
        self.name()
    }
}

impl From<PyLanguage> for Language {
    /// Converts the Python-facing enum into `litsea`'s enum.
    ///
    /// # Arguments
    /// * `language` - The Python-facing language.
    ///
    /// # Returns
    /// The corresponding [`Language`].
    fn from(language: PyLanguage) -> Self {
        match language {
            PyLanguage::Japanese => Language::Japanese,
            PyLanguage::Chinese => Language::Chinese,
            PyLanguage::Korean => Language::Korean,
            PyLanguage::English => Language::English,
        }
    }
}

impl From<Language> for PyLanguage {
    /// Converts `litsea`'s enum into the Python-facing enum.
    ///
    /// [`Language`] is `#[non_exhaustive]`, so a language added upstream
    /// without a member here falls back to Japanese, `litsea`'s own default.
    /// `test_every_supported_language_has_a_member` fails first if that ever
    /// happens.
    ///
    /// # Arguments
    /// * `language` - The `litsea` language.
    ///
    /// # Returns
    /// The corresponding [`PyLanguage`].
    fn from(language: Language) -> Self {
        match language {
            Language::Japanese => PyLanguage::Japanese,
            Language::Chinese => PyLanguage::Chinese,
            Language::Korean => PyLanguage::Korean,
            Language::English => PyLanguage::English,
            _ => PyLanguage::Japanese,
        }
    }
}

/// A language argument: either a `Language` member or its name as a string.
///
/// Lets `Segmenter.open(Language.JAPANESE, ...)` and
/// `Segmenter.open("ja", ...)` both work.
#[derive(FromPyObject)]
pub enum LanguageArg {
    /// A `litsea.Language` member.
    Enum(PyLanguage),
    /// A language name or ISO 639-1 code.
    Name(String),
}

impl LanguageArg {
    /// Resolves the argument to a `litsea` language.
    ///
    /// # Returns
    /// The resolved [`Language`].
    ///
    /// # Errors
    /// Raises `InvalidArgumentError` if a string does not name a supported
    /// language.
    pub fn resolve(&self) -> PyResult<Language> {
        match self {
            LanguageArg::Enum(language) => Ok(Language::from(*language)),
            LanguageArg::Name(name) => map_err(parse_language(name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_supported_language_has_a_member() {
        for language in SUPPORTED_LANGUAGES {
            let member = PyLanguage::from(language);
            assert_eq!(
                Language::from(member),
                language,
                "{language} does not round-trip through PyLanguage; add a member for it"
            );
        }
    }

    #[test]
    fn test_names_round_trip() {
        for language in SUPPORTED_LANGUAGES {
            let member = PyLanguage::from(language);
            assert_eq!(member.name(), language.to_string());
        }
    }
}
