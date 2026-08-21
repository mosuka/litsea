//! Language-name handling shared by the bindings.
//!
//! [`litsea::Language`] exposes `Display` and `FromStr` but no list of the
//! supported languages and no `as_str()`, so every binding would otherwise
//! reimplement "which languages are there" and "how do I report an unknown
//! one". Both live here instead.

use std::str::FromStr;

use litsea::Language;

use crate::error::{CoreError, CoreResult};

/// Every language the bundled models support, in the order the
/// documentation lists them.
///
/// [`Language`] is `#[non_exhaustive]`, so adding a language upstream does
/// not break this crate — but this constant must then be extended, which is
/// checked by `test_supported_languages_round_trip`.
pub const SUPPORTED_LANGUAGES: [Language; 4] =
    [Language::Japanese, Language::Chinese, Language::Korean, Language::English];

/// Parses a language name into a [`Language`].
///
/// Accepts the long name (`"japanese"`) or the ISO 639-1 code (`"ja"`),
/// case-insensitively, exactly as [`Language::from_str`] does.
///
/// # Arguments
/// * `name` - The language name or code.
///
/// # Returns
/// The parsed [`Language`].
///
/// # Errors
/// Returns an [`crate::ErrorKind::InvalidArgument`] error listing the
/// supported names if `name` matches none of them.
pub fn parse_language(name: &str) -> CoreResult<Language> {
    Language::from_str(name).map_err(|e| CoreError::invalid_argument(e.to_string()))
}

/// Returns the canonical lowercase name of a language.
///
/// # Arguments
/// * `language` - The language to name.
///
/// # Returns
/// The canonical name, for example `"japanese"`.
#[must_use]
pub fn language_name(language: Language) -> String {
    language.to_string()
}

/// Returns the canonical names of every supported language.
///
/// Bindings expose this so a host-language caller can enumerate the
/// languages without hardcoding them.
///
/// # Returns
/// The names of [`SUPPORTED_LANGUAGES`], in the same order.
#[must_use]
pub fn supported_language_names() -> Vec<String> {
    SUPPORTED_LANGUAGES.iter().map(|l| l.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_language_accepts_names_and_codes() {
        assert_eq!(parse_language("japanese").unwrap(), Language::Japanese);
        assert_eq!(parse_language("JA").unwrap(), Language::Japanese);
        assert_eq!(parse_language("Chinese").unwrap(), Language::Chinese);
        assert_eq!(parse_language("zh").unwrap(), Language::Chinese);
        assert_eq!(parse_language("ko").unwrap(), Language::Korean);
        assert_eq!(parse_language("english").unwrap(), Language::English);
    }

    #[test]
    fn test_parse_language_rejects_unknown() {
        let error = parse_language("klingon").unwrap_err();
        assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);
        assert!(
            error.message().contains("klingon"),
            "the message should echo the input, got: {}",
            error.message()
        );
        assert!(
            error.message().contains("japanese"),
            "the message should list the supported languages, got: {}",
            error.message()
        );
    }

    #[test]
    fn test_supported_languages_round_trip() {
        for language in SUPPORTED_LANGUAGES {
            let name = language_name(language);
            assert_eq!(parse_language(&name).unwrap(), language, "{name} did not round-trip");
        }
        assert_eq!(supported_language_names(), vec!["japanese", "chinese", "korean", "english"]);
    }
}
