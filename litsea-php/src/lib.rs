//! PHP binding for Litsea.
//!
//! Built with [ext-php-rs](https://github.com/davidcole1340/ext-php-rs).
//! Everything FFI-independent lives in `litsea-binding-core`, so this crate
//! only maps that surface onto PHP classes and exceptions.
//!
//! Unlike the Python and Node.js bindings, nothing here runs in the
//! background: a PHP request is single-threaded and cannot execute code
//! while a native call blocks. A `CancelToken` therefore only takes effect
//! when it is cancelled *before* `train()` is called - see
//! [`trainer`](crate::trainer) for the details.
//!
//! ```php
//! use Litsea\Segmenter;
//!
//! $seg = Segmenter::open('japanese', 'models/japanese.model');
//! $seg->segment('これはテストです。');
//! ```

pub mod error;
pub mod metrics;
pub mod segmenter;
pub mod token;
pub mod trainer;

use ext_php_rs::prelude::*;

/// Returns the version of the underlying `litsea` crate.
///
/// # Returns
/// The version string, for example `"0.12.0"`.
#[php_function]
#[php(name = "Litsea\\version")]
pub fn version() -> String {
    litsea::version().to_string()
}

/// Returns the names of every supported language.
///
/// # Returns
/// The canonical names, in documentation order.
#[php_function]
#[php(name = "Litsea\\supported_languages")]
pub fn supported_languages() -> Vec<String> {
    litsea_binding_core::supported_language_names()
}

/// Registers the extension's classes and functions with PHP.
///
/// # Arguments
/// * `module` - The module builder PHP supplies at load time.
///
/// # Returns
/// The builder with every class and function registered.
#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
        // Exceptions: the base class must be registered before its subclasses.
        .class::<error::LitseaException>()
        .class::<error::InvalidArgumentException>()
        .class::<error::ModelException>()
        .class::<error::IoException>()
        .class::<error::ParseException>()
        .class::<error::UnsupportedException>()
        .class::<error::PosUnavailableException>()
        // Data classes.
        .class::<token::Token>()
        .class::<metrics::PhpBinaryMetrics>()
        .class::<metrics::PhpMulticlassMetrics>()
        .class::<metrics::PhpTwoStageMetrics>()
        // Behaviour.
        .class::<segmenter::Segmenter>()
        .class::<trainer::CancelToken>()
        .class::<trainer::Extractor>()
        .class::<trainer::Trainer>()
        .class::<trainer::PerceptronTrainer>()
        .class::<trainer::TwoStageTrainer>()
        // Functions must be registered explicitly; only classes are picked
        // up by the builder on their own.
        .function(wrap_function!(version))
        .function(wrap_function!(supported_languages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_matches_litsea() {
        assert_eq!(version(), litsea::version());
    }

    #[test]
    fn test_supported_languages() {
        assert_eq!(supported_languages(), vec!["japanese", "chinese", "korean", "english"]);
    }
}
