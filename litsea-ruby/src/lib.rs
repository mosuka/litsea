//! Ruby binding for Litsea.
//!
//! Built with [magnus](https://github.com/matsadler/magnus). Everything
//! FFI-independent lives in `litsea-binding-core`, so this crate only maps
//! that surface onto Ruby classes and exceptions.
//!
//! Long-running work releases the GVL (see [`gvl`]), so other Ruby threads
//! keep running during training - which is what lets one of them cancel a
//! run that is already going.
//!
//! ```ruby
//! require "litsea"
//!
//! seg = Litsea::Segmenter.open("japanese", "models/japanese.model")
//! seg.segment("これはテストです。")
//! ```

pub mod error;
pub mod gvl;
pub mod language;
pub mod metrics;
pub mod segmenter;
pub mod token;
pub mod trainer;

use magnus::{Ruby, error::Error, function};

/// Returns the version of the underlying `litsea` crate.
///
/// # Returns
/// The version string, for example `"0.12.0"`.
fn version() -> String {
    litsea::version().to_string()
}

/// Returns the names of every supported language.
///
/// # Returns
/// The canonical names, in documentation order.
fn supported_languages() -> Vec<String> {
    litsea_binding_core::supported_language_names()
}

/// Entry point Ruby calls when the extension is required.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
///
/// # Returns
/// `()` once every class has been defined.
///
/// # Errors
/// Returns a Ruby exception if a class cannot be defined.
#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let module = ruby.define_module("Litsea")?;

    // Exceptions first: everything else raises through them.
    error::define_exceptions(ruby, &module)?;
    token::define(ruby, &module)?;
    metrics::define(ruby, &module)?;
    segmenter::define(ruby, &module)?;
    trainer::define(ruby, &module)?;

    module.define_module_function("version", function!(version, 0))?;
    module.define_module_function("supported_languages", function!(supported_languages, 0))?;

    Ok(())
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
