//! Python binding for Litsea.
//!
//! The compiled extension is `litsea._litsea`; the public surface is
//! re-exported from `python/litsea/__init__.py`, which also carries the type
//! stubs. Everything FFI-independent lives in `litsea-binding-core`, so this
//! crate only maps that surface onto Python types, exceptions, and GIL
//! behaviour.
//!
//! ```python
//! from litsea import Segmenter, Language
//!
//! seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")
//! seg.segment("すもももももももものうち")
//! ```

pub mod error;
pub mod language;
pub mod metrics;
pub mod segmenter;
pub mod trainer;
pub mod upos;

use pyo3::prelude::*;

/// Returns the version of the underlying `litsea` crate.
///
/// # Returns
/// The version string, for example `"0.12.0"`.
#[pyfunction]
fn version() -> &'static str {
    litsea::version()
}

/// Initializes the `litsea._litsea` extension module.
///
/// # Arguments
/// * `m` - The module being initialized.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a [`PyErr`] if a class or function cannot be registered.
#[pymodule]
fn _litsea(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", litsea::version())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    m.add_class::<language::PyLanguage>()?;
    m.add_class::<upos::PyUpos>()?;
    m.add_class::<upos::PyToken>()?;
    m.add_class::<segmenter::PySegmenter>()?;
    m.add_class::<trainer::PyCancelToken>()?;
    m.add_class::<trainer::PyExtractor>()?;
    m.add_class::<trainer::PyTrainer>()?;
    m.add_class::<trainer::PyPerceptronTrainer>()?;
    m.add_class::<trainer::PyTwoStageTrainer>()?;
    m.add_class::<metrics::PyBinaryMetrics>()?;
    m.add_class::<metrics::PyMulticlassMetrics>()?;
    m.add_class::<metrics::PyTwoStageMetrics>()?;

    error::register(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_matches_litsea() {
        assert_eq!(version(), litsea::version());
    }
}
