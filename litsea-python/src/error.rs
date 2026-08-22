//! Exception hierarchy exposed to Python.
//!
//! `litsea_binding_core::ErrorKind` already categorizes every failure, so
//! this module is a one-to-one mapping from those categories onto Python
//! exception classes. Every class derives from `LitseaError`, so
//! `except LitseaError` catches everything the binding can raise; no class
//! also inherits from a builtin (single inheritance keeps that guarantee
//! simple to reason about).

use litsea_binding_core::{CoreError, ErrorKind};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(litsea, LitseaError, PyException, "Base class for every error raised by litsea.");
create_exception!(
    litsea,
    InvalidArgumentError,
    LitseaError,
    "An argument was invalid, such as an unknown language name."
);
create_exception!(
    litsea,
    ModelError,
    LitseaError,
    "A model could not be obtained, or is not the kind the call needs."
);
create_exception!(litsea, IoError, LitseaError, "A file could not be read or written.");
create_exception!(litsea, ParseError, LitseaError, "A model or training data file is malformed.");
create_exception!(
    litsea,
    UnsupportedError,
    LitseaError,
    "The operation is not supported in this build or environment."
);
create_exception!(
    litsea,
    PosUnavailableError,
    LitseaError,
    "POS tagging was requested from a segmentation-only model."
);

/// Converts a [`CoreError`] into the matching Python exception.
///
/// # Arguments
/// * `error` - The error to convert.
///
/// # Returns
/// A [`PyErr`] of the class matching the error's kind;
/// [`ErrorKind::Runtime`] maps to the `LitseaError` base class.
pub fn to_py_err(error: CoreError) -> PyErr {
    let message = error.message().to_string();
    match error.kind() {
        ErrorKind::InvalidArgument => InvalidArgumentError::new_err(message),
        ErrorKind::Model => ModelError::new_err(message),
        ErrorKind::Io => IoError::new_err(message),
        ErrorKind::Parse => ParseError::new_err(message),
        ErrorKind::Unsupported => UnsupportedError::new_err(message),
        ErrorKind::PosUnavailable => PosUnavailableError::new_err(message),
        ErrorKind::Runtime => LitseaError::new_err(message),
    }
}

/// Result alias for binding methods.
pub type PyLitseaResult<T> = Result<T, PyErr>;

/// Converts a [`CoreError`] result into a Python result.
///
/// # Arguments
/// * `result` - The result to convert.
///
/// # Returns
/// The original value, or the mapped Python exception.
pub fn map_err<T>(result: Result<T, CoreError>) -> PyLitseaResult<T> {
    result.map_err(to_py_err)
}

/// Registers the exception classes on the extension module.
///
/// # Arguments
/// * `m` - The module to register the classes on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a [`PyErr`] if a class cannot be added to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("LitseaError", m.py().get_type::<LitseaError>())?;
    m.add("InvalidArgumentError", m.py().get_type::<InvalidArgumentError>())?;
    m.add("ModelError", m.py().get_type::<ModelError>())?;
    m.add("IoError", m.py().get_type::<IoError>())?;
    m.add("ParseError", m.py().get_type::<ParseError>())?;
    m.add("UnsupportedError", m.py().get_type::<UnsupportedError>())?;
    m.add("PosUnavailableError", m.py().get_type::<PosUnavailableError>())?;
    Ok(())
}
