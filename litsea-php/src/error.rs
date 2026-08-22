//! The PHP exception hierarchy.
//!
//! ext-php-rs can register classes that extend `\Exception`, so this mirrors
//! the Python binding exactly: one class per [`ErrorKind`], all deriving from
//! `Litsea\LitseaException`, so a single `catch (LitseaException $e)` catches
//! everything the binding throws.

use ext_php_rs::exception::PhpException;
use ext_php_rs::prelude::*;
use ext_php_rs::zend::ce;
use litsea_binding_core::{CoreError, ErrorKind};

/// Base class for every exception the binding throws.
#[php_class]
#[php(name = "Litsea\\LitseaException")]
#[php(extends(ce = ce::exception, stub = "\\Exception"))]
#[derive(Default)]
pub struct LitseaException;

/// An argument was invalid, such as an unknown language name.
#[php_class]
#[php(name = "Litsea\\InvalidArgumentException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct InvalidArgumentException;

/// A model could not be obtained, or is not the kind the call needs.
#[php_class]
#[php(name = "Litsea\\ModelException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct ModelException;

/// A file could not be read or written.
#[php_class]
#[php(name = "Litsea\\IoException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct IoException;

/// A model or training data file is malformed.
#[php_class]
#[php(name = "Litsea\\ParseException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct ParseException;

/// The operation is not supported in this build or environment.
#[php_class]
#[php(name = "Litsea\\UnsupportedException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct UnsupportedException;

/// POS tagging was requested from a segmentation-only model.
#[php_class]
#[php(name = "Litsea\\PosUnavailableException")]
#[php(extends(LitseaException))]
#[derive(Default)]
pub struct PosUnavailableException;

/// Converts a [`CoreError`] into the matching PHP exception.
///
/// # Arguments
/// * `error` - The error to convert.
///
/// # Returns
/// A [`PhpException`] of the class matching the error's kind;
/// [`ErrorKind::Runtime`] uses the `LitseaException` base class.
pub fn to_php_exception(error: CoreError) -> PhpException {
    let message = error.message().to_string();
    match error.kind() {
        ErrorKind::InvalidArgument => PhpException::from_class::<InvalidArgumentException>(message),
        ErrorKind::Model => PhpException::from_class::<ModelException>(message),
        ErrorKind::Io => PhpException::from_class::<IoException>(message),
        ErrorKind::Parse => PhpException::from_class::<ParseException>(message),
        ErrorKind::Unsupported => PhpException::from_class::<UnsupportedException>(message),
        ErrorKind::PosUnavailable => PhpException::from_class::<PosUnavailableException>(message),
        ErrorKind::Runtime => PhpException::from_class::<LitseaException>(message),
    }
}

/// Maps a core result onto a PHP result.
///
/// # Arguments
/// * `result` - The result to convert.
///
/// # Returns
/// The original value, or the mapped exception.
pub fn map_err<T>(result: Result<T, CoreError>) -> PhpResult<T> {
    result.map_err(to_php_exception)
}
