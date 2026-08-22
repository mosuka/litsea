//! The Ruby exception hierarchy.
//!
//! Mirrors the Python and PHP bindings: one class per [`ErrorKind`], all
//! below `Litsea::Error`, so `rescue Litsea::Error` catches everything the
//! binding raises.

use litsea_binding_core::{CoreError, ErrorKind};
use magnus::{Module, RModule, Ruby, error::Error, exception::ExceptionClass};

/// Holds the exception classes, looked up from the `Litsea` module.
///
/// Ruby classes are values, not types, so they are resolved on demand rather
/// than stored in Rust statics (a `Ruby` handle is only valid on a Ruby
/// thread).
struct Exceptions {
    /// `Litsea::Error`, the base class.
    base: ExceptionClass,
    /// `Litsea::InvalidArgumentError`.
    invalid_argument: ExceptionClass,
    /// `Litsea::ModelError`.
    model: ExceptionClass,
    /// `Litsea::IoError`.
    io: ExceptionClass,
    /// `Litsea::ParseError`.
    parse: ExceptionClass,
    /// `Litsea::UnsupportedError`.
    unsupported: ExceptionClass,
    /// `Litsea::PosUnavailableError`.
    pos_unavailable: ExceptionClass,
}

/// Looks up the exception classes defined by [`define_exceptions`].
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
///
/// # Returns
/// The exception classes, or `None` if the module has not been defined yet.
fn exceptions(ruby: &Ruby) -> Option<Exceptions> {
    let module: RModule = ruby.class_object().const_get("Litsea").ok()?;
    Some(Exceptions {
        base: module.const_get("Error").ok()?,
        invalid_argument: module.const_get("InvalidArgumentError").ok()?,
        model: module.const_get("ModelError").ok()?,
        io: module.const_get("IoError").ok()?,
        parse: module.const_get("ParseError").ok()?,
        unsupported: module.const_get("UnsupportedError").ok()?,
        pos_unavailable: module.const_get("PosUnavailableError").ok()?,
    })
}

/// Defines `Litsea::Error` and its subclasses on the given module.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `module` - The `Litsea` module to define the classes on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a Ruby exception if a class cannot be defined.
pub fn define_exceptions(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let base = module.define_error("Error", ruby.exception_standard_error())?;
    module.define_error("InvalidArgumentError", base)?;
    module.define_error("ModelError", base)?;
    module.define_error("IoError", base)?;
    module.define_error("ParseError", base)?;
    module.define_error("UnsupportedError", base)?;
    module.define_error("PosUnavailableError", base)?;
    Ok(())
}

/// Converts a [`CoreError`] into the matching Ruby exception.
///
/// # Arguments
/// * `error` - The error to convert.
///
/// # Returns
/// A magnus [`Error`] carrying the class that matches the error's kind;
/// [`ErrorKind::Runtime`] uses `Litsea::Error` itself.
pub fn to_ruby_error(error: CoreError) -> Error {
    let message = error.message().to_string();
    let Ok(ruby) = Ruby::get() else {
        // Not on a Ruby thread, which cannot happen for a call that came
        // from Ruby - but the type system does not know that. The
        // handle-based replacement for this constructor needs the very
        // handle we just failed to get, so the deprecated free function is
        // the only way to build an error here.
        #[allow(deprecated)]
        return Error::new(magnus::exception::fatal(), message);
    };

    let Some(classes) = exceptions(&ruby) else {
        return Error::new(ruby.exception_runtime_error(), message);
    };

    let class = match error.kind() {
        ErrorKind::InvalidArgument => classes.invalid_argument,
        ErrorKind::Model => classes.model,
        ErrorKind::Io => classes.io,
        ErrorKind::Parse => classes.parse,
        ErrorKind::Unsupported => classes.unsupported,
        ErrorKind::PosUnavailable => classes.pos_unavailable,
        ErrorKind::Runtime => classes.base,
    };

    Error::new(class, message)
}

/// Maps a core result onto a Ruby result.
///
/// # Arguments
/// * `result` - The result to convert.
///
/// # Returns
/// The original value, or the mapped Ruby exception.
pub fn map_err<T>(result: Result<T, CoreError>) -> Result<T, Error> {
    result.map_err(to_ruby_error)
}
