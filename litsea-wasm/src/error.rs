//! Error conversion for the WebAssembly binding.
//!
//! The Node.js binding puts the error kind on `err.code` (#203); this one
//! does the same, so the two JavaScript bindings agree on how a caller
//! branches on a failure. Here it is direct: build a `js_sys::Error` and set
//! the property.

use js_sys::{Error as JsError, Reflect};
use litsea_binding_core::{CoreError, ErrorKind};
use wasm_bindgen::JsValue;

/// Converts a [`CoreError`] into a JavaScript `Error` carrying its category.
///
/// # Arguments
/// * `error` - The error to convert.
///
/// # Returns
/// A `JsValue` holding an `Error` whose `message` is the original text and
/// whose `code` is the stable identifier from [`ErrorKind::as_str`].
pub fn to_js_error(error: CoreError) -> JsValue {
    let js_error = JsError::new(error.message());
    // `Reflect::set` only fails if the target is not an object, which an
    // Error always is; ignoring the result keeps this infallible without an
    // unwrap.
    let _ = Reflect::set(
        &js_error,
        &JsValue::from_str("code"),
        &JsValue::from_str(error.kind().as_str()),
    );
    js_error.into()
}

/// Creates an invalid-argument JavaScript error.
///
/// # Arguments
/// * `message` - The message to report.
///
/// # Returns
/// A `JsValue` holding an `Error` with the `invalid_argument` code.
pub fn invalid_argument(message: impl Into<String>) -> JsValue {
    to_js_error(CoreError::new(ErrorKind::InvalidArgument, message))
}

/// Maps a core result onto a JavaScript-facing result.
///
/// # Arguments
/// * `result` - The result to convert.
///
/// # Returns
/// The original value, or the converted error.
pub fn map_err<T>(result: Result<T, CoreError>) -> Result<T, JsValue> {
    result.map_err(to_js_error)
}
