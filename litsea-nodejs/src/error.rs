//! Error conversion for the Node.js binding.
//!
//! JavaScript has no exception hierarchy to mirror the Python binding's
//! classes, so the error category travels as `err.code` - the property Node
//! convention already reserves for branching on an error without matching
//! message text.
//!
//! Getting that code onto both the synchronous and the asynchronous path
//! takes two mechanisms, because `napi::Status` is a closed enum:
//!
//! - synchronous calls return [`KindError`] (`napi::Error<String>`), whose
//!   status string becomes `err.code` directly;
//! - `Task::compute` must return `napi::Result`, so a task records the kind
//!   with [`task_call`] and rebuilds the error with the code in
//!   `Task::reject` via [`error_with_code`].

use litsea_binding_core::{CoreError, ErrorKind};
use napi::{Env, Error, JsValue, Status};

/// A napi error whose status string becomes the JavaScript `err.code`.
pub type KindError = Error<String>;

/// Converts a [`CoreError`] into a JavaScript error carrying its category.
///
/// # Arguments
/// * `error` - The error to convert.
///
/// # Returns
/// A [`KindError`] whose `code` is the stable identifier from
/// [`ErrorKind::as_str`] (`invalid_argument`, `model`, `io`, `parse`,
/// `unsupported`, `pos_unavailable`, or `runtime`).
pub fn to_napi_error(error: CoreError) -> KindError {
    Error::new(error.kind().as_str().to_string(), error.message().to_string())
}

/// Creates an invalid-argument error.
///
/// # Arguments
/// * `message` - The message to report.
///
/// # Returns
/// A [`KindError`] with the `invalid_argument` code.
pub fn invalid_argument(message: impl Into<String>) -> KindError {
    to_napi_error(CoreError::new(ErrorKind::InvalidArgument, message))
}

/// Maps a core result onto a synchronous binding result.
///
/// # Arguments
/// * `result` - The result to convert.
///
/// # Returns
/// The original value, or the converted error.
pub fn map_err<T>(result: Result<T, CoreError>) -> Result<T, KindError> {
    result.map_err(to_napi_error)
}

/// Runs a core call inside a task, remembering the error kind.
///
/// `Task::compute` is constrained to `napi::Result`, which cannot carry a
/// custom code, so the kind is stashed in the task for
/// [`error_with_code`] to use.
///
/// # Arguments
/// * `kind_slot` - The task field that records the failing kind.
/// * `result` - The core call's result.
///
/// # Returns
/// The original value, or a napi error carrying the message.
pub fn task_call<T>(
    kind_slot: &mut Option<String>,
    result: Result<T, CoreError>,
) -> napi::Result<T> {
    result.map_err(|error| {
        *kind_slot = Some(error.kind().as_str().to_string());
        Error::new(Status::GenericFailure, error.message().to_string())
    })
}

/// Rebuilds a task failure as an error carrying `code`.
///
/// Builds a real JavaScript `Error` object, sets `code` on it, and wraps it
/// back into a [`napi::Error`]; napi reuses that referenced object when it
/// rejects the promise, so the property survives. Throwing directly with
/// [`Env::throw_error`] does **not** work here - it raises an uncaught
/// exception instead of rejecting the promise, which the
/// "rejected promise carries the same code" test catches.
///
/// # Arguments
/// * `env` - The N-API environment.
/// * `kind` - The recorded error kind, if any.
/// * `error` - The error `compute` returned.
///
/// # Returns
/// The error to reject with: the original one if no kind was recorded or
/// the JavaScript object could not be built.
pub fn error_with_code(env: Env, kind: Option<&str>, error: Error) -> Error {
    let Some(kind) = kind else {
        return error;
    };

    let js_error = env.create_error(Error::new(Status::GenericFailure, error.reason.clone()));
    match js_error {
        Ok(mut object) => match object.set("code", kind) {
            Ok(()) => Error::from(object.to_unknown()),
            Err(_) => error,
        },
        Err(_) => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_carries_the_kind() {
        let error = to_napi_error(CoreError::new(ErrorKind::PosUnavailable, "no POS model"));
        assert_eq!(error.status, "pos_unavailable");
        assert!(error.reason.contains("no POS model"));
    }

    #[test]
    fn test_invalid_argument_helper() {
        assert_eq!(invalid_argument("bad language").status, "invalid_argument");
    }

    #[test]
    fn test_task_call_records_the_kind() {
        let mut kind = None;
        let result: napi::Result<()> =
            task_call(&mut kind, Err(CoreError::new(ErrorKind::Model, "wrong kind")));

        assert!(result.is_err());
        assert_eq!(kind.as_deref(), Some("model"));
        assert_eq!(result.unwrap_err().reason, "wrong kind");
    }

    #[test]
    fn test_task_call_passes_success_through() {
        let mut kind = None;
        let value = task_call(&mut kind, Ok::<_, CoreError>(42)).unwrap();
        assert_eq!(value, 42);
        assert_eq!(kind, None);
    }
}
