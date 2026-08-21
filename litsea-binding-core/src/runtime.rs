//! Blocking wrappers for `litsea`'s async model loading.
//!
//! The only async surface in `litsea` is `load_model(uri)` / the URI reader,
//! which exists so that `http(s)://` models can be fetched; the CLI runs
//! `#[tokio::main]` for that reason alone. Python, PHP, and Ruby have no
//! async story to hand that future to, so this module runs it to completion
//! on a private current-thread runtime.
//!
//! Bindings with a real event loop (Node.js, WASM) should use the `async`
//! functions directly instead.

use std::future::Future;

use crate::error::{CoreError, CoreResult};

/// Runs a future to completion on a private current-thread runtime.
///
/// # Arguments
/// * `future` - The future to drive.
///
/// # Returns
/// The future's output.
///
/// # Errors
/// Returns an [`crate::ErrorKind::Runtime`] error if called from inside an
/// existing Tokio runtime (blocking there would panic) or if the runtime
/// cannot be created.
pub fn block_on<F: Future>(future: F) -> CoreResult<F::Output> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(CoreError::runtime(
            "cannot block on a model download from inside an async runtime; use the async API instead",
        ));
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CoreError::runtime(format!("failed to create a Tokio runtime: {}", e)))?;

    Ok(runtime.block_on(future))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_on_runs_the_future() {
        let value = block_on(async { 6 * 7 }).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_block_on_inside_a_runtime_errors_instead_of_panicking() {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let error = runtime.block_on(async { block_on(async { 1 }).unwrap_err() });
        assert_eq!(error.kind(), crate::ErrorKind::Runtime);
        assert!(
            error.message().contains("async runtime"),
            "unexpected message: {}",
            error.message()
        );
    }
}
