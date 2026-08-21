//! Cooperative cancellation for long-running training runs.
//!
//! `litsea`'s trainers take a `running: &AtomicBool` and stop early when it
//! is cleared, returning the metrics of the partially trained model (see
//! [`CancelToken`] for the exact semantics). The CLI drives that flag from a
//! `ctrlc` handler, but a library must not: `ctrlc::set_handler` is
//! process-global and can only be installed once, and a Python or Node.js
//! host normally owns SIGINT already. Bindings therefore hand the caller a
//! [`CancelToken`] and let the host decide what triggers it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shareable flag that asks a running training job to stop.
///
/// Cancelling is cooperative and **not** an error: the trainer finishes its
/// current unit of work, writes the partially trained model to the
/// destination path, and returns its metrics normally. The check happens
/// once per boosting iteration for AdaBoost training, and once per epoch and
/// per instance for perceptron training, so perceptron training reacts much
/// faster.
///
/// Clones share one flag, so a token handed to a background thread cancels
/// the training its sibling is driving.
#[derive(Debug, Clone)]
pub struct CancelToken {
    /// `true` while training should continue; matches `litsea`'s `running`
    /// flag so it can be passed straight through.
    running: Arc<AtomicBool>,
}

impl CancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The new [`CancelToken`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Requests cancellation.
    ///
    /// Training stops at its next check point and still saves the partially
    /// trained model. Calling this more than once is harmless.
    pub fn cancel(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Returns whether cancellation has been requested.
    ///
    /// # Returns
    /// `true` once [`CancelToken::cancel`] has been called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        !self.running.load(Ordering::SeqCst)
    }

    /// Returns the token to the "keep running" state so it can drive
    /// another training run.
    pub fn reset(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Returns the underlying flag in the form `litsea`'s trainers expect.
    ///
    /// # Returns
    /// A reference to the `running` flag: `true` means "keep going".
    #[must_use]
    pub fn running_flag(&self) -> &AtomicBool {
        &self.running
    }
}

impl Default for CancelToken {
    /// Creates a token in the "keep running" state.
    ///
    /// # Returns
    /// The same value as [`CancelToken::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        assert!(token.running_flag().load(Ordering::SeqCst));

        token.cancel();
        assert!(token.is_cancelled());
        assert!(!token.running_flag().load(Ordering::SeqCst));

        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_clones_share_one_flag() {
        let token = CancelToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(token.is_cancelled(), "cancelling a clone must cancel the original");
    }

    #[test]
    fn test_cancel_from_another_thread() {
        let token = CancelToken::new();
        let worker = token.clone();
        std::thread::spawn(move || worker.cancel()).join().unwrap();
        assert!(token.is_cancelled());
    }
}
