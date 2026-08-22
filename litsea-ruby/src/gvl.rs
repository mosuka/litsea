//! Releasing the GVL around long-running work.
//!
//! Ruby is genuinely multi-threaded, so a `train()` that holds the Global VM
//! Lock blocks every other Ruby thread in the process - including the one
//! that would cancel it. Releasing the lock is what makes
//! `Litsea::CancelToken` usable while training runs, exactly as
//! `Python::detach` does for the Python binding.
//!
//! Neither magnus nor `rb-sys` exposes `rb_thread_call_without_gvl`: magnus
//! lists it among the C functions it does not wrap, and it is declared in
//! `ruby/thread.h`, which is outside the bindings `rb-sys` generates. The
//! declaration below is therefore written by hand. The symbol resolves
//! because the extension links against libruby regardless.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

unsafe extern "C" {
    /// Runs `func` with the GVL released.
    ///
    /// `ubf` is the "unblocking function" Ruby calls to interrupt the work;
    /// passing null selects Ruby's default, which defers interrupts until
    /// the call returns. That is what we want: the work is pure computation
    /// with no blocking syscall to interrupt, and it stops on its own when
    /// the cancellation flag is cleared.
    fn rb_thread_call_without_gvl(
        func: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        data1: *mut c_void,
        ubf: *const c_void,
        data2: *mut c_void,
    ) -> *mut c_void;
}

/// State handed to the trampoline and filled in with the closure's result.
struct Payload<F, R> {
    /// The closure to run, taken by the trampoline.
    func: Option<F>,
    /// Where the result lands; `None` if the closure panicked.
    result: Option<R>,
}

/// The `extern "C"` entry point Ruby calls with the GVL released.
///
/// # Safety
/// `data` must be a valid `*mut Payload<F, R>` that outlives the call, which
/// [`without_gvl`] guarantees by keeping the payload on its own stack frame
/// for the duration of `rb_thread_call_without_gvl`.
unsafe extern "C" fn trampoline<F, R>(data: *mut c_void) -> *mut c_void
where
    F: FnOnce() -> R,
{
    // SAFETY: `without_gvl` passes a pointer to a live `Payload<F, R>` and
    // does not touch it until this function returns.
    let payload = unsafe { &mut *(data as *mut Payload<F, R>) };

    if let Some(func) = payload.func.take() {
        // A panic must not unwind across the C frame Ruby put on the stack:
        // that is undefined behaviour. Catch it here and report it as a
        // missing result, which `without_gvl` turns back into a panic on the
        // Rust side of the boundary.
        payload.result = catch_unwind(AssertUnwindSafe(func)).ok();
    }

    std::ptr::null_mut()
}

/// Runs `func` with the GVL released, so other Ruby threads keep running.
///
/// The closure runs on the calling thread, not a new one. It **must not**
/// touch the Ruby API - doing so without the GVL is undefined behaviour.
/// Every caller in this crate passes pure Rust work (segmentation or
/// training) that only reaches `litsea`.
///
/// # Arguments
/// * `func` - The work to run without the GVL.
///
/// # Returns
/// Whatever `func` returns.
///
/// # Panics
/// Re-panics on the Rust side if `func` panicked, after the panic has been
/// contained inside the C call.
pub fn without_gvl<F, R>(func: F) -> R
where
    F: FnOnce() -> R,
{
    let mut payload = Payload {
        func: Some(func),
        result: None,
    };

    // SAFETY: `trampoline::<F, R>` matches the signature Ruby expects, and
    // `&mut payload` stays valid for the whole call because
    // `rb_thread_call_without_gvl` returns before this frame is dropped. A
    // null unblocking function selects Ruby's default deferred-interrupt
    // behaviour, which is correct for non-blocking computation.
    unsafe {
        rb_thread_call_without_gvl(
            trampoline::<F, R>,
            std::ptr::addr_of_mut!(payload) as *mut c_void,
            std::ptr::null(),
            std::ptr::null_mut(),
        );
    }

    match payload.result.take() {
        Some(result) => result,
        // The closure panicked; the panic was contained in the trampoline so
        // it could not unwind through C, and is re-raised here.
        None => panic!("a Litsea operation panicked while the GVL was released"),
    }
}
