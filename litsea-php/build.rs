//! Build script required by ext-php-rs to locate the PHP headers, plus the
//! linker flags a PHP extension needs on macOS.

use std::env;

fn main() {
    // Re-run when the PHP installation changes; `ext-php-rs` reads the
    // headers through `php-config`.
    println!("cargo:rerun-if-env-changed=PHP");
    println!("cargo:rerun-if-env-changed=PHP_CONFIG");

    // A PHP extension references Zend symbols (`zval_ptr_dtor`, ...) that are
    // provided by the `php` executable at load time, not by any library the
    // cdylib links against. ELF linkers allow that by default; Apple's ld
    // rejects undefined symbols unless told to resolve them at load time,
    // so without these flags the macOS build fails with "Undefined symbols
    // for architecture arm64". The flags are emitted for the cdylib only,
    // the way PyO3 documents for Python extension modules; the rlib and the
    // unit-test binary are unaffected.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
