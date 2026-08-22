//! Build script required by ext-php-rs to locate the PHP headers.

fn main() {
    // Re-run when the PHP installation changes; `ext-php-rs` reads the
    // headers through `php-config`.
    println!("cargo:rerun-if-env-changed=PHP");
    println!("cargo:rerun-if-env-changed=PHP_CONFIG");
}
