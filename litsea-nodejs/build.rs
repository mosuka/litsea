//! Build script required by napi-rs to emit the Node.js addon linkage.

extern crate napi_build;

/// Configures the napi-rs build.
fn main() {
    napi_build::setup();
}
