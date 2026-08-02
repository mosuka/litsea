//! Integration tests for the litsea CLI binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(name)
}

/// Regression test for #102: a downstream consumer closing stdout early
/// (e.g. `litsea segment model | head -1`) is normal termination for a
/// line-oriented CLI, not an error. The child's stdout is dropped
/// immediately and enough input is written to overflow the child's output
/// buffer, forcing a write into the closed pipe.
#[test]
fn test_broken_pipe_exits_zero() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_litsea"))
        .arg("segment")
        .arg("-l")
        .arg("japanese")
        .arg(model_path("RWCP.model"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn litsea");

    // Simulate the downstream consumer exiting immediately.
    drop(child.stdout.take());

    // Feed well over the 8 KiB BufWriter capacity so the child is forced to
    // write into the closed pipe. The child may exit mid-write, so stdin
    // write errors are expected and ignored.
    {
        let mut stdin = child.stdin.take().expect("stdin");
        for _ in 0..2000 {
            if writeln!(stdin, "これはテストです。").is_err() {
                break;
            }
        }
    }

    let output = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected exit 0 on broken pipe, got {:?} (stderr: {})",
        output.status,
        stderr
    );
    assert!(
        !stderr.contains("Broken pipe"),
        "broken pipe must not be reported as an error: {}",
        stderr
    );
}
