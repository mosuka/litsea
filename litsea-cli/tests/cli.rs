//! Integration tests for the litsea CLI binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(name)
}

/// Runs the litsea binary with `args`, feeding `stdin` (if any), and returns
/// the collected output.
fn run_litsea(args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_litsea"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn litsea");
    if let Some(input) = stdin {
        let mut handle = child.stdin.take().expect("stdin");
        handle.write_all(input.as_bytes()).expect("write stdin");
    }
    drop(child.stdin.take());
    child.wait_with_output().expect("wait")
}

/// Pins the plain word-segmentation output (same expectation as the golden
/// test suite) and the `--language` wiring.
#[test]
fn test_segment_golden_output() {
    let output = run_litsea(
        &["segment", "-l", "japanese", model_path("RWCP.model").to_str().unwrap()],
        Some("これはテストです。\n"),
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "これ は テスト です 。\n");
}

/// Pins segment's `--pos` routing: the Averaged Perceptron model must be
/// selected and word/POS pairs printed (same expectation as the golden suite).
#[test]
fn test_segment_pos_golden_output() {
    let output = run_litsea(
        &[
            "segment",
            "--pos",
            "-l",
            "japanese",
            model_path("japanese_pos.model").to_str().unwrap(),
        ],
        Some("これはテストです。\n"),
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT\n"
    );
}

/// Pins the extract output format: each line is a boundary label (1 / -1)
/// followed by tab-separated features.
#[test]
fn test_extract_features_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, "これ は テスト です 。\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "-l",
            "japanese",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let content = std::fs::read_to_string(&features).expect("read features");
    assert!(!content.is_empty());
    for line in content.lines() {
        let mut fields = line.split('\t');
        let label = fields.next().expect("label");
        assert!(label == "1" || label == "-1", "unexpected label {label:?} in {line:?}");
        assert!(fields.next().is_some(), "line has no features: {line:?}");
    }
}

/// Pins extract's `--format tsv` routing: tab-separated tokens with a
/// literal space token are accepted, and the preserved space shows up
/// inside character-context features (issue #152).
#[test]
fn test_extract_tsv_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.tsv");
    std::fs::write(&corpus, "나는\t \t봄\t.\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "-l",
            "korean",
            "--format",
            "tsv",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let content = std::fs::read_to_string(&features).expect("read features");
    // "나는 봄." = 5 chars -> 4 rows (first position skipped).
    assert_eq!(content.lines().count(), 4);
    let has_space_feature = content
        .lines()
        .flat_map(|l| l.split('\t').skip(1))
        .any(|f| f.starts_with("UW") && f.ends_with(' '));
    assert!(has_space_feature, "expected a UW feature containing the space character");
}

/// Pins the `--pos --format tsv` combination as an explicit error: the POS
/// pipeline has no TSV variant.
#[test]
fn test_extract_pos_rejects_tsv_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.tsv");
    std::fs::write(&corpus, "나는\t \t봄\t.\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "--pos",
            "-l",
            "korean",
            "--format",
            "tsv",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(!output.status.success(), "expected --pos --format tsv to fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not supported"), "unexpected stderr: {stderr}");
}

/// Pins the evaluate subcommand: known model + tiny gold corpus must print
/// the metrics block with plausible percentages.
#[test]
fn test_evaluate_segmentation_output() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold.txt");
    std::fs::write(&gold, "これ は テスト です 。\n").expect("write gold");

    let output = run_litsea(
        &[
            "evaluate",
            "-l",
            "japanese",
            model_path("japanese.model").to_str().unwrap(),
            gold.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Evaluation Metrics:"), "unexpected output: {stderr}");
    assert!(stderr.contains("Sentences: 1"), "unexpected output: {stderr}");
    // japanese.model segments this sentence exactly (golden test), so the
    // tiny corpus scores 100%.
    assert!(stderr.contains("Word F1: 100.00%"), "unexpected output: {stderr}");
}

/// Pins evaluate's `--format tsv` routing: a space token in the gold TSV is
/// excluded from scoring but preserved in the reconstructed text.
#[test]
fn test_evaluate_tsv_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold.tsv");
    std::fs::write(&gold, "이것은\t \t테스트입니다\t.\n").expect("write gold");

    let output = run_litsea(
        &[
            "evaluate",
            "-l",
            "korean",
            "--format",
            "tsv",
            model_path("korean.model").to_str().unwrap(),
            gold.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    // korean.model reproduces this segmentation exactly (golden test).
    assert!(stderr.contains("Word F1: 100.00%"), "unexpected output: {stderr}");
}

/// Pins evaluate's `--pos` routing: POS gold + perceptron model prints the
/// tagged-word metrics block.
#[test]
fn test_evaluate_pos_output() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold_pos.txt");
    std::fs::write(&gold, "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT\n").expect("write gold");

    let output = run_litsea(
        &[
            "evaluate",
            "--pos",
            "-l",
            "japanese",
            model_path("japanese_pos.model").to_str().unwrap(),
            gold.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Evaluation Metrics (POS):"), "unexpected output: {stderr}");
    // japanese_pos.model reproduces this tagging exactly (golden test).
    assert!(stderr.contains("Tagged Word F1: 100.00%"), "unexpected output: {stderr}");
}

/// Pins extract's `--pos` routing: labels become SegmentLabel strings
/// (O / B-<UPOS>) instead of boundary labels.
#[test]
fn test_extract_pos_features_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, "これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT\n")
        .expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "--pos",
            "-l",
            "japanese",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let content = std::fs::read_to_string(&features).expect("read features");
    assert!(!content.is_empty());
    let mut boundary_labels = 0;
    for line in content.lines() {
        let label = line.split('\t').next().expect("label");
        assert!(
            label == "O" || label.starts_with("B-"),
            "unexpected POS label {label:?} in {line:?}"
        );
        if label.starts_with("B-") {
            boundary_labels += 1;
        }
    }
    assert!(boundary_labels > 0, "expected at least one B-<POS> label");
}

/// Smoke-tests both train modes end to end: metrics are reported on stderr
/// and a model file is written.
#[test]
fn test_train_smoke() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Plain AdaBoost training on separable data.
    let features = dir.path().join("features.txt");
    std::fs::write(&features, "1\tfa\n-1\tfb\n1\tfa\n-1\tfb\n1\tfa\n-1\tfb\n")
        .expect("write features");
    let model = dir.path().join("out.model");
    let output = run_litsea(&["train", features.to_str().unwrap(), model.to_str().unwrap()], None);
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Result Metrics"));
    assert!(model.exists() && std::fs::metadata(&model).unwrap().len() > 0);

    // POS (Averaged Perceptron) training routes to the other learner.
    let pos_features = dir.path().join("pos_features.txt");
    std::fs::write(&pos_features, "B-NOUN\tf1\nO\tf2\nB-VERB\tf3\nO\tf4\n")
        .expect("write pos features");
    let pos_model = dir.path().join("out_pos.model");
    let output = run_litsea(
        &[
            "train",
            "--pos",
            "--num-epochs",
            "3",
            pos_features.to_str().unwrap(),
            pos_model.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Result Metrics (POS)"));
    assert!(pos_model.exists() && std::fs::metadata(&pos_model).unwrap().len() > 0);
}

/// A missing model path must exit non-zero with an `Error:` line on stderr.
#[test]
fn test_missing_model_error() {
    let output = run_litsea(&["segment", "-l", "japanese", "/nonexistent/path.model"], Some("x\n"));
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Error:"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// An unsupported language must be rejected with the FromStr message.
#[test]
fn test_unsupported_language_error() {
    let output = run_litsea(
        &["segment", "-l", "french", model_path("RWCP.model").to_str().unwrap()],
        Some("x\n"),
    );
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Unsupported language: 'french'"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
