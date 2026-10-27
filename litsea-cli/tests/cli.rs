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
        // A child that fails fast (e.g. a missing model) may exit before
        // reading stdin, closing the pipe; that BrokenPipe is part of the
        // scenario under test, not a harness error.
        if let Err(e) = handle.write_all(input.as_bytes()) {
            assert_eq!(e.kind(), std::io::ErrorKind::BrokenPipe, "write stdin: {e}");
        }
    }
    drop(child.stdin.take());
    child.wait_with_output().expect("wait")
}

/// Pins the plain word-segmentation output (same expectation as the golden
/// test suite) and the `--language` wiring.
#[test]
fn test_segment_golden_output() {
    let output = run_litsea(
        &["segment", "-l", "japanese", model_path("japanese.model").to_str().unwrap()],
        Some("これはテストです。\n"),
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "これ は テスト です 。\n");
}

/// Pins English word-segmentation output (same expectation as the golden
/// test suite) and the `-l english` wiring, including the space token.
#[test]
fn test_segment_english_golden_output() {
    let output = run_litsea(
        &["segment", "-l", "english", model_path("english.model").to_str().unwrap()],
        Some("I don't know.\n"),
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout), "I   do n't   know .\n");
}

/// Pins `--threads` (issue #185): parallel output must be byte-identical
/// to the single-threaded default, including trim/empty-line handling and
/// input order, for both the plain and the `--pos` pipelines.
#[test]
fn test_segment_threads_output_identical() {
    // Enough lines to split across several workers, with varied lengths,
    // empty lines, whitespace-only lines, and surrounding whitespace.
    // Every content line embeds its index, so any reordering (across
    // workers or within a worker) changes the output — a symmetric input
    // would let an order bug cancel out invisibly.
    let mut input = String::new();
    for i in 0..120 {
        match i % 5 {
            0 => input.push_str(&format!("これは{i}番目のテストです。\n")),
            1 => input.push_str(&format!("  価格は{i}円です。  \n")),
            2 => input.push('\n'),
            3 => input.push_str(&format!("東京都{i}に住んでいます。私の猫は可愛い。\n")),
            _ => input.push_str("   \n"),
        }
    }

    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();
    let sequential = run_litsea(&["segment", "-l", "japanese", model], Some(&input));
    assert!(sequential.status.success());
    for threads in ["2", "4"] {
        let parallel =
            run_litsea(&["segment", "--threads", threads, "-l", "japanese", model], Some(&input));
        assert!(parallel.status.success());
        assert_eq!(
            parallel.stdout, sequential.stdout,
            "--threads {threads} diverged from single-threaded output"
        );
    }

    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();
    let sequential = run_litsea(&["segment", "--pos", "-l", "japanese", pos_model], Some(&input));
    assert!(sequential.status.success());
    let parallel = run_litsea(
        &["segment", "--pos", "--threads", "4", "-l", "japanese", pos_model],
        Some(&input),
    );
    assert!(parallel.status.success());
    assert_eq!(parallel.stdout, sequential.stdout, "--pos --threads 4 diverged");
}

/// Pins `--threads` validation: zero is rejected at argument parsing.
#[test]
fn test_segment_threads_rejects_zero() {
    let output = run_litsea(
        &[
            "segment",
            "--threads",
            "0",
            "-l",
            "japanese",
            model_path("japanese.model").to_str().unwrap(),
        ],
        Some("これはテストです。\n"),
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--threads"), "unexpected stderr: {stderr}");
}

/// Pins segment's `--pos` routing: the two-stage model must be loaded and
/// word/POS pairs printed (same expectation as the golden suite).
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

/// Pins segment's `--pos` rejection of the removed joint model format: a
/// bare Averaged Perceptron file (class-count first line) must exit
/// non-zero with a migration hint, not be tagged with.
#[test]
fn test_segment_pos_rejects_joint_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir.path().join("joint.model");
    std::fs::write(&model, "2\nNOUN\nVERB\nUW4:x\tNOUN\t0.5\n").expect("write model");

    let output = run_litsea(
        &["segment", "--pos", "-l", "japanese", model.to_str().unwrap()],
        Some("テスト\n"),
    );
    assert!(!output.status.success(), "expected a joint-format model to be rejected");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no longer supported") && stderr.contains("train --pos"),
        "unexpected stderr: {stderr}"
    );
}

/// Pins issue #281: a byte order mark at the start of stdin is ignored on
/// every read path (plain and `--pos`, single-threaded and `--threads`),
/// and a BOM-only first line is skipped like an empty line. A U+FEFF
/// anywhere else, including a second one at the start, is read as text.
#[test]
fn test_segment_bom() {
    let model_owned = model_path("japanese.model");
    let pos_model_owned = model_path("japanese_pos.model");
    // Two lines, so that --threads 2 hands one to each worker.
    let input = "雪だるまを作る\n新幹線で帰る\n";
    for (pos, model) in [(false, &model_owned), (true, &pos_model_owned)] {
        for threads in ["1", "2"] {
            let mut args = vec!["segment", "--threads", threads, "-l", "japanese"];
            if pos {
                args.push("--pos");
            }
            args.push(model.to_str().unwrap());
            let segment = |input: &str| {
                let output = run_litsea(&args, Some(input));
                assert!(
                    output.status.success(),
                    "stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                String::from_utf8(output.stdout).unwrap()
            };
            let case = format!("--pos {pos}, --threads {threads}");

            let expected = segment(input);
            assert_eq!(segment(&format!("\u{FEFF}{input}")), expected, "{case}: BOM");
            assert_eq!(
                segment(&format!("\u{FEFF}\n{input}")),
                expected,
                "{case}: BOM-only first line"
            );

            let second_line = segment("雪だるまを作る\n\u{FEFF}新幹線で帰る\n");
            assert!(
                second_line.lines().nth(1).is_some_and(|line| line.starts_with('\u{FEFF}')),
                "{case}: U+FEFF at the start of the second line: {second_line:?}"
            );
            let second_bom = segment(&format!("\u{FEFF}\u{FEFF}{input}"));
            assert!(second_bom.starts_with('\u{FEFF}'), "{case}: second BOM: {second_bom:?}");
        }
    }
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

/// Pins extract's `--tag-free` routing (issue #183): the output must
/// contain no tag-dependent (UP*/BP*/UQ*/BQ*/TQ*) feature columns.
#[test]
fn test_extract_tag_free_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, "これ は テスト です 。\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "--tag-free",
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
    let tag_prefixes = ["UP", "BP", "UQ", "BQ", "TQ"];
    for line in content.lines() {
        let mut fields = line.split('\t');
        let label = fields.next().expect("label");
        assert!(label == "1" || label == "-1", "unexpected label {label:?} in {line:?}");
        let mut has_features = false;
        for feature in fields {
            has_features = true;
            let is_tag = tag_prefixes.iter().any(|p| {
                feature.starts_with(p)
                    && feature[p.len()..].starts_with(|c: char| c.is_ascii_digit())
            });
            assert!(!is_tag, "tag-dependent feature {feature:?} in {line:?}");
        }
        assert!(has_features, "line has no features: {line:?}");
    }
}

/// Pins the `--tag-free` conflict rule: it is a boundary-pipeline flag and
/// must be rejected together with --pos.
#[test]
fn test_extract_tag_free_rejects_pos() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, "これ/PRON は/ADP\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "--tag-free",
            "--pos",
            "-l",
            "japanese",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(!output.status.success(), "expected --tag-free --pos to fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot be combined"), "unexpected stderr: {stderr}");
}

/// Pins the removal of the old flag spelling: `--two-stage` is not accepted
/// (no alias), so clap rejects it at argument parsing.
#[test]
fn test_extract_rejects_removed_two_stage_flag() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus.txt");
    std::fs::write(&corpus, "これ/PRON は/ADP\n").expect("write corpus");
    let features = dir.path().join("features.txt");

    let output = run_litsea(
        &[
            "extract",
            "--two-stage",
            "-l",
            "japanese",
            corpus.to_str().unwrap(),
            features.to_str().unwrap(),
        ],
        None,
    );
    assert!(!output.status.success(), "expected --two-stage to be rejected");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unexpected argument"), "unexpected stderr: {stderr}");
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

/// Pins `extract --pos --format tsv` (issue #198): a space-preserving POS
/// corpus produces stage-1 rows covering the spaces, no stage-2 row for the
/// space token, and a lexicon entry recording it.
#[test]
fn test_extract_pos_tsv_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let corpus = dir.path().join("corpus_pos.tsv");
    std::fs::write(&corpus, "나는/PRON\t \t봄/NOUN\t./PUNCT\n").expect("write corpus");
    let prefix = dir.path().join("features");

    let output = run_litsea(
        &[
            "extract",
            "--pos",
            "-l",
            "korean",
            "--format",
            "tsv",
            corpus.to_str().unwrap(),
            prefix.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let stage1 = std::fs::read_to_string(dir.path().join("features.stage1")).expect("stage1");
    let stage2 = std::fs::read_to_string(dir.path().join("features.stage2")).expect("stage2");
    let lexicon = std::fs::read_to_string(dir.path().join("features.lexicon")).expect("lexicon");

    // "나는 봄." = 5 chars including the space; the POS pipeline emits a row
    // at every position including the first.
    assert_eq!(stage1.lines().count(), 5);
    // One stage-2 row per non-whitespace word (나는 / 봄 / .), not per token.
    assert_eq!(stage2.lines().count(), 3, "the space token must not get a stage-2 row");
    // The space is still recorded in the lexicon, as a single-candidate
    // entry so the packed model tags it without the classifier.
    assert!(
        lexicon.lines().any(|l| l.starts_with(" \t")),
        "lexicon should record the space surface: {lexicon:?}"
    );
}

/// Pins issue #281: a byte order mark at the start of the corpus is ignored
/// in every corpus format, so the output files equal those of the same
/// corpus without it.
#[test]
fn test_extract_bom() {
    let dir = tempfile::tempdir().expect("tempdir");
    let two_stage: &[&str] = &[".stage1", ".stage2", ".lexicon"];
    let cases: [(&[&str], &str, &[&str]); 4] = [
        (&["-l", "japanese"], "これ は テスト です 。\n", &[""]),
        (&["-l", "korean", "--format", "tsv"], "나는\t \t봄\t.\n", &[""]),
        (
            &["--pos", "-l", "japanese"],
            "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT\n",
            two_stage,
        ),
        (
            &["--pos", "-l", "korean", "--format", "tsv"],
            "나는/PRON\t \t봄/NOUN\t./PUNCT\n",
            two_stage,
        ),
    ];
    for (options, corpus, suffixes) in cases {
        let extract = |content: &str, name: &str| -> Vec<String> {
            let corpus_path = dir.path().join(format!("{name}.txt"));
            std::fs::write(&corpus_path, content).expect("write corpus");
            let out = dir.path().join(name);
            let mut args = vec!["extract"];
            args.extend_from_slice(options);
            args.push(corpus_path.to_str().unwrap());
            args.push(out.to_str().unwrap());
            let output = run_litsea(&args, None);
            assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
            suffixes
                .iter()
                .map(|suffix| {
                    std::fs::read_to_string(format!("{}{suffix}", out.display())).expect("read")
                })
                .collect()
        };
        let expected = extract(corpus, "plain");
        assert!(expected.iter().all(|file| !file.is_empty()), "{options:?}: empty output");
        assert_eq!(extract(&format!("\u{FEFF}{corpus}"), "bom"), expected, "{options:?}");
    }
}

/// Pins `--user-words` (issue #263): a registered word comes out as one
/// token on the plain, `--threads`, and `--pos` paths; the file format
/// skips empty lines, accepts CRLF, and ignores one-character words; and
/// `--user-words-gate 0` applies a word only where the model already
/// agrees.
#[test]
fn test_segment_user_words() {
    let dir = tempfile::tempdir().expect("tempdir");
    let words = dir.path().join("words.txt");
    std::fs::write(&words, "雪だるま\n\n新幹線\r\nあ\n").expect("write words");
    let words = words.to_str().unwrap();
    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();
    let input = "雪だるまを作る\n新幹線で帰る\n";

    let plain = run_litsea(&["segment", "-l", "japanese", model], Some(input));
    assert!(plain.status.success(), "stderr: {}", String::from_utf8_lossy(&plain.stderr));
    assert_eq!(String::from_utf8_lossy(&plain.stdout), "雪 だる ま を 作る\n新 幹線 で 帰る\n");

    let with_words =
        run_litsea(&["segment", "--user-words", words, "-l", "japanese", model], Some(input));
    assert!(
        with_words.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&with_words.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&with_words.stdout),
        "雪だるま を 作る\n新幹線 で 帰る\n"
    );

    let parallel = run_litsea(
        &["segment", "--threads", "2", "--user-words", words, "-l", "japanese", model],
        Some(input),
    );
    assert!(parallel.status.success());
    assert_eq!(parallel.stdout, with_words.stdout, "--threads diverged with a dictionary");

    let gated = run_litsea(
        &[
            "segment",
            "--user-words",
            words,
            "--user-words-gate",
            "0",
            "-l",
            "japanese",
            model,
        ],
        Some(input),
    );
    assert!(gated.status.success(), "stderr: {}", String::from_utf8_lossy(&gated.stderr));
    assert_eq!(gated.stdout, plain.stdout, "gate 0 must leave the model's output alone");

    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();
    let pos = run_litsea(
        &["segment", "--pos", "--user-words", words, "-l", "japanese", pos_model],
        Some("雪だるまを作る\n"),
    );
    assert!(pos.status.success(), "stderr: {}", String::from_utf8_lossy(&pos.stderr));
    let stdout = String::from_utf8_lossy(&pos.stdout);
    assert!(stdout.starts_with("雪だるま/"), "unexpected output: {stdout}");
}

/// Pins `--user-words` error handling: a gate without a dictionary is an
/// argument error, a gate outside 0..=1 and a missing file are reported
/// with their cause.
#[test]
fn test_segment_user_words_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let words = dir.path().join("words.txt");
    std::fs::write(&words, "雪だるま\n").expect("write words");
    let words = words.to_str().unwrap();
    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();

    let output = run_litsea(
        &["segment", "--user-words-gate", "0.5", "-l", "japanese", model],
        Some("雪だるまを作る\n"),
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--user-words"), "unexpected stderr: {stderr}");

    let output = run_litsea(
        &[
            "segment",
            "--user-words",
            words,
            "--user-words-gate",
            "1.5",
            "-l",
            "japanese",
            model,
        ],
        Some("雪だるまを作る\n"),
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("0.0..=1.0"), "unexpected stderr: {stderr}");

    let output = run_litsea(
        &["segment", "--user-words", "/nonexistent/words.txt", "-l", "japanese", model],
        Some("雪だるまを作る\n"),
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot read user dictionary"), "unexpected stderr: {stderr}");
}

/// Pins `evaluate --user-words` (issue #263): the dictionary is applied
/// during evaluation, so a word the model splits scores 100% once
/// registered.
#[test]
fn test_evaluate_user_words() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold.txt");
    std::fs::write(&gold, "雪だるま を 作る\n").expect("write gold");
    let gold = gold.to_str().unwrap();
    let words = dir.path().join("words.txt");
    std::fs::write(&words, "雪だるま\n").expect("write words");
    let words = words.to_str().unwrap();
    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();

    let plain = run_litsea(&["evaluate", "-l", "japanese", model, gold], None);
    assert!(plain.status.success(), "stderr: {}", String::from_utf8_lossy(&plain.stderr));
    let stderr = String::from_utf8_lossy(&plain.stderr);
    assert!(!stderr.contains("Word F1: 100.00%"), "unexpected output: {stderr}");

    let with_words =
        run_litsea(&["evaluate", "--user-words", words, "-l", "japanese", model, gold], None);
    assert!(
        with_words.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&with_words.stderr)
    );
    let stderr = String::from_utf8_lossy(&with_words.stderr);
    assert!(stderr.contains("Word F1: 100.00%"), "unexpected output: {stderr}");

    let gated = run_litsea(
        &[
            "evaluate",
            "--user-words",
            words,
            "--user-words-gate",
            "0",
            "-l",
            "japanese",
            model,
            gold,
        ],
        None,
    );
    assert!(gated.status.success(), "stderr: {}", String::from_utf8_lossy(&gated.stderr));
    assert_eq!(gated.stderr, plain.stderr, "gate 0 must reproduce the plain metrics");
}

/// Pins tagged `--user-words` entries (issue #264): with `--pos`, a token
/// that is a tagged word gets the registered tag and every other token
/// keeps the tagger's; without `--pos` the tags change nothing. The file
/// accepts a trailing space or CR after the tag, keeps a tag across an
/// untagged repetition of the word, and ignores a tagged one-character
/// word.
#[test]
fn test_segment_user_words_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let untagged = dir.path().join("untagged.txt");
    std::fs::write(&untagged, "ポケモン\nチャットGPT\n雪だるま\n").expect("write words");
    let untagged = untagged.to_str().unwrap();
    let tagged = dir.path().join("tagged.txt");
    std::fs::write(
        &tagged,
        "ポケモン\tPROPN\nチャットGPT\tPROPN \r\n雪だるま\n猫\tNOUN\nポケモン\n",
    )
    .expect("write words");
    let tagged = tagged.to_str().unwrap();
    let input = "ポケモンを集める\nチャットGPTに聞く\n雪だるまを作る\n猫が好き\n";
    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();

    let run_pos = |words: &str, extra: &[&str]| {
        let mut args = vec!["segment", "--pos", "--user-words", words];
        args.extend_from_slice(extra);
        args.extend_from_slice(&["-l", "japanese", pos_model]);
        let output = run_litsea(&args, Some(input));
        assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
        String::from_utf8(output.stdout).unwrap()
    };
    let baseline = run_pos(untagged, &[]);
    // The tagger's own tags, so the override below is visible.
    assert!(baseline.contains("ポケモン/NOUN"), "unexpected output: {baseline}");
    assert!(baseline.contains("チャットGPT/NOUN"), "unexpected output: {baseline}");
    let expected = baseline
        .replace("ポケモン/NOUN", "ポケモン/PROPN")
        .replace("チャットGPT/NOUN", "チャットGPT/PROPN");
    assert_eq!(run_pos(tagged, &[]), expected);
    assert_eq!(run_pos(tagged, &["--threads", "2"]), expected, "--threads diverged with tags");

    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();
    let run_plain = |words: &str| {
        run_litsea(&["segment", "--user-words", words, "-l", "japanese", model], Some(input))
    };
    let plain_untagged = run_plain(untagged);
    let plain_tagged = run_plain(tagged);
    assert!(
        plain_tagged.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&plain_tagged.stderr)
    );
    assert_eq!(plain_tagged.stdout, plain_untagged.stdout, "tags must not change segmentation");
}

/// Pins the error cases of tagged `--user-words` entries (issue #264): each
/// names the file's line and the offending part, and the file is validated
/// with or without `--pos` (a one-character word's tag included).
#[test]
fn test_segment_user_words_tag_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();
    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();
    let cases: [(&str, &[&str]); 6] = [
        (
            "雪だるま\nポケモン\tNoun\n",
            &["line 2", "unknown UPOS tag \"Noun\"", "expected one of ADJ, ADP"],
        ),
        ("猫\tCAT\n", &["line 1", "unknown UPOS tag \"CAT\""]),
        ("\tNOUN\n", &["line 1", "empty word before the TAB"]),
        ("ポケモン\tPROPN\tX\n", &["line 1", "too many columns"]),
        ("ポケモン\t \n", &["line 1", "empty UPOS tag after \"ポケモン\""]),
        (
            "ポケモン\tPROPN\n\nポケモン\tNOUN\n",
            &["line 3", "\"ポケモン\" is tagged NOUN here but PROPN on line 1"],
        ),
    ];
    for (index, (content, needles)) in cases.iter().enumerate() {
        let words = dir.path().join(format!("words{index}.txt"));
        std::fs::write(&words, content).expect("write words");
        let words = words.to_str().unwrap();
        for args in [
            vec!["segment", "--user-words", words, "-l", "japanese", model],
            vec!["segment", "--pos", "--user-words", words, "-l", "japanese", pos_model],
        ] {
            let output = run_litsea(&args, Some("ポケモンを集める\n"));
            assert!(!output.status.success(), "{content:?} with {args:?} succeeded");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("user dictionary"), "{content:?}: {stderr}");
            for needle in *needles {
                assert!(stderr.contains(needle), "{content:?}: missing {needle:?} in {stderr}");
            }
        }
    }
}

/// Pins the byte order mark rule of `--user-words` (issue #275): a U+FEFF at
/// the start of the file is dropped, so a file saved as "UTF-8 with BOM"
/// loads the same dictionary as without it, whether its first line is
/// untagged or tagged, and a BOM-only first line counts as empty. A U+FEFF
/// anywhere else stays part of its word, which then never matches.
#[test]
fn test_segment_user_words_bom() {
    let dir = tempfile::tempdir().expect("tempdir");
    let words_owned = dir.path().join("words.txt");
    let words = words_owned.to_str().unwrap();
    let model_owned = model_path("japanese.model");
    let model = model_owned.to_str().unwrap();
    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();
    let segment = |content: &str, pos: bool| {
        std::fs::write(words, content).expect("write words");
        let (args, input) = if pos {
            (
                vec!["segment", "--pos", "--user-words", words, "-l", "japanese", pos_model],
                "チャットGPTに聞く\n",
            )
        } else {
            (
                vec!["segment", "--user-words", words, "-l", "japanese", model],
                "雪だるまを作る\n新幹線で帰る\n",
            )
        };
        let output = run_litsea(&args, Some(input));
        assert!(
            output.status.success(),
            "{content:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };

    let plain = segment("雪だるま\n新幹線\n", false);
    assert_eq!(plain, "雪だるま を 作る\n新幹線 で 帰る\n");
    assert_eq!(segment("\u{FEFF}雪だるま\n新幹線\n", false), plain);
    assert_eq!(segment("\u{FEFF}\n雪だるま\n新幹線\n", false), plain, "BOM-only first line");

    let tagged = segment("チャットGPT\tPROPN\n", true);
    assert_eq!(tagged, "チャットGPT/PROPN に/ADP 聞く/VERB\n");
    assert_eq!(segment("\u{FEFF}チャットGPT\tPROPN\n", true), tagged);

    assert_eq!(
        segment("雪だるま\n\u{FEFF}新幹線\n", false),
        "雪だるま を 作る\n新 幹線 で 帰る\n"
    );
    assert_eq!(
        segment("\u{FEFF}\u{FEFF}雪だるま\n新幹線\n", false),
        "雪 だる ま を 作る\n新幹線 で 帰る\n",
        "only one BOM is dropped"
    );
}

/// Pins `evaluate --pos --user-words` with tags (issue #264): the tag fixes
/// the tagged metrics while the segmentation metrics stay as they are.
#[test]
fn test_evaluate_user_words_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold.txt");
    std::fs::write(&gold, "ポケモン/PROPN を/ADP 集める/VERB\n").expect("write gold");
    let gold = gold.to_str().unwrap();
    let untagged = dir.path().join("untagged.txt");
    std::fs::write(&untagged, "ポケモン\n").expect("write words");
    let tagged = dir.path().join("tagged.txt");
    std::fs::write(&tagged, "ポケモン\tPROPN\n").expect("write words");
    let pos_model_owned = model_path("japanese_pos.model");
    let pos_model = pos_model_owned.to_str().unwrap();

    let evaluate = |words: &PathBuf| {
        let output = run_litsea(
            &[
                "evaluate",
                "--pos",
                "--user-words",
                words.to_str().unwrap(),
                "-l",
                "japanese",
                pos_model,
                gold,
            ],
            None,
        );
        assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
        String::from_utf8(output.stderr).unwrap()
    };
    let without_tag = evaluate(&untagged);
    assert!(without_tag.contains("  Word F1: 100.00%"), "unexpected output: {without_tag}");
    assert!(
        !without_tag.contains("Tagged Word F1: 100.00%"),
        "unexpected output: {without_tag}"
    );
    let with_tag = evaluate(&tagged);
    assert!(with_tag.contains("  Word F1: 100.00%"), "unexpected output: {with_tag}");
    assert!(with_tag.contains("Tagged Word F1: 100.00%"), "unexpected output: {with_tag}");
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

/// Pins evaluate's `--pos` routing: POS gold + two-stage model prints the
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

/// Pins evaluate's `--pos --format tsv` routing (issue #196): a
/// space-preserving POS gold, for measuring real-world (spaced) quality
/// instead of the unspaced protocol `--pos` alone uses. The gold line is
/// english_pos.model's own pinned real-spaced-input output for "I don't
/// know." (golden.rs's golden_segment_with_pos_english_two_stage), so this
/// scores 100% by construction; the space tokens carry no /POS suffix and
/// are excluded from scoring regardless.
#[test]
fn test_evaluate_pos_tsv_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold = dir.path().join("gold_pos.tsv");
    std::fs::write(&gold, "I/PRON\t \tdo/AUX\tn't/PART\t \tknow/VERB\t./PUNCT\n")
        .expect("write gold");

    let output = run_litsea(
        &[
            "evaluate",
            "--pos",
            "--format",
            "tsv",
            "-l",
            "english",
            model_path("english_pos.model").to_str().unwrap(),
            gold.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Evaluation Metrics (POS):"), "unexpected output: {stderr}");
    assert!(stderr.contains("Word F1: 100.00%"), "unexpected output: {stderr}");
    assert!(stderr.contains("Tagged Word F1: 100.00%"), "unexpected output: {stderr}");
}

/// Pins issue #281: a byte order mark at the start of the gold corpus is
/// ignored in every gold format, while a U+FEFF at the start of a later
/// line, or a second one at the start, is read as text and changes the
/// metrics.
#[test]
fn test_evaluate_bom() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gold_owned = dir.path().join("gold.txt");
    let gold = gold_owned.to_str().unwrap();
    // The golds of the tests above, which each model reproduces exactly.
    let cases: [(&[&str], &str, &str); 4] = [
        (&["-l", "japanese"], "japanese.model", "これ は テスト です 。"),
        (
            &["-l", "korean", "--format", "tsv"],
            "korean.model",
            "이것은\t \t테스트입니다\t.",
        ),
        (
            &["--pos", "-l", "japanese"],
            "japanese_pos.model",
            "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT",
        ),
        (
            &["--pos", "--format", "tsv", "-l", "english"],
            "english_pos.model",
            "I/PRON\t \tdo/AUX\tn't/PART\t \tknow/VERB\t./PUNCT",
        ),
    ];
    for (options, model, sentence) in cases {
        let model_owned = model_path(model);
        let mut args = vec!["evaluate"];
        args.extend_from_slice(options);
        args.push(model_owned.to_str().unwrap());
        args.push(gold);
        let evaluate = |content: String| {
            std::fs::write(&gold_owned, content).expect("write gold");
            let output = run_litsea(&args, None);
            assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8(output.stderr).unwrap()
        };

        let expected = evaluate(format!("{sentence}\n{sentence}\n"));
        assert!(expected.contains("  Word F1: 100.00%"), "{model}: {expected}");
        assert_eq!(evaluate(format!("\u{FEFF}{sentence}\n{sentence}\n")), expected, "{model}");
        assert_ne!(
            evaluate(format!("{sentence}\n\u{FEFF}{sentence}\n")),
            expected,
            "{model}: U+FEFF at the start of the second line"
        );
        assert_ne!(
            evaluate(format!("\u{FEFF}\u{FEFF}{sentence}\n{sentence}\n")),
            expected,
            "{model}: second BOM"
        );
    }
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

    // Generic Averaged Perceptron training routes to the other learner
    // (labels are opaque strings; B/O mirrors the collapse recipe).
    let perceptron_features = dir.path().join("perceptron_features.txt");
    std::fs::write(&perceptron_features, "B\tf1\nO\tf2\nB\tf3\nO\tf4\n")
        .expect("write perceptron features");
    let perceptron_model = dir.path().join("out_perceptron.model");
    let output = run_litsea(
        &[
            "train",
            "--perceptron",
            "--num-epochs",
            "3",
            perceptron_features.to_str().unwrap(),
            perceptron_model.to_str().unwrap(),
        ],
        None,
    );
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Result Metrics (Perceptron)"));
    assert!(perceptron_model.exists() && std::fs::metadata(&perceptron_model).unwrap().len() > 0);
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
        &["segment", "-l", "french", model_path("japanese.model").to_str().unwrap()],
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
        .arg(model_path("japanese.model"))
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
