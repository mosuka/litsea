//! Criterion benchmarks for the segmentation and POS-tagging hot paths.
//!
//! Covers short- and long-text `segment()` micro-benches
//! ([`bench_segment_short`]/[`bench_segment_long`]), the `external_corpus`
//! throughput group that mirrors the external tokenizer-speed-bench harness
//! ([`bench_external_corpus`]; see `docs/src/advanced/benchmarking.md`), and
//! a handful of internal component benchmarks (character-type
//! classification, corpus ingestion, single-instance AdaBoost prediction).

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::{SegmentBuffer, Segmenter};
use litsea::two_stage::TwoStageLearner;

/// Load an AdaBoost model file from the models directory.
fn load_adaboost_model(model_name: &str) -> AdaBoost {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model_name);
    let mut learner = AdaBoost::new(0.01, 100);
    learner
        .load_model_from_path(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
    learner
}

/// Load a two-stage model file from the models directory.
fn load_two_stage_model(model_name: &str) -> TwoStageLearner {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model_name);
    let mut learner = TwoStageLearner::new();
    learner
        .load_model_from_path(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
    learner
}

// ---------------------------------------------------------------------------
// Segmentation micro-benchmarks
// ---------------------------------------------------------------------------

/// Benchmarks AdaBoost-format `segment()` on a short sentence for each
/// language.
fn bench_segment_short(c: &mut Criterion) {
    let cases: &[(&str, Language, &str)] = &[
        ("japanese", Language::Japanese, "japanese.model"),
        ("chinese", Language::Chinese, "chinese.model"),
        ("korean", Language::Korean, "korean.model"),
        ("english", Language::English, "english.model"),
    ];

    let inputs: &[(&str, &str)] = &[
        ("japanese", "これはテストです。"),
        ("chinese", "这是一个测试。"),
        ("korean", "이것은테스트입니다."),
        ("english", "This is a test."),
    ];

    let mut group = c.benchmark_group("segment_short");

    for (lang, language, ada_model) in cases {
        let input = inputs.iter().find(|(l, _)| l == lang).unwrap().1;

        let ada_learner = load_adaboost_model(ada_model);
        let ada_segmenter = Segmenter::with_learner(*language, ada_learner);
        group.bench_with_input(BenchmarkId::new("adaboost", lang), &input, |b, &text| {
            b.iter(|| black_box(ada_segmenter.segment(black_box(text))));
        });
    }

    group.finish();
}

/// Benchmarks AdaBoost-format `segment()` on a long text (bocchan.txt) for
/// Japanese.
fn bench_segment_long(c: &mut Criterion) {
    let text_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../resources")
        .join("bocchan.txt");
    let text = fs::read_to_string(&text_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", text_path.display(), e));
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

    let mut group = c.benchmark_group("segment_long_japanese");

    let ada_learner = load_adaboost_model("japanese.model");
    let ada_segmenter = Segmenter::with_learner(Language::Japanese, ada_learner);
    group.bench_function("adaboost", |b| {
        b.iter(|| {
            for line in &lines {
                black_box(ada_segmenter.segment(black_box(line)));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// External-corpus throughput benchmarks (tokenizer-speed-bench parity)
// ---------------------------------------------------------------------------

/// Reads a benchmark corpus as unfiltered lines (no trimming, no empty-line
/// filtering — the same workload as tokenizer-speed-bench) and returns the
/// lines together with their total character count (newline-free, the same
/// chars/sec definition as the external harness).
fn load_corpus_lines(corpus_name: &str) -> (Vec<String>, u64) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources").join(corpus_name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    let chars = lines.iter().map(|l| l.chars().count() as u64).sum();
    (lines, chars)
}

/// Runs seven benches: the four segmentation cases of the external
/// tokenizer-speed-bench harness, reproduced in-repo, plus three
/// `*-two-stage` POS cases (issue #169).
/// One iteration segments every line of the corpus, and criterion's
/// `Throughput::Elements` makes the report read as chars/sec. Methodology
/// differences from the external harness (criterion sampling instead of
/// process-interleaved single passes; the workspace's tuned release
/// profile) are documented in `docs/src/advanced/benchmarking.md`.
fn bench_external_corpus(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_corpus");
    group.sample_size(30);

    // (bench id, language, AdaBoost-format model, corpus)
    let segment_cases: &[(&str, Language, &str, &str)] = &[
        ("japanese", Language::Japanese, "japanese.model", "wagahaiwa_nekodearu.txt"),
        ("japanese-rwcp", Language::Japanese, "RWCP.model", "wagahaiwa_nekodearu.txt"),
        ("korean", Language::Korean, "korean.model", "mujeong.txt"),
        ("chinese", Language::Chinese, "chinese.model", "rulin_waishi.txt"),
        ("english", Language::English, "english.model", "pride_and_prejudice.txt"),
    ];
    for (id, language, model, corpus) in segment_cases {
        let (lines, chars) = load_corpus_lines(corpus);
        let segmenter = Segmenter::with_learner(*language, load_adaboost_model(model));
        group.throughput(Throughput::Elements(chars));
        group.bench_function(*id, |b| {
            b.iter(|| {
                for line in &lines {
                    black_box(segmenter.segment(black_box(line)));
                }
            });
        });
    }

    // (bench id, language, two-stage model, corpus) -- issue #147/#169
    let two_stage_cases: &[(&str, Language, &str, &str)] = &[
        (
            "japanese-two-stage",
            Language::Japanese,
            "japanese_pos.model",
            "wagahaiwa_nekodearu.txt",
        ),
        ("korean-two-stage", Language::Korean, "korean_pos.model", "mujeong.txt"),
        ("chinese-two-stage", Language::Chinese, "chinese_pos.model", "rulin_waishi.txt"),
        (
            "english-two-stage",
            Language::English,
            "english_pos.model",
            "pride_and_prejudice.txt",
        ),
    ];
    for (id, language, model, corpus) in two_stage_cases {
        let (lines, chars) = load_corpus_lines(corpus);
        let segmenter = Segmenter::with_two_stage_learner(*language, load_two_stage_model(model));
        group.throughput(Throughput::Elements(chars));
        group.bench_function(*id, |b| {
            b.iter(|| {
                for line in &lines {
                    black_box(segmenter.segment_with_pos(black_box(line)).unwrap());
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Internal component benchmarks
// ---------------------------------------------------------------------------

/// Benchmarks single-character type classification (Japanese Hiragana).
fn bench_char_type(c: &mut Criterion) {
    let segmenter = Segmenter::new(Language::Japanese);
    c.bench_function("char_type_hiragana", |b| {
        b.iter(|| black_box(segmenter.char_type(black_box('あ'))));
    });
}

/// Benchmarks adding one short sentence's training instances to a fresh
/// segmenter's AdaBoost learner.
fn bench_add_corpus(c: &mut Criterion) {
    c.bench_function("add_corpus", |b| {
        b.iter_batched(
            || Segmenter::new(Language::Japanese),
            |mut segmenter| segmenter.add_corpus(black_box("これ は テスト です 。")),
            criterion::BatchSize::SmallInput,
        );
    });
}

/// Benchmarks a single string-keyed [`litsea::adaboost::AdaBoost::predict`]
/// call against a realistic attribute set (the test-only reference path, not
/// the packed `segment()` hot path).
fn bench_predict_adaboost(c: &mut Criterion) {
    let learner = load_adaboost_model("japanese.model");
    let segmenter = Segmenter::with_learner(Language::Japanese, learner);

    // Capture a realistic attribute set from the corpus pipeline.
    let mut attrs = None;
    segmenter.add_corpus_with_writer("テスト です", |a, _| {
        if attrs.is_none() {
            attrs = Some(a);
        }
    });
    let attrs = attrs.expect("corpus should produce at least one attribute set");

    c.bench_function("predict_adaboost", |b| {
        b.iter(|| segmenter.learner().predict(black_box(&attrs)));
    });
}

/// Paired comparison of the owned-output `segment()` API against the
/// buffer-reusing `segment_into()` API (issue #184) on the same three
/// segmentation workloads as `external_corpus` (same corpora, same
/// per-line iteration, chars/sec via `Throughput::Elements`). The
/// `*-strings` ids are the `segment()` baseline; the `*-ranges` ids reuse
/// one `SegmentBuffer` across the whole corpus, so their difference is the
/// per-call allocation cost the new API removes. Compare ids within one
/// run (same machine state), per the paired methodology in
/// `docs/src/advanced/benchmarking.md`.
fn bench_segment_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_into");
    group.sample_size(30);

    let cases: &[(&str, Language, &str, &str)] = &[
        ("japanese", Language::Japanese, "japanese.model", "wagahaiwa_nekodearu.txt"),
        ("korean", Language::Korean, "korean.model", "mujeong.txt"),
        ("chinese", Language::Chinese, "chinese.model", "rulin_waishi.txt"),
        ("english", Language::English, "english.model", "pride_and_prejudice.txt"),
    ];
    for (id, language, model, corpus) in cases {
        let (lines, chars) = load_corpus_lines(corpus);
        let segmenter = Segmenter::with_learner(*language, load_adaboost_model(model));
        group.throughput(Throughput::Elements(chars));
        group.bench_function(format!("{id}-strings"), |b| {
            b.iter(|| {
                for line in &lines {
                    black_box(segmenter.segment(black_box(line)));
                }
            });
        });
        group.bench_function(format!("{id}-ranges"), |b| {
            let mut buf = SegmentBuffer::new();
            b.iter(|| {
                for line in &lines {
                    black_box(segmenter.segment_into(black_box(line), &mut buf));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_segment_short,
    bench_segment_long,
    bench_external_corpus,
    bench_segment_into,
    bench_char_type,
    bench_add_corpus,
    bench_predict_adaboost,
);
criterion_main!(benches);
