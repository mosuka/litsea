use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;

/// Load an AdaBoost model file from the models directory.
fn load_adaboost_model(model_name: &str) -> AdaBoost {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model_name);
    let mut learner = AdaBoost::new(0.01, 100);
    learner
        .load_model_from_path(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
    learner
}

/// Load an AveragedPerceptron model file from the models directory.
fn load_perceptron_model(model_name: &str) -> AveragedPerceptron {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model_name);
    let mut learner = AveragedPerceptron::new();
    learner
        .load_model_from_path(&model_path)
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
    learner
}

// ---------------------------------------------------------------------------
// AdaBoost vs Averaged Perceptron comparison benchmarks
// ---------------------------------------------------------------------------

/// Compares AdaBoost `segment()` and Averaged Perceptron `segment_with_pos()`
/// on a short sentence for each language.
fn bench_segment_short(c: &mut Criterion) {
    let cases: &[(&str, Language, &str, &str)] = &[
        ("japanese", Language::Japanese, "japanese.model", "japanese_pos.model"),
        ("chinese", Language::Chinese, "chinese.model", "chinese_pos.model"),
        ("korean", Language::Korean, "korean.model", "korean_pos.model"),
    ];

    let inputs: &[(&str, &str)] = &[
        ("japanese", "これはテストです。"),
        ("chinese", "这是一个测试。"),
        ("korean", "이것은테스트입니다."),
    ];

    let mut group = c.benchmark_group("segment_short");

    for (lang, language, ada_model, pos_model) in cases {
        let input = inputs.iter().find(|(l, _)| l == lang).unwrap().1;

        // AdaBoost (word segmentation only)
        let ada_learner = load_adaboost_model(ada_model);
        let ada_segmenter = Segmenter::with_learner(*language, ada_learner);
        group.bench_with_input(BenchmarkId::new("adaboost", lang), &input, |b, &text| {
            b.iter(|| black_box(ada_segmenter.segment(black_box(text))));
        });

        // Averaged Perceptron (segmentation + POS)
        let pos_learner = load_perceptron_model(pos_model);
        let pos_segmenter = Segmenter::with_pos_learner(*language, pos_learner);
        group.bench_with_input(
            BenchmarkId::new("averaged_perceptron", lang),
            &input,
            |b, &text| {
                b.iter(|| black_box(pos_segmenter.segment_with_pos(black_box(text)).unwrap()));
            },
        );
    }

    group.finish();
}

/// Compares AdaBoost `segment()` and Averaged Perceptron `segment_with_pos()`
/// on a long text (bocchan.txt) for Japanese.
fn bench_segment_long(c: &mut Criterion) {
    let text_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../resources")
        .join("bocchan.txt");
    let text = fs::read_to_string(&text_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", text_path.display(), e));
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

    let mut group = c.benchmark_group("segment_long_japanese");

    // AdaBoost
    let ada_learner = load_adaboost_model("japanese.model");
    let ada_segmenter = Segmenter::with_learner(Language::Japanese, ada_learner);
    group.bench_function("adaboost", |b| {
        b.iter(|| {
            for line in &lines {
                black_box(ada_segmenter.segment(black_box(line)));
            }
        });
    });

    // Averaged Perceptron
    let pos_learner = load_perceptron_model("japanese_pos.model");
    let pos_segmenter = Segmenter::with_pos_learner(Language::Japanese, pos_learner);
    group.bench_function("averaged_perceptron", |b| {
        b.iter(|| {
            for line in &lines {
                black_box(pos_segmenter.segment_with_pos(black_box(line)).unwrap());
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

/// Reproduces the seven litsea benches of the external tokenizer-speed-bench
/// harness in-repo: one iteration segments every line of the corpus, and
/// criterion's `Throughput::Elements` makes the report read as chars/sec.
/// Methodology differences from the external harness (criterion sampling
/// instead of process-interleaved single passes; the workspace's tuned
/// release profile) are documented in `docs/src/advanced/benchmarking.md`.
fn bench_external_corpus(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_corpus");
    group.sample_size(30);

    // (bench id, language, AdaBoost model, corpus)
    let segment_cases: &[(&str, Language, &str, &str)] = &[
        ("japanese", Language::Japanese, "japanese.model", "wagahaiwa_nekodearu.txt"),
        ("japanese-rwcp", Language::Japanese, "RWCP.model", "wagahaiwa_nekodearu.txt"),
        ("korean", Language::Korean, "korean.model", "mujeong.txt"),
        ("chinese", Language::Chinese, "chinese.model", "rulin_waishi.txt"),
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

    // (bench id, language, Averaged Perceptron model, corpus)
    let pos_cases: &[(&str, Language, &str, &str)] = &[
        (
            "japanese-pos",
            Language::Japanese,
            "japanese_pos.model",
            "wagahaiwa_nekodearu.txt",
        ),
        ("korean-pos", Language::Korean, "korean_pos.model", "mujeong.txt"),
        ("chinese-pos", Language::Chinese, "chinese_pos.model", "rulin_waishi.txt"),
    ];
    for (id, language, model, corpus) in pos_cases {
        let (lines, chars) = load_corpus_lines(corpus);
        let segmenter = Segmenter::with_pos_learner(*language, load_perceptron_model(model));
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

fn bench_char_type(c: &mut Criterion) {
    let segmenter = Segmenter::new(Language::Japanese);
    c.bench_function("char_type_hiragana", |b| {
        b.iter(|| black_box(segmenter.char_type(black_box('あ'))));
    });
}

fn bench_add_corpus(c: &mut Criterion) {
    c.bench_function("add_corpus", |b| {
        b.iter_batched(
            || Segmenter::new(Language::Japanese),
            |mut segmenter| segmenter.add_corpus(black_box("これ は テスト です 。")),
            criterion::BatchSize::SmallInput,
        );
    });
}

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

criterion_group!(
    benches,
    bench_segment_short,
    bench_segment_long,
    bench_external_corpus,
    bench_char_type,
    bench_add_corpus,
    bench_predict_adaboost,
);
criterion_main!(benches);
