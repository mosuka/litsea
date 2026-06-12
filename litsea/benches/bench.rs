use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

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
        let ada_segmenter = Segmenter::new(*language, Some(ada_learner));
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
                b.iter(|| black_box(pos_segmenter.segment_with_pos(black_box(text))));
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
    let ada_segmenter = Segmenter::new(Language::Japanese, Some(ada_learner));
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
                black_box(pos_segmenter.segment_with_pos(black_box(line)));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Internal component benchmarks
// ---------------------------------------------------------------------------

fn bench_char_type(c: &mut Criterion) {
    let segmenter = Segmenter::new(Language::Japanese, None);
    c.bench_function("get_type_hiragana", |b| {
        b.iter(|| black_box(segmenter.char_type(black_box("あ"))));
    });
}

fn bench_add_corpus(c: &mut Criterion) {
    c.bench_function("add_corpus", |b| {
        b.iter_batched(
            || Segmenter::new(Language::Japanese, None),
            |mut segmenter| segmenter.add_corpus(black_box("これ は テスト です 。")),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_predict_adaboost(c: &mut Criterion) {
    let learner = load_adaboost_model("japanese.model");
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));

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
    bench_char_type,
    bench_add_corpus,
    bench_predict_adaboost,
);
criterion_main!(benches);
