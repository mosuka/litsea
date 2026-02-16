use std::fs;
use std::path::PathBuf;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;

/// Load a model file from the resources directory.
fn load_model(model_name: &str) -> AdaBoost {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let model_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources").join(model_name);
    let mut learner = AdaBoost::new(0.01, 100);
    rt.block_on(learner.load_model(model_path.to_str().unwrap()))
        .unwrap_or_else(|e| panic!("Failed to load model {}: {}", model_path.display(), e));
    learner
}

fn bench_segment_japanese(c: &mut Criterion) {
    let learner = load_model("japanese.model");
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));
    c.bench_function("segment_japanese_short", |b| {
        b.iter(|| black_box(segmenter.segment(black_box("これはテストです。"))));
    });
}

fn bench_segment_japanese_long(c: &mut Criterion) {
    let learner = load_model("japanese.model");
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));
    let text_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../resources")
        .join("bocchan.txt");
    let text = fs::read_to_string(&text_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", text_path.display(), e));
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    c.bench_function("segment_japanese_long", |b| {
        b.iter(|| {
            for line in &lines {
                black_box(segmenter.segment(black_box(line)));
            }
        });
    });
}

fn bench_segment_chinese(c: &mut Criterion) {
    let learner = load_model("chinese.model");
    let segmenter = Segmenter::new(Language::Chinese, Some(learner));
    c.bench_function("segment_chinese_short", |b| {
        b.iter(|| black_box(segmenter.segment(black_box("这是一个测试。"))));
    });
}

fn bench_segment_korean(c: &mut Criterion) {
    let learner = load_model("korean.model");
    let segmenter = Segmenter::new(Language::Korean, Some(learner));
    c.bench_function("segment_korean_short", |b| {
        b.iter(|| black_box(segmenter.segment(black_box("이것은테스트입니다."))));
    });
}

fn bench_get_type(c: &mut Criterion) {
    let segmenter = Segmenter::new(Language::Japanese, None);
    c.bench_function("get_type_hiragana", |b| {
        b.iter(|| black_box(segmenter.get_type(black_box("あ"))));
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

/// Benchmarks `char_type_patterns()` which includes regex compilation cost on every call.
/// This measures the full cost of creating patterns, not just matching.
fn bench_char_type_patterns(c: &mut Criterion) {
    c.bench_function("char_type_patterns_japanese", |b| {
        b.iter(|| Language::Japanese.char_type_patterns());
    });
}

fn bench_predict(c: &mut Criterion) {
    let learner = load_model("japanese.model");
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));

    // Build a realistic attribute set from the segment pipeline.
    let sentence = "テスト";
    let mut tags = vec!["U".to_string(); 4];
    let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
    let mut types = vec!["O".to_string(); 3];
    for ch in sentence.chars() {
        let s = ch.to_string();
        types.push(segmenter.get_type(&s).to_string());
        chars.push(s);
    }
    chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
    types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);
    tags.extend(vec!["O".to_string(); chars.len() - 4]);

    // Use index 4 to get a valid attribute set via the public API.
    let attrs = segmenter.get_attributes(4, &tags, &chars, &types);

    c.bench_function("predict", |b| {
        b.iter(|| segmenter.learner.predict(black_box(attrs.clone())));
    });
}

criterion_group!(
    benches,
    bench_segment_japanese,
    bench_segment_japanese_long,
    bench_segment_chinese,
    bench_segment_korean,
    bench_get_type,
    bench_add_corpus,
    bench_char_type_patterns,
    bench_predict,
);
criterion_main!(benches);
