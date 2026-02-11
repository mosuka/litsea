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
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../resources")
        .join(model_name);
    let mut learner = AdaBoost::new(0.01, 100, 1);
    rt.block_on(learner.load_model(model_path.to_str().unwrap()))
        .unwrap();
    learner
}

fn bench_segment_japanese(c: &mut Criterion) {
    let learner = load_model("japanese.model");
    let segmenter = Segmenter::new(Language::Japanese, Some(learner));
    c.bench_function("segment_japanese_short", |b| {
        b.iter(|| segmenter.segment(black_box("これはテストです。")));
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
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    c.bench_function("segment_japanese_long", |b| {
        b.iter(|| {
            for line in &lines {
                segmenter.segment(black_box(line));
            }
        });
    });
}

fn bench_segment_chinese(c: &mut Criterion) {
    let learner = load_model("chinese.model");
    let segmenter = Segmenter::new(Language::Chinese, Some(learner));
    c.bench_function("segment_chinese_short", |b| {
        b.iter(|| segmenter.segment(black_box("这是一个测试。")));
    });
}

fn bench_segment_korean(c: &mut Criterion) {
    let learner = load_model("korean.model");
    let segmenter = Segmenter::new(Language::Korean, Some(learner));
    c.bench_function("segment_korean_short", |b| {
        b.iter(|| segmenter.segment(black_box("이것은테스트입니다.")));
    });
}

fn bench_get_type(c: &mut Criterion) {
    let segmenter = Segmenter::new(Language::Japanese, None);
    c.bench_function("get_type_hiragana", |b| {
        b.iter(|| segmenter.get_type(black_box("あ")));
    });
}

fn bench_add_corpus(c: &mut Criterion) {
    c.bench_function("add_corpus", |b| {
        b.iter(|| {
            let mut segmenter = Segmenter::new(Language::Japanese, None);
            segmenter.add_corpus(black_box("これ は テスト です 。"));
        });
    });
}

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
    let mut chars =
        vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
    let mut types = vec!["O".to_string(); 3];
    for ch in sentence.chars() {
        let s = ch.to_string();
        types.push(segmenter.get_type(&s).to_string());
        chars.push(s);
    }
    chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
    types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);
    tags.extend(vec!["O".to_string(); chars.len() - 4]);

    // Use index 4 to get a valid attribute set.
    let attrs =
        build_attributes(4, &tags, &chars, &types, &segmenter);

    c.bench_function("predict", |b| {
        b.iter(|| segmenter.learner.predict(black_box(attrs.clone())));
    });
}

/// Build attributes matching the Segmenter::get_attributes logic.
fn build_attributes(
    i: usize,
    tags: &[String],
    chars: &[String],
    types: &[String],
    segmenter: &Segmenter,
) -> std::collections::HashSet<String> {
    let w1 = &chars[i - 3];
    let w2 = &chars[i - 2];
    let w3 = &chars[i - 1];
    let w4 = &chars[i];
    let w5 = &chars[i + 1];
    let w6 = &chars[i + 2];
    let c1 = &types[i - 3];
    let c2 = &types[i - 2];
    let c3 = &types[i - 1];
    let c4 = &types[i];
    let c5 = &types[i + 1];
    let c6 = &types[i + 2];
    let p1 = &tags[i - 3];
    let p2 = &tags[i - 2];
    let p3 = &tags[i - 1];

    let mut attrs: std::collections::HashSet<String> = [
        format!("UP1:{}", p1),
        format!("UP2:{}", p2),
        format!("UP3:{}", p3),
        format!("BP1:{}{}", p1, p2),
        format!("BP2:{}{}", p2, p3),
        format!("UW1:{}", w1),
        format!("UW2:{}", w2),
        format!("UW3:{}", w3),
        format!("UW4:{}", w4),
        format!("UW5:{}", w5),
        format!("UW6:{}", w6),
        format!("BW1:{}{}", w2, w3),
        format!("BW2:{}{}", w3, w4),
        format!("BW3:{}{}", w4, w5),
        format!("UC1:{}", c1),
        format!("UC2:{}", c2),
        format!("UC3:{}", c3),
        format!("UC4:{}", c4),
        format!("UC5:{}", c5),
        format!("UC6:{}", c6),
        format!("BC1:{}{}", c2, c3),
        format!("BC2:{}{}", c3, c4),
        format!("BC3:{}{}", c4, c5),
        format!("TC1:{}{}{}", c1, c2, c3),
        format!("TC2:{}{}{}", c2, c3, c4),
        format!("TC3:{}{}{}", c3, c4, c5),
        format!("TC4:{}{}{}", c4, c5, c6),
        format!("UQ1:{}{}", p1, c1),
        format!("UQ2:{}{}", p2, c2),
        format!("UQ3:{}{}", p3, c3),
        format!("BQ1:{}{}{}", p2, c2, c3),
        format!("BQ2:{}{}{}", p2, c3, c4),
        format!("BQ3:{}{}{}", p3, c2, c3),
        format!("BQ4:{}{}{}", p3, c3, c4),
        format!("TQ1:{}{}{}{}", p2, c1, c2, c3),
        format!("TQ2:{}{}{}{}", p2, c2, c3, c4),
        format!("TQ3:{}{}{}{}", p3, c1, c2, c3),
        format!("TQ4:{}{}{}{}", p3, c2, c3, c4),
    ]
    .iter()
    .cloned()
    .collect();

    match segmenter.language {
        Language::Japanese | Language::Chinese => {
            attrs.insert(format!("WC1:{}{}", w3, c4));
            attrs.insert(format!("WC2:{}{}", c3, w4));
            attrs.insert(format!("WC3:{}{}", w3, c3));
            attrs.insert(format!("WC4:{}{}", w4, c4));
        }
        _ => {}
    }

    attrs
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
