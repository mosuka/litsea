//! Golden tests: snapshot the segmentation output of every pre-trained
//! model in `models/` — the AdaBoost-format segmentation models and the
//! two-stage models — so that refactoring can be verified to preserve
//! behavior.
//!
//! These snapshots capture the current behavior of the bundled models. If a
//! behavior change is intentional (e.g. retraining a model), update the
//! affected expectations in the same PR and call the change out explicitly
//! in the PR description.

use std::path::PathBuf;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::segmenter::Segmenter;
use litsea::two_stage::TwoStageLearner;
use litsea::upos::Upos;

fn model_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(name)
}

fn adaboost_segmenter(language: Language, model: &str) -> Segmenter {
    let mut learner = AdaBoost::new(0.01, 100);
    learner
        .load_model_from_path(&model_path(model))
        .unwrap_or_else(|e| panic!("failed to load {}: {}", model, e));
    Segmenter::with_learner(language, learner)
}

fn two_stage_segmenter(language: Language, model: &str) -> Segmenter {
    let mut learner = TwoStageLearner::new();
    learner
        .load_model_from_path(&model_path(model))
        .unwrap_or_else(|e| panic!("failed to load {}: {}", model, e));
    Segmenter::with_two_stage_learner(language, learner)
}

fn assert_segment(segmenter: &Segmenter, cases: &[(&str, &[&str])]) {
    for (input, expected) in cases {
        let actual = segmenter.segment(input);
        assert_eq!(&actual, expected, "segment({:?}) diverged from golden output", input);
    }
}

fn assert_segment_with_pos(segmenter: &Segmenter, cases: &[(&str, &[(&str, &str)])]) {
    for (input, expected) in cases {
        let actual: Vec<(String, Upos)> =
            segmenter.segment_with_pos(input).expect("POS learner is set");
        let actual_str: Vec<(String, String)> =
            actual.into_iter().map(|(w, p)| (w, p.to_string())).collect();
        let expected_owned: Vec<(String, String)> =
            expected.iter().map(|(w, p)| (w.to_string(), p.to_string())).collect();
        assert_eq!(
            actual_str, expected_owned,
            "segment_with_pos({:?}) diverged from golden output",
            input
        );
    }
}

// ---------------------------------------------------------------------------
// Word segmentation (AdaBoost-format models — RWCP.model and the JEITA
// model are genuinely AdaBoost-trained; japanese.model, chinese.model, and
// korean.model are a 2-class Averaged Perceptron losslessly collapsed to
// AdaBoost-format scalar weights, per issue #165)
// ---------------------------------------------------------------------------

#[test]
fn golden_segment_japanese() {
    let segmenter = adaboost_segmenter(Language::Japanese, "japanese.model");
    assert_segment(
        &segmenter,
        &[
            ("これはテストです。", &["これ", "は", "テスト", "です", "。"]),
            ("私の猫は可愛い。", &["私", "の", "猫", "は", "可愛い", "。"]),
            (
                "東京都に住んでいます。",
                &["東京", "都", "に", "住ん", "で", "い", "ます", "。"],
            ),
            // Edge case: single character (whole sentence is one word)
            ("字", &["字"]),
            ("こんにちは", &["こんに", "ち", "は"]),
            // Digits and mixed scripts
            ("価格は1000円です。", &["価格", "は", "1000", "円", "です", "。"]),
            ("RustでNLPを実装する。", &["Rust", "で", "NLP", "を", "実装", "する", "。"]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

#[test]
fn golden_segment_japanese_rwcp() {
    let segmenter = adaboost_segmenter(Language::Japanese, "RWCP.model");
    assert_segment(
        &segmenter,
        &[
            ("これはテストです。", &["これ", "は", "テスト", "です", "。"]),
            ("私の猫は可愛い。", &["私", "の", "猫", "は", "可愛い", "。"]),
            ("東京都に住んでいます。", &["東京都", "に", "住ん", "でい", "ます", "。"]),
            ("字", &["字"]),
            // Edge case: whole sentence is a single word
            ("こんにちは", &["こんにちは"]),
            ("価格は1000円です。", &["価格", "は", "1", "0", "0", "0", "円", "です", "。"]),
            ("RustでNLPを実装する。", &["Rust", "で", "NLP", "を", "実装", "する", "。"]),
        ],
    );
}

#[test]
fn golden_segment_japanese_jeita() {
    let segmenter = adaboost_segmenter(Language::Japanese, "JEITA_Genpaku_ChaSen_IPAdic.model");
    assert_segment(
        &segmenter,
        &[
            ("これはテストです。", &["これ", "は", "テスト", "です", "。"]),
            ("私の猫は可愛い。", &["私", "の", "猫", "は", "可愛", "い", "。"]),
            (
                "東京都に住んでいます。",
                &["東京", "都", "に", "住ん", "で", "い", "ます", "。"],
            ),
            ("字", &["字"]),
            ("こんにちは", &["こん", "にち", "は"]),
            ("価格は1000円です。", &["価格", "は", "1000", "円", "です", "。"]),
            ("RustでNLPを実装する。", &["Rust", "で", "NLP", "を", "実装", "する", "。"]),
        ],
    );
}

#[test]
fn golden_segment_chinese() {
    let segmenter = adaboost_segmenter(Language::Chinese, "chinese.model");
    assert_segment(
        &segmenter,
        &[
            ("这是一个测试。", &["这", "是", "一", "个", "测试", "。"]),
            ("我喜欢吃中国菜。", &["我", "喜欢", "吃", "中", "国菜", "。"]),
            ("他在北京工作。", &["他", "在", "北京", "工作", "。"]),
            ("好", &["好"]),
            ("2024年的春天。", &["2024", "年", "的", "春天", "。"]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

#[test]
fn golden_segment_korean() {
    let segmenter = adaboost_segmenter(Language::Korean, "korean.model");
    assert_segment(
        &segmenter,
        &[
            ("이것은 테스트입니다.", &["이것은", " ", "테스트입니다", "."]),
            ("나는 고양이를 좋아한다.", &["나는", " ", "고양이를", " ", "좋아한다", "."]),
            ("한국어 형태소 분석기.", &["한국어", " ", "형태소", " ", "분석기", "."]),
            ("글", &["글"]),
            ("2024년 봄.", &["2024년", " ", "봄", "."]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

// ---------------------------------------------------------------------------
// Two-stage segmentation + POS tagging (issue #147)
//
// Stage 1 segments with a binary boundary classifier, then stage 2 tags
// each word through the candidate-tag lexicon (skipping the classifier
// entirely for single-candidate and dominance-dominant surfaces).
//
// These are snapshots of current behavior, not of correct behavior. Chinese
// "我喜欢吃中国菜。" (gold: 我 / 喜欢 / 吃 / 中国 / 菜) is the clearest
// case — the model recovers the first three tokens but splits "中国" as
// "中"/"国菜", where "国菜" is not a word (UD Chinese-GSD tokenizes 中國 as
// one token throughout). The snapshots are not a target to converge on;
// when a model is retrained, re-derive them from its actual output.
// ---------------------------------------------------------------------------

#[test]
fn golden_segment_with_pos_japanese_two_stage() {
    let segmenter = two_stage_segmenter(Language::Japanese, "japanese_pos.model");
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "これはテストです。",
                &[
                    ("これ", "PRON"),
                    ("は", "ADP"),
                    ("テスト", "NOUN"),
                    ("です", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "私の猫は可愛い。",
                &[
                    ("私", "PRON"),
                    ("の", "ADP"),
                    ("猫", "NOUN"),
                    ("は", "ADP"),
                    ("可愛い", "ADJ"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "東京都に住んでいます。",
                &[
                    ("東京", "PROPN"),
                    ("都", "NOUN"),
                    ("に", "ADP"),
                    ("住ん", "VERB"),
                    ("で", "SCONJ"),
                    ("い", "VERB"),
                    ("ます", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
            ("字", &[("字", "NOUN")]),
            (
                "こんにちは",
                &[("こん", "VERB"), ("に", "SCONJ"), ("ち", "NOUN"), ("は", "ADP")],
            ),
            (
                "価格は1000円です。",
                &[
                    ("価格", "NOUN"),
                    ("は", "ADP"),
                    ("1000", "NUM"),
                    ("円", "NOUN"),
                    ("です", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "RustでNLPを実装する。",
                &[
                    ("Rust", "NOUN"),
                    ("で", "ADP"),
                    ("NLP", "NOUN"),
                    ("を", "ADP"),
                    ("実装", "VERB"),
                    ("する", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
        ],
    );
    assert!(segmenter.segment_with_pos("").expect("two-stage learner is set").is_empty());
}

#[test]
fn golden_segment_with_pos_chinese_two_stage() {
    let segmenter = two_stage_segmenter(Language::Chinese, "chinese_pos.model");
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "这是一个测试。",
                &[
                    ("这", "PROPN"),
                    ("是", "AUX"),
                    ("一", "NUM"),
                    ("个", "NOUN"),
                    ("测试", "NOUN"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "我喜欢吃中国菜。",
                &[
                    ("我", "PRON"),
                    ("喜欢", "VERB"),
                    ("吃", "VERB"),
                    ("中", "ADP"),
                    ("国菜", "NOUN"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "他在北京工作。",
                &[
                    ("他", "PRON"),
                    ("在", "VERB"),
                    ("北京", "PROPN"),
                    ("工作", "NOUN"),
                    ("。", "PUNCT"),
                ],
            ),
            ("好", &[("好", "ADJ")]),
            (
                "2024年的春天。",
                &[
                    ("2024", "NUM"),
                    ("年", "NOUN"),
                    ("的", "PART"),
                    ("春天", "NOUN"),
                    ("。", "PUNCT"),
                ],
            ),
        ],
    );
    assert!(segmenter.segment_with_pos("").expect("two-stage learner is set").is_empty());
}

#[test]
fn golden_segment_with_pos_korean_two_stage() {
    // Note: korean_pos.model is trained on the unspaced `word/POS`
    // corpus, not the space-preserving TSV corpus korean.model uses.
    // Inference still receives the spaced text as-is, so spaces surface as
    // their own tokens here.
    let segmenter = two_stage_segmenter(Language::Korean, "korean_pos.model");
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "이것은 테스트입니다.",
                &[("이것은", "PRON"), (" ", "PUNCT"), ("테스트입니다", "VERB"), (".", "PUNCT")],
            ),
            (
                "나는 고양이를 좋아한다.",
                &[
                    ("나는", "PRON"),
                    (" ", "PUNCT"),
                    ("고양이를", "NOUN"),
                    (" ", "PUNCT"),
                    ("좋아한다", "VERB"),
                    (".", "PUNCT"),
                ],
            ),
            (
                "한국어 형태소 분석기.",
                &[
                    ("한국어", "NOUN"),
                    (" ", "PUNCT"),
                    ("형태소", "NOUN"),
                    (" ", "PUNCT"),
                    ("분석기", "NOUN"),
                    (".", "PUNCT"),
                ],
            ),
            ("글", &[("글", "NOUN")]),
            (
                "2024년 봄.",
                &[("2024년", "NOUN"), (" ", "PUNCT"), ("봄", "NOUN"), (".", "PUNCT")],
            ),
        ],
    );
    assert!(segmenter.segment_with_pos("").expect("two-stage learner is set").is_empty());
}

// ---------------------------------------------------------------------------
// Model file round-trip: load -> save -> load must preserve predictions.
// Guards the on-disk model format compatibility across refactoring.
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_adaboost_model() {
    let sentences = ["これはテストです。", "私の猫は可愛い。", "価格は1000円です。", "こんにちは"];

    let mut original = AdaBoost::new(0.01, 100);
    original.load_model_from_path(&model_path("japanese.model")).unwrap();

    let temp = tempfile::NamedTempFile::new().unwrap();
    original.save_model(temp.path()).unwrap();

    let mut reloaded = AdaBoost::new(0.01, 100);
    reloaded.load_model_from_path(temp.path()).unwrap();

    let seg_original = Segmenter::with_learner(Language::Japanese, original);
    let seg_reloaded = Segmenter::with_learner(Language::Japanese, reloaded);
    for s in sentences {
        assert_eq!(
            seg_original.segment(s),
            seg_reloaded.segment(s),
            "round-tripped AdaBoost model diverged on {:?}",
            s
        );
    }
}

#[test]
fn roundtrip_two_stage_model() {
    let sentences = ["これはテストです。", "私の猫は可愛い。", "価格は1000円です。", "こんにちは"];

    let mut original = TwoStageLearner::new();
    original.load_model_from_path(&model_path("japanese_pos.model")).unwrap();

    let temp = tempfile::NamedTempFile::new().unwrap();
    original.save_model(temp.path()).unwrap();

    let mut reloaded = TwoStageLearner::new();
    reloaded.load_model_from_path(temp.path()).unwrap();

    let seg_original = Segmenter::with_two_stage_learner(Language::Japanese, original);
    let seg_reloaded = Segmenter::with_two_stage_learner(Language::Japanese, reloaded);
    for s in sentences {
        assert_eq!(
            seg_original.segment_with_pos(s).expect("two-stage learner is set"),
            seg_reloaded.segment_with_pos(s).expect("two-stage learner is set"),
            "round-tripped two-stage model diverged on {:?}",
            s
        );
    }
}
