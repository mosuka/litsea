//! Golden tests: snapshot the segmentation output of every pre-trained model
//! in `models/` so that refactoring can be verified to preserve behavior.
//!
//! These snapshots capture the CURRENT behavior of v0.4.0. If a behavior
//! change is intentional (e.g. fixing the first-word POS handling in
//! `segment_with_pos`), update the affected expectations in the same PR and
//! call the change out explicitly in the PR description.

use std::path::PathBuf;

use litsea::adaboost::AdaBoost;
use litsea::language::Language;
use litsea::perceptron::AveragedPerceptron;
use litsea::segmenter::Segmenter;
use litsea::upos::Upos;

fn model_path(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(name)
        .to_str()
        .unwrap()
        .to_string()
}

async fn adaboost_segmenter(language: Language, model: &str) -> Segmenter {
    let mut learner = AdaBoost::new(0.01, 100);
    learner
        .load_model(&model_path(model))
        .await
        .unwrap_or_else(|e| panic!("failed to load {}: {}", model, e));
    Segmenter::new(language, Some(learner))
}

async fn pos_segmenter(language: Language, model: &str) -> Segmenter {
    let mut learner = AveragedPerceptron::new();
    learner
        .load_model(&model_path(model))
        .await
        .unwrap_or_else(|e| panic!("failed to load {}: {}", model, e));
    Segmenter::with_pos_learner(language, learner)
}

fn assert_segment(segmenter: &Segmenter, cases: &[(&str, &[&str])]) {
    for (input, expected) in cases {
        let actual = segmenter.segment(input);
        assert_eq!(&actual, expected, "segment({:?}) diverged from golden output", input);
    }
}

fn assert_segment_with_pos(segmenter: &Segmenter, cases: &[(&str, &[(&str, &str)])]) {
    for (input, expected) in cases {
        let actual: Vec<(String, Upos)> = segmenter.segment_with_pos(input);
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
// Word segmentation (AdaBoost models)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn golden_segment_japanese() {
    let segmenter = adaboost_segmenter(Language::Japanese, "japanese.model").await;
    assert_segment(
        &segmenter,
        &[
            ("これはテストです。", &["これ", "は", "テスト", "で", "す", "。"]),
            ("私の猫は可愛い。", &["私", "の", "猫", "は", "可愛", "い", "。"]),
            (
                "東京都に住んでいます。",
                &["東京", "都", "に", "住ん", "で", "いま", "す", "。"],
            ),
            // Edge case: single character (whole sentence is one word)
            ("字", &["字"]),
            ("こんにちは", &["こん", "にち", "は"]),
            // Digits and mixed scripts
            ("価格は1000円です。", &["価格", "は", "1000", "円", "で", "す", "。"]),
            ("RustでNLPを実装する。", &["Rust", "で", "NLP", "を", "実装", "する", "。"]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

#[tokio::test]
async fn golden_segment_japanese_rwcp() {
    let segmenter = adaboost_segmenter(Language::Japanese, "RWCP.model").await;
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

#[tokio::test]
async fn golden_segment_japanese_jeita() {
    let segmenter =
        adaboost_segmenter(Language::Japanese, "JEITA_Genpaku_ChaSen_IPAdic.model").await;
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

#[tokio::test]
async fn golden_segment_chinese() {
    let segmenter = adaboost_segmenter(Language::Chinese, "chinese.model").await;
    assert_segment(
        &segmenter,
        &[
            ("这是一个测试。", &["这", "是", "一个", "测试", "。"]),
            ("我喜欢吃中国菜。", &["我喜", "欢吃", "中国", "菜", "。"]),
            ("他在北京工作。", &["他", "在", "北京", "工作", "。"]),
            ("好", &["好"]),
            ("2024年的春天。", &["2024", "年", "的", "春天", "。"]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

#[tokio::test]
async fn golden_segment_korean() {
    let segmenter = adaboost_segmenter(Language::Korean, "korean.model").await;
    assert_segment(
        &segmenter,
        &[
            ("이것은 테스트입니다.", &["이것은", " ", "테스트", "입니다", "."]),
            ("나는 고양이를 좋아한다.", &["나는", " ", "고양이를", " ", "좋아한다", "."]),
            ("한국어 형태소 분석기.", &["한국어", " ", "형태소", " ", "분석기", "."]),
            ("글", &["글"]),
            ("2024년 봄.", &["2024년", " ", "봄."]),
        ],
    );
    assert!(segmenter.segment("").is_empty());
}

// ---------------------------------------------------------------------------
// Joint segmentation + POS tagging (Averaged Perceptron models)
//
// Note: the first word of every sentence is currently tagged "X" because
// segment_with_pos() determines the first word's POS from a prediction made
// before any boundary context exists (see segmenter.rs). This is a known
// quirk snapshotted as-is; Phase 2 of the refactoring plan may change it
// intentionally.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn golden_segment_with_pos_japanese() {
    let segmenter = pos_segmenter(Language::Japanese, "japanese_pos.model").await;
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "これはテストです。",
                &[
                    ("これ", "X"),
                    ("は", "ADP"),
                    ("テスト", "NOUN"),
                    ("です", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "私の猫は可愛い。",
                &[
                    ("私", "X"),
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
                    ("東京", "X"),
                    ("都", "NOUN"),
                    ("に", "ADP"),
                    ("住ん", "VERB"),
                    ("で", "SCONJ"),
                    ("い", "VERB"),
                    ("ます", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
            ("字", &[("字", "SYM")]),
            ("こんにちは", &[("こん", "X"), ("にち", "ADP"), ("は", "ADP")]),
            (
                "価格は1000円です。",
                &[
                    ("価格", "X"),
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
                    ("Rust", "X"),
                    ("で", "ADP"),
                    ("NLP", "PROPN"),
                    ("を", "ADP"),
                    ("実装", "VERB"),
                    ("する", "AUX"),
                    ("。", "PUNCT"),
                ],
            ),
        ],
    );
    assert!(segmenter.segment_with_pos("").is_empty());
}

#[tokio::test]
async fn golden_segment_with_pos_chinese() {
    let segmenter = pos_segmenter(Language::Chinese, "chinese_pos.model").await;
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "这是一个测试。",
                &[
                    ("这", "X"),
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
                    ("我", "X"),
                    ("喜欢", "VERB"),
                    ("吃中", "VERB"),
                    ("国菜", "VERB"),
                    ("。", "PUNCT"),
                ],
            ),
            (
                "他在北京工作。",
                &[("他", "X"), ("在", "ADP"), ("北京", "PROPN"), ("工作", "NOUN"), ("。", "PUNCT")],
            ),
            ("好", &[("好", "PUNCT")]),
            (
                "2024年的春天。",
                &[("2024", "X"), ("年", "NOUN"), ("的", "PART"), ("春天", "NOUN"), ("。", "PUNCT")],
            ),
        ],
    );
    assert!(segmenter.segment_with_pos("").is_empty());
}

#[tokio::test]
async fn golden_segment_with_pos_korean() {
    let segmenter = pos_segmenter(Language::Korean, "korean_pos.model").await;
    assert_segment_with_pos(
        &segmenter,
        &[
            (
                "이것은 테스트입니다.",
                &[("이것은", "X"), (" ", "PUNCT"), ("테스트입니다", "NOUN"), (".", "PUNCT")],
            ),
            (
                "나는 고양이를 좋아한다.",
                &[
                    ("나는", "X"),
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
                    ("한국어", "X"),
                    (" ", "PUNCT"),
                    ("형태소", "NOUN"),
                    (" ", "PUNCT"),
                    ("분석기", "NOUN"),
                    (".", "PUNCT"),
                ],
            ),
            ("글", &[("글", "PUNCT")]),
            ("2024년 봄.", &[("2024년", "X"), (" ", "PUNCT"), ("봄", "NOUN"), (".", "PUNCT")]),
        ],
    );
    assert!(segmenter.segment_with_pos("").is_empty());
}

// ---------------------------------------------------------------------------
// Model file round-trip: load -> save -> load must preserve predictions.
// Guards the on-disk model format compatibility across refactoring.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn roundtrip_adaboost_model() {
    let sentences = ["これはテストです。", "私の猫は可愛い。", "価格は1000円です。", "こんにちは"];

    let mut original = AdaBoost::new(0.01, 100);
    original.load_model(&model_path("japanese.model")).await.unwrap();

    let temp = tempfile::NamedTempFile::new().unwrap();
    original.save_model(temp.path()).unwrap();

    let mut reloaded = AdaBoost::new(0.01, 100);
    reloaded.load_model(temp.path().to_str().unwrap()).await.unwrap();

    let seg_original = Segmenter::new(Language::Japanese, Some(original));
    let seg_reloaded = Segmenter::new(Language::Japanese, Some(reloaded));
    for s in sentences {
        assert_eq!(
            seg_original.segment(s),
            seg_reloaded.segment(s),
            "round-tripped AdaBoost model diverged on {:?}",
            s
        );
    }
}

#[tokio::test]
async fn roundtrip_perceptron_model() {
    let sentences = ["これはテストです。", "私の猫は可愛い。", "価格は1000円です。"];

    let mut original = AveragedPerceptron::new();
    original.load_model(&model_path("japanese_pos.model")).await.unwrap();

    let temp = tempfile::NamedTempFile::new().unwrap();
    original.save_model(temp.path()).unwrap();

    let mut reloaded = AveragedPerceptron::new();
    reloaded.load_model(temp.path().to_str().unwrap()).await.unwrap();

    let seg_original = Segmenter::with_pos_learner(Language::Japanese, original);
    let seg_reloaded = Segmenter::with_pos_learner(Language::Japanese, reloaded);
    for s in sentences {
        assert_eq!(
            seg_original.segment_with_pos(s),
            seg_reloaded.segment_with_pos(s),
            "round-tripped Perceptron model diverged on {:?}",
            s
        );
    }
}
