//! Browser tests for the WebAssembly binding.
//!
//! Run with `wasm-pack test --headless --firefox` (or via
//! `make test-litsea-wasm`, which regenerates the fixtures first).

use js_sys::{Object, Reflect};
use litsea_wasm::segmenter::Segmenter;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

/// The bundled models, compiled into the test binary because a browser
/// cannot read files. They are not part of the shipped package.
const JAPANESE: &[u8] = include_bytes!("../../models/japanese.model");
const CHINESE: &[u8] = include_bytes!("../../models/chinese.model");
const KOREAN: &[u8] = include_bytes!("../../models/korean.model");
const ENGLISH: &[u8] = include_bytes!("../../models/english.model");
const JAPANESE_POS: &[u8] = include_bytes!("../../models/japanese_pos.model");
const KOREAN_POS: &[u8] = include_bytes!("../../models/korean_pos.model");

/// Expected output, produced by the `litsea` CLI (see
/// `tests/generate_fixtures.sh`) so the reference implementation decides
/// what is correct, exactly as in the other bindings' parity tests.
///
/// Committed rather than generated on demand: `include_str!` resolves at
/// compile time, so the file has to exist for any build of this target. CI
/// regenerates it and fails on a diff.
const FIXTURES: &str = include_str!("fixtures.tsv");

/// Returns the model bytes for a fixture row.
///
/// # Arguments
/// * `model` - The model file name.
///
/// # Returns
/// The compiled-in bytes.
fn model_bytes(model: &str) -> &'static [u8] {
    match model {
        "japanese.model" => JAPANESE,
        "chinese.model" => CHINESE,
        "korean.model" => KOREAN,
        "english.model" => ENGLISH,
        "japanese_pos.model" => JAPANESE_POS,
        "korean_pos.model" => KOREAN_POS,
        other => panic!("unknown fixture model: {other}"),
    }
}

/// One row of `fixtures.tsv`.
struct Fixture {
    /// The language name passed to `fromBytes`.
    language: &'static str,
    /// The model file name.
    model: &'static str,
    /// `"seg"` or `"pos"`.
    mode: &'static str,
    /// The untagged user dictionary words (#263), or `None` for the `-`
    /// placeholder.
    words: Option<Vec<String>>,
    /// The tagged user dictionary words (#264), `(word, tag)` from the
    /// `word=TAG` entries of the words field.
    tags: Vec<(String, String)>,
    /// The input sentence.
    sentence: &'static str,
    /// The CLI's rendered output line.
    expected: &'static str,
}

impl Fixture {
    /// Builds the segmenter this row describes, dictionary included.
    ///
    /// # Returns
    /// The segmenter.
    ///
    /// # Panics
    /// Panics if the model cannot be loaded.
    fn segmenter(&self) -> Segmenter {
        Segmenter::from_bytes(
            self.language,
            model_bytes(self.model),
            self.words.clone(),
            None,
            tag_object(&self.tags),
        )
        .unwrap()
    }

    /// Renders `segmenter`'s output for this row the way the CLI prints it:
    /// tokens for a `"seg"` row, `surface/TAG` tokens for a `"pos"` row.
    ///
    /// # Arguments
    /// * `segmenter` - The segmenter to run on the row's sentence.
    ///
    /// # Returns
    /// The rendered line.
    ///
    /// # Panics
    /// Panics if a `"pos"` row's segmenter cannot tag.
    fn render(&self, segmenter: &Segmenter) -> String {
        if self.mode == "pos" {
            segmenter
                .segment_with_pos(self.sentence)
                .unwrap()
                .into_iter()
                .map(|token| format!("{}/{}", token.surface(), token.pos().unwrap_or_default()))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            segmenter.segment(self.sentence).join(" ")
        }
    }
}

/// Builds the `userWordTags` object of `fromBytes` from `(word, tag)` pairs.
///
/// # Arguments
/// * `tags` - The tagged words.
///
/// # Returns
/// `None` for no pairs, otherwise a plain object mapping each word to its
/// tag.
fn tag_object(tags: &[(String, String)]) -> Option<Object> {
    if tags.is_empty() {
        return None;
    }
    let object = Object::new();
    for (word, tag) in tags {
        Reflect::set(&object, &JsValue::from_str(word), &JsValue::from_str(tag)).unwrap();
    }
    Some(object)
}

/// Parses `fixtures.tsv`: `language`, `model`, `mode`, `words` (comma
/// separated, `word=TAG` for a tagged word, or `-` for no dictionary),
/// `sentence`, `expected`.
///
/// # Returns
/// Every fixture row, in file order.
///
/// # Panics
/// Panics on a row with fewer than six fields.
fn fixtures() -> Vec<Fixture> {
    FIXTURES
        .lines()
        .map(|line| {
            let mut fields = line.split('\t');
            let mut next = || fields.next().expect("fixture row has six fields");
            let (language, model, mode, words, sentence, expected) =
                (next(), next(), next(), next(), next(), next());
            let entries: Vec<&str> =
                if words == "-" { Vec::new() } else { words.split(',').collect() };
            let tags = entries
                .iter()
                .filter_map(|entry| entry.split_once('='))
                .map(|(word, tag)| (word.to_string(), tag.to_string()))
                .collect();
            Fixture {
                language,
                model,
                mode,
                words: (words != "-").then(|| {
                    entries.iter().filter(|e| !e.contains('=')).map(|e| e.to_string()).collect()
                }),
                tags,
                sentence,
                expected,
            }
        })
        .collect()
}

/// Unwraps the error side of a result whose success type is not `Debug`.
///
/// # Arguments
/// * `result` - The result to inspect.
///
/// # Returns
/// The thrown value.
///
/// # Panics
/// Panics if the call unexpectedly succeeded.
fn expect_err<T>(result: Result<T, JsValue>) -> JsValue {
    match result {
        Ok(_) => panic!("expected the call to throw"),
        Err(error) => error,
    }
}

/// Reads the error `code` property a thrown value carries.
///
/// # Arguments
/// * `error` - The value the binding threw.
///
/// # Returns
/// The `code` string, or `None` if the value has no such property.
fn error_code(error: &JsValue) -> Option<String> {
    js_sys::Reflect::get(error, &JsValue::from_str("code"))
        .ok()
        .and_then(|code| code.as_string())
}

#[wasm_bindgen_test]
fn segment_matches_the_cli() {
    for fixture in fixtures().iter().filter(|f| f.mode == "seg") {
        // Compare the rendered line, not a re-split of it: the CLI joins
        // tokens with a space, so a whitespace token cannot be recovered by
        // splitting the output again.
        assert_eq!(
            fixture.segmenter().segment(fixture.sentence).join(" "),
            fixture.expected,
            "{} {:?}",
            fixture.language,
            fixture.words
        );
    }
}

#[wasm_bindgen_test]
fn segment_with_pos_matches_the_cli() {
    for fixture in fixtures().iter().filter(|f| f.mode == "pos") {
        let rendered = fixture
            .segmenter()
            .segment_with_pos(fixture.sentence)
            .unwrap()
            .into_iter()
            .map(|token| format!("{}/{}", token.surface(), token.pos().unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(rendered, fixture.expected, "{} {:?}", fixture.language, fixture.words);
    }
}

#[wasm_bindgen_test]
fn user_words_change_the_output() {
    // Guards the fixtures themselves: every dictionary row must render
    // differently from the same row without its dictionary, or the parity
    // tests above would pass with the dictionary (or its tags) ignored.
    let with_words: Vec<Fixture> = fixtures().into_iter().filter(|f| f.words.is_some()).collect();
    assert!(!with_words.is_empty(), "no user-dictionary fixtures");
    assert!(with_words.iter().any(|f| !f.tags.is_empty()), "no tagged fixtures");
    for fixture in &with_words {
        let plain =
            Segmenter::from_bytes(fixture.language, model_bytes(fixture.model), None, None, None)
                .unwrap();
        assert_ne!(
            fixture.render(&plain),
            fixture.render(&fixture.segmenter()),
            "{:?} {:?} has no effect on {}",
            fixture.words,
            fixture.tags,
            fixture.sentence
        );
    }
}

#[wasm_bindgen_test]
fn user_words_gate() {
    let words = || Some(vec![String::from("雪だるま")]);
    let plain = Segmenter::from_bytes("japanese", JAPANESE, None, None, None)
        .unwrap()
        .segment("雪だるまを作る");
    // Gate 0 applies a word only where the model already agrees with it.
    let gated = Segmenter::from_bytes("japanese", JAPANESE, words(), Some(0.0), None).unwrap();
    assert_eq!(gated.segment("雪だるまを作る"), plain);
    let open = Segmenter::from_bytes("japanese", JAPANESE, words(), Some(1.0), None).unwrap();
    assert_eq!(open.segment("雪だるまを作る"), vec!["雪だるま", "を", "作る"]);
    // An empty list is no dictionary at all.
    let empty = Segmenter::from_bytes("japanese", JAPANESE, Some(Vec::new()), None, None).unwrap();
    assert_eq!(empty.segment("雪だるまを作る"), plain);
}

#[wasm_bindgen_test]
fn an_out_of_range_user_words_gate_throws_with_a_code() {
    for words in [Some(vec![String::from("雪だるま")]), None] {
        let error = expect_err(Segmenter::from_bytes("japanese", JAPANESE, words, Some(1.5), None));
        assert_eq!(error_code(&error).as_deref(), Some("invalid_argument"));
    }
}

#[wasm_bindgen_test]
fn user_word_tags_override_the_tagger() {
    // #264: `チャットGPT` is unknown to the lexicon and the tagger says NOUN;
    // the registered tag wins, also without `userWords`, and segmentation
    // alone ignores it.
    let tags = [(String::from("チャットGPT"), String::from("PROPN"))];
    let segmenter =
        Segmenter::from_bytes("japanese", JAPANESE_POS, None, None, tag_object(&tags)).unwrap();
    let tokens = segmenter.segment_with_pos("チャットGPTに聞く").unwrap();
    assert_eq!(tokens[0].surface(), "チャットGPT");
    assert_eq!(tokens[0].pos().as_deref(), Some("PROPN"));
    assert_eq!(segmenter.segment("チャットGPTに聞く"), vec!["チャットGPT", "に", "聞く"]);
}

#[wasm_bindgen_test]
fn an_invalid_user_word_tag_throws_with_a_code() {
    let tags = [(String::from("ポケモン"), String::from("Noun"))];
    let error =
        expect_err(Segmenter::from_bytes("japanese", JAPANESE_POS, None, None, tag_object(&tags)));
    assert_eq!(error_code(&error).as_deref(), Some("invalid_argument"));
    // A tag that is not a string is rejected before it is parsed.
    let object = Object::new();
    Reflect::set(&object, &JsValue::from_str("ポケモン"), &JsValue::from_f64(1.0)).unwrap();
    let error =
        expect_err(Segmenter::from_bytes("japanese", JAPANESE_POS, None, None, Some(object)));
    assert_eq!(error_code(&error).as_deref(), Some("invalid_argument"));
}

#[wasm_bindgen_test]
fn byte_offsets_tile_the_input() {
    for fixture in fixtures().iter().filter(|f| f.mode == "seg") {
        let (language, sentence) = (fixture.language, fixture.sentence);
        let tokens = fixture.segmenter().segment_tokens(sentence);
        assert!(!tokens.is_empty(), "{language}");

        let mut expected_start = 0u32;
        let mut joined = String::new();
        for token in &tokens {
            assert_eq!(token.start(), expected_start, "{language}: gap in offsets");
            let slice = &sentence.as_bytes()[token.start() as usize..token.end() as usize];
            assert_eq!(
                std::str::from_utf8(slice).unwrap(),
                token.surface(),
                "{language}: offsets must slice back to the surface"
            );
            assert!(token.pos().is_none());
            expected_start = token.end();
            joined.push_str(&token.surface());
        }
        assert_eq!(expected_start as usize, sentence.len(), "{language}");
        assert_eq!(joined, sentence, "{language}");
    }
}

#[wasm_bindgen_test]
fn whitespace_is_its_own_token() {
    let segmenter = Segmenter::from_bytes("korean", KOREAN, None, None, None).unwrap();
    assert_eq!(
        segmenter.segment("안녕하세요 반갑습니다"),
        vec!["안녕하세요", " ", "반갑습니다"]
    );
}

#[wasm_bindgen_test]
fn model_kind_is_detected() {
    assert!(!Segmenter::from_bytes("ja", JAPANESE, None, None, None).unwrap().has_pos());
    assert!(Segmenter::from_bytes("ja", JAPANESE_POS, None, None, None).unwrap().has_pos());
}

#[wasm_bindgen_test]
fn language_names_and_codes_are_interchangeable() {
    let expected = Segmenter::from_bytes("japanese", JAPANESE, None, None, None)
        .unwrap()
        .segment("これはテストです。");
    for name in ["ja", "JA", "Japanese"] {
        assert_eq!(
            Segmenter::from_bytes(name, JAPANESE, None, None, None)
                .unwrap()
                .segment("これはテストです。"),
            expected,
            "{name}"
        );
    }
}

#[wasm_bindgen_test]
fn pos_on_a_segmentation_model_throws_with_a_code() {
    let segmenter = Segmenter::from_bytes("japanese", JAPANESE, None, None, None).unwrap();
    let error = expect_err(segmenter.segment_with_pos("これはテストです。"));
    assert_eq!(error_code(&error).as_deref(), Some("pos_unavailable"));
}

#[wasm_bindgen_test]
fn an_unknown_language_throws_with_a_code() {
    let error = expect_err(Segmenter::from_bytes("klingon", JAPANESE, None, None, None));
    assert_eq!(error_code(&error).as_deref(), Some("invalid_argument"));
}

#[wasm_bindgen_test]
fn a_legacy_joint_model_throws_with_a_code() {
    // A bare integer first line is the joint class-count header.
    let error = expect_err(Segmenter::from_bytes("japanese", b"17\nfoo\t1.0\n", None, None, None));
    assert_eq!(error_code(&error).as_deref(), Some("model"));
}

#[wasm_bindgen_test]
fn a_malformed_model_throws_with_a_code() {
    let error =
        expect_err(Segmenter::from_bytes("japanese", b"this is not a model\n", None, None, None));
    assert_eq!(error_code(&error).as_deref(), Some("parse"));
}

#[wasm_bindgen_test]
fn module_functions() {
    assert_eq!(litsea_wasm::version(), litsea::version());
    assert_eq!(
        litsea_wasm::supported_languages(),
        vec!["japanese", "chinese", "korean", "english"]
    );
}
