use std::collections::HashSet;
use std::fmt::Write as _;

use crate::adaboost::AdaBoost;
use crate::language::Language;
use crate::perceptron::AveragedPerceptron;
use crate::upos::{SegmentLabel, Upos};

/// Segmenter struct for text segmentation using AdaBoost
/// It uses predefined patterns to classify characters and segment sentences into words.
pub struct Segmenter {
    language: Language,
    learner: AdaBoost,
    /// 品詞推定用のAveraged Perceptron（オプション）
    pos_learner: Option<AveragedPerceptron>,
}

impl Segmenter {
    /// Creates a new instance of [`Segmenter`].
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    /// * `learner` - An optional AdaBoost instance. If None, a default AdaBoost instance is created.
    ///
    /// # Returns
    /// A new Segmenter instance with the specified language and AdaBoost learner.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese, None);
    /// ```
    pub fn new(language: Language, learner: Option<AdaBoost>) -> Self {
        Segmenter {
            language,
            learner: learner.unwrap_or_else(|| AdaBoost::new(0.01, 100)),
            pos_learner: None,
        }
    }

    /// Creates a new instance of [`Segmenter`] with a POS learner.
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    /// * `pos_learner` - An AveragedPerceptron instance for POS tagging.
    ///
    /// # Returns
    /// A new Segmenter instance configured for joint segmentation + POS tagging.
    pub fn with_pos_learner(language: Language, pos_learner: AveragedPerceptron) -> Self {
        Segmenter {
            language,
            learner: AdaBoost::new(0.01, 100),
            pos_learner: Some(pos_learner),
        }
    }

    /// Returns the language this segmenter was created for.
    #[must_use]
    pub fn language(&self) -> Language {
        self.language
    }

    /// Returns a reference to the AdaBoost learner used for segmentation.
    #[must_use]
    pub fn learner(&self) -> &AdaBoost {
        &self.learner
    }

    /// Returns a mutable reference to the AdaBoost learner used for segmentation.
    pub fn learner_mut(&mut self) -> &mut AdaBoost {
        &mut self.learner
    }

    /// Returns a reference to the POS learner, if one is set.
    #[must_use]
    pub fn pos_learner(&self) -> Option<&AveragedPerceptron> {
        self.pos_learner.as_ref()
    }

    /// Returns a mutable reference to the POS learner, if one is set.
    pub fn pos_learner_mut(&mut self) -> Option<&mut AveragedPerceptron> {
        self.pos_learner.as_mut()
    }

    /// Gets the type of a character based on language-specific patterns.
    ///
    /// # Arguments
    /// * `ch` - A string slice representing a single character.
    ///
    /// # Returns
    /// A string slice representing the type code of the character.
    /// The type codes are language-specific. Returns "O" (Other) if no pattern matches.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese, None);
    /// let char_type = segmenter.char_type("あ");
    /// assert_eq!(char_type, "I"); // Hiragana
    /// ```
    #[must_use]
    pub fn char_type(&self, ch: &str) -> &str {
        match ch.chars().next() {
            Some(c) => self.language.char_type(c),
            None => "O",
        }
    }

    /// Builds the padded character and character-type arrays for a text.
    ///
    /// Returns `(chars, types)` where the first three entries are the B3/B2/B1
    /// head sentinels and the last three are the E1/E2/E3 tail sentinels, so
    /// real characters occupy indices `3..chars.len() - 3`.
    fn sentence_context(&self, text: &str) -> (Vec<String>, Vec<&'static str>) {
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types: Vec<&'static str> = vec!["O"; 3];
        for ch in text.chars() {
            types.push(self.language.char_type(ch));
            chars.push(ch.to_string());
        }
        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O", "O", "O"]);
        (chars, types)
    }

    /// Shared corpus-processing pipeline behind `process_corpus` and
    /// `process_corpus_with_pos`.
    ///
    /// `tokens` yields `(word, label)` pairs where `label` is assigned to the
    /// first character of the word; continuation characters receive
    /// `cont_label`. Builds the padded context arrays and invokes `callback`
    /// with the feature set and label for each character position except the
    /// first one, which has no preceding boundary decision to learn from.
    fn process_tokens<'a, L, I, F>(&self, tokens: I, cont_label: L, mut callback: F)
    where
        L: Clone,
        I: Iterator<Item = (&'a str, L)>,
        F: FnMut(HashSet<String>, L),
    {
        // Padding for lookback: tags[i-3], tags[i-2], tags[i-1] are referenced by
        // the attribute builder. The real characters' tags follow the padding.
        let mut tags: Vec<&'static str> = vec!["U"; 3];
        let mut labels: Vec<L> = Vec::new();
        let mut text = String::new();

        for (word, label) in tokens {
            let char_count = word.chars().count();
            if char_count == 0 {
                continue;
            }
            tags.push("B");
            labels.push(label);
            for _ in 1..char_count {
                tags.push("O");
                labels.push(cont_label.clone());
            }
            text.push_str(word);
        }

        if tags.len() < 4 {
            return;
        }
        // Override the first real character's tag to "U" (Unknown) instead of "B",
        // because there is no preceding word boundary decision to reference at position 0.
        tags[3] = "U";

        let (chars, types) = self.sentence_context(&text);

        for i in 4..(chars.len() - 3) {
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            callback(attrs, labels[i - 3].clone());
        }
    }

    /// Processes a corpus string ("word word ..."), calling the callback for
    /// each character position with its attributes and boundary label
    /// (1 = word start, -1 = continuation).
    fn process_corpus<F>(&self, corpus: &str, callback: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_tokens(corpus.split(' ').map(|word| (word, 1i8)), -1i8, callback);
    }

    /// 品詞付きコーパスを処理し、各文字位置の特徴量とSegmentLabelを返す。
    ///
    /// コーパスフォーマット: "単語/品詞 単語/品詞 ..."
    /// 例: "これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT"
    fn process_corpus_with_pos<F>(&self, corpus: &str, callback: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        let tokens = corpus.split(' ').map(|token| {
            // "単語/品詞" をパース（スラッシュが無い場合はXとして扱う）
            let (word, pos) = match token.rfind('/') {
                Some(idx) => (&token[..idx], token[idx + 1..].parse().unwrap_or(Upos::X)),
                None => (token, Upos::X),
            };
            (word, SegmentLabel::B(pos))
        });
        self.process_tokens(tokens, SegmentLabel::O, callback);
    }

    /// Adds a corpus to the segmenter with a custom writer function.
    ///
    /// # Arguments
    /// * `corpus` - A string slice representing the corpus to be added.
    /// * `writer` - A closure that takes a HashSet of attributes and a label (i8) and writes them.
    ///
    /// # Note
    /// The writer function is called for each word in the corpus, allowing for custom handling of the attributes and labels.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese, None);
    /// segmenter.add_corpus_with_writer("テスト です", |attrs, label| {
    ///    println!("Attributes: {:?}, Label: {}", attrs, label);
    /// });
    /// ```
    ///
    /// This will process the corpus and call the writer function for each word, passing the attributes and label.
    pub fn add_corpus_with_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_corpus(corpus, writer);
    }

    /// Adds a corpus to the segmenter.
    ///
    /// # Arguments
    /// * `corpus` - A string slice representing the corpus to be added.
    ///
    /// This method processes the corpus, extracts features, and adds instances to the AdaBoost learner.
    /// If the corpus is empty, it does nothing.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let mut segmenter = Segmenter::new(Language::Japanese, None);
    /// segmenter.add_corpus("テスト です");
    /// ```
    /// This will process the corpus and add instances to the segmenter.
    pub fn add_corpus(&mut self, corpus: &str) {
        let mut instances = Vec::new();
        self.process_corpus(corpus, |attrs, label| {
            instances.push((attrs, label));
        });
        for (attrs, label) in instances {
            self.learner.add_instance(attrs, label);
        }
    }

    /// 品詞付きコーパスをAveraged Perceptronの学習データとして追加する。
    ///
    /// # Arguments
    /// * `corpus` - 品詞付きコーパス（"単語/品詞 単語/品詞 ..."形式）
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let mut segmenter = Segmenter::new(Language::Japanese, None);
    /// segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
    /// ```
    pub fn add_corpus_with_pos(&mut self, corpus: &str) {
        let mut instances = Vec::new();
        self.process_corpus_with_pos(corpus, |attrs, label| {
            instances.push((attrs, label));
        });
        let pos_learner = self.pos_learner.get_or_insert_with(AveragedPerceptron::new);
        for (attrs, label) in instances {
            pos_learner.add_instance(attrs, label.to_string());
        }
    }

    /// 品詞付きコーパスの特徴量をカスタムライター関数で処理する。
    ///
    /// # Arguments
    /// * `corpus` - 品詞付きコーパス（"単語/品詞 単語/品詞 ..."形式）
    /// * `writer` - 各文字位置の特徴量セットとSegmentLabelを受け取るクロージャ
    pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        self.process_corpus_with_pos(corpus, writer);
    }

    /// Segments a sentence into words.
    ///
    /// # Arguments
    /// * `sentence` - A string slice representing the sentence to be parsed.
    ///
    /// # Returns
    /// A vector of strings, where each string is a segmented word from the sentence.
    ///
    /// # Note
    /// The method processes the sentence character by character, using the AdaBoost learner to predict whether a character is the beginning of a new word or not.
    /// It constructs attributes based on the surrounding characters and their types, allowing for accurate segmentation.
    /// If the sentence is empty, it returns an empty vector.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    ///
    /// use litsea::adaboost::AdaBoost;
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let model_file =
    ///     PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join("RWCP.model");
    /// let mut learner = AdaBoost::new(0.01, 100);
    /// learner.load_model_from_path(&model_file).unwrap();
    ///
    /// let segmenter = Segmenter::new(Language::Japanese, Some(learner));
    /// let result = segmenter.segment("これはテストです。");
    /// assert_eq!(result, vec!["これ", "は", "テスト", "です", "。"]);
    /// ```
    /// This will segment the sentence into words and return them as a vector of strings.
    #[must_use]
    pub fn segment(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let learner = &self.learner;
        let (chars, types) = self.sentence_context(sentence);
        // Padding for lookback: tags[0..3] are fixed "U" (Unknown), and tags[3]
        // is also "U" since there is no boundary decision before the first character.
        let mut tags: Vec<&'static str> = vec!["U"; 4];

        // The bias is a sum over all model weights; compute it once per
        // sentence instead of once per character.
        let bias = learner.bias();

        let mut result = Vec::new();
        let mut word = chars[3].clone();
        for (i, ch) in chars.iter().enumerate().take(chars.len() - 3).skip(4) {
            // Sum the weights of the attributes directly instead of building
            // a HashSet per position.
            let mut score = bias;
            self.write_attributes(i, &tags, &chars, &types, &mut |attr| {
                score += learner.weight(attr);
            });
            if score >= 0.0 {
                result.push(std::mem::take(&mut word));
                tags.push("B");
            } else {
                tags.push("O");
            }
            word += ch;
        }
        result.push(word);
        result
    }

    /// 文を単語に分割し、各単語のUPOS品詞タグを同時に推定する。
    ///
    /// Averaged Perceptron（`pos_learner`）を使って各文字位置のラベル
    /// （`B-品詞`/`O`）を予測し、単語と品詞のペアを返す。
    /// 先頭単語の品詞は先頭文字位置の予測ラベルから決定する。
    ///
    /// # Arguments
    /// * `sentence` - 分割対象の文
    ///
    /// # Returns
    /// `Vec<(String, Upos)>` - 単語と品詞タグのペアのベクタ
    ///
    /// # Panics
    /// `pos_learner`が設定されていない場合にパニックする。
    /// `with_pos_learner()`または`add_corpus_with_pos()`で事前に設定すること。
    #[must_use]
    pub fn segment_with_pos(&self, sentence: &str) -> Vec<(String, Upos)> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let pos_learner = self
            .pos_learner
            .as_ref()
            .expect("pos_learner is not set. Use with_pos_learner() or add_corpus_with_pos().");

        let (chars, types) = self.sentence_context(sentence);
        let mut tags: Vec<&'static str> = vec!["U"; 4];
        // Attribute buffer reused across positions to amortize allocations.
        let mut attrs_buf: Vec<String> = Vec::new();

        // The first character always starts the first word; its predicted
        // label is used only to determine the first word's POS.
        self.collect_attributes(3, &tags, &chars, &types, &mut attrs_buf);
        let first_label: SegmentLabel =
            pos_learner.predict_slice(&attrs_buf).parse().unwrap_or(SegmentLabel::O);
        let mut current_pos = first_label.pos().unwrap_or(Upos::X);

        let mut result: Vec<(String, Upos)> = Vec::new();
        let mut word = chars[3].clone();

        for (i, ch) in chars.iter().enumerate().take(chars.len() - 3).skip(4) {
            self.collect_attributes(i, &tags, &chars, &types, &mut attrs_buf);
            let label: SegmentLabel =
                pos_learner.predict_slice(&attrs_buf).parse().unwrap_or(SegmentLabel::O);
            if label.is_boundary() {
                // 現在の単語を確定して結果に追加
                result.push((std::mem::take(&mut word), current_pos));
                current_pos = label.pos().unwrap_or(Upos::X);
                tags.push("B");
            } else {
                tags.push("O");
            }
            word += ch;
        }

        result.push((word, current_pos));
        result
    }

    /// Builds the attribute set for a specific index (used by the corpus
    /// processing pipeline, where the public callbacks expect a `HashSet`).
    fn get_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[String],
        types: &[&'static str],
    ) -> HashSet<String> {
        let mut attrs = HashSet::with_capacity(48);
        self.write_attributes(i, tags, chars, types, &mut |attr| {
            attrs.insert(attr.to_string());
        });
        attrs
    }

    /// Collects the attributes for a position into a reusable `Vec<String>`,
    /// reusing the existing string allocations where possible.
    fn collect_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[String],
        types: &[&'static str],
        out: &mut Vec<String>,
    ) {
        let mut idx = 0;
        self.write_attributes(i, tags, chars, types, &mut |attr| {
            if idx < out.len() {
                out[idx].clear();
                out[idx].push_str(attr);
            } else {
                out.push(attr.to_string());
            }
            idx += 1;
        });
        out.truncate(idx);
    }

    /// Writes each attribute for position `i` into a reusable buffer and
    /// passes it to `sink`. This is the single source of truth for the
    /// feature template.
    ///
    /// # Panics
    /// Panics if `i` is less than 3 or if `i + 2` exceeds the length of
    /// `chars` or `types`. Callers must ensure that `i` is within the valid
    /// range `[3, chars.len() - 3)`.
    fn write_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[String],
        types: &[&'static str],
        sink: &mut dyn FnMut(&str),
    ) {
        let w1 = &chars[i - 3];
        let w2 = &chars[i - 2];
        let w3 = &chars[i - 1];
        let w4 = &chars[i];
        let w5 = &chars[i + 1];
        let w6 = &chars[i + 2];
        let c1 = types[i - 3];
        let c2 = types[i - 2];
        let c3 = types[i - 1];
        let c4 = types[i];
        let c5 = types[i + 1];
        let c6 = types[i + 2];
        let p1 = tags[i - 3];
        let p2 = tags[i - 2];
        let p3 = tags[i - 1];

        let mut buf = String::with_capacity(32);
        macro_rules! attr {
            ($($arg:tt)*) => {{
                buf.clear();
                let _ = write!(buf, $($arg)*);
                sink(&buf);
            }};
        }

        attr!("UP1:{}", p1);
        attr!("UP2:{}", p2);
        attr!("UP3:{}", p3);
        attr!("BP1:{}{}", p1, p2);
        attr!("BP2:{}{}", p2, p3);
        attr!("UW1:{}", w1);
        attr!("UW2:{}", w2);
        attr!("UW3:{}", w3);
        attr!("UW4:{}", w4);
        attr!("UW5:{}", w5);
        attr!("UW6:{}", w6);
        attr!("BW1:{}{}", w2, w3);
        attr!("BW2:{}{}", w3, w4);
        attr!("BW3:{}{}", w4, w5);
        attr!("UC1:{}", c1);
        attr!("UC2:{}", c2);
        attr!("UC3:{}", c3);
        attr!("UC4:{}", c4);
        attr!("UC5:{}", c5);
        attr!("UC6:{}", c6);
        attr!("BC1:{}{}", c2, c3);
        attr!("BC2:{}{}", c3, c4);
        attr!("BC3:{}{}", c4, c5);
        attr!("TC1:{}{}{}", c1, c2, c3);
        attr!("TC2:{}{}{}", c2, c3, c4);
        attr!("TC3:{}{}{}", c3, c4, c5);
        attr!("TC4:{}{}{}", c4, c5, c6);
        attr!("UQ1:{}{}", p1, c1);
        attr!("UQ2:{}{}", p2, c2);
        attr!("UQ3:{}{}", p3, c3);
        attr!("BQ1:{}{}{}", p2, c2, c3);
        attr!("BQ2:{}{}{}", p2, c3, c4);
        attr!("BQ3:{}{}{}", p3, c2, c3);
        attr!("BQ4:{}{}{}", p3, c3, c4);
        attr!("TQ1:{}{}{}{}", p2, c1, c2, c3);
        attr!("TQ2:{}{}{}{}", p2, c2, c3, c4);
        attr!("TQ3:{}{}{}{}", p3, c1, c2, c3);
        attr!("TQ4:{}{}{}{}", p3, c2, c3, c4);

        // Language-specific features: char + char-type mixed features for Japanese and Chinese.
        // Korean is excluded because its uniform character types (SN/SF only) make these features noise.
        match self.language {
            Language::Japanese | Language::Chinese => {
                attr!("WC1:{}{}", w3, c4);
                attr!("WC2:{}{}", c3, w4);
                attr!("WC3:{}{}", w3, c3);
                attr!("WC4:{}{}", w4, c4);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_char_type_japanese() {
        let segmenter = Segmenter::new(Language::Japanese, None);

        assert_eq!(segmenter.char_type("あ"), "I"); // Hiragana
        assert_eq!(segmenter.char_type("漢"), "H"); // Kanji
        assert_eq!(segmenter.char_type("。"), "P"); // Punctuation
        assert_eq!(segmenter.char_type("A"), "A"); // Latin
        assert_eq!(segmenter.char_type("1"), "N"); // Digit
        assert_eq!(segmenter.char_type("@"), "O"); // Not matching any pattern
        assert_eq!(segmenter.char_type(""), "O"); // Empty string
    }

    #[test]
    fn test_char_type_chinese() {
        let segmenter = Segmenter::new(Language::Chinese, None);

        assert_eq!(segmenter.char_type("的"), "F"); // Function word
        assert_eq!(segmenter.char_type("中"), "C"); // CJK Unified
        assert_eq!(segmenter.char_type("国"), "C"); // CJK Unified
        assert_eq!(segmenter.char_type("。"), "P"); // Punctuation
        assert_eq!(segmenter.char_type("A"), "A"); // Latin
        assert_eq!(segmenter.char_type("5"), "N"); // Digit
        assert_eq!(segmenter.char_type("@"), "O"); // Other
    }

    #[test]
    fn test_char_type_korean() {
        let segmenter = Segmenter::new(Language::Korean, None);

        assert_eq!(segmenter.char_type("는"), "E"); // Particle (topic marker)
        assert_eq!(segmenter.char_type("가"), "SN"); // Hangul Syllable without 받침
        assert_eq!(segmenter.char_type("한"), "SF"); // Hangul Syllable with 받침
        assert_eq!(segmenter.char_type("ㄱ"), "G"); // Compatibility Jamo
        assert_eq!(segmenter.char_type("漢"), "H"); // Hanja
        assert_eq!(segmenter.char_type("A"), "A"); // Latin
        assert_eq!(segmenter.char_type("5"), "N"); // Digit
        assert_eq!(segmenter.char_type("@"), "O"); // Other
    }

    #[test]
    fn test_add_corpus_with_writer() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let sentence = "テスト です";
        let mut collected = Vec::new();

        segmenter.add_corpus_with_writer(sentence, |attrs, label| {
            collected.push((attrs, label));
        });

        // "テスト です" has 5 characters; the callback loop runs for indices 4..8
        // (skipping the first character at index 3), producing 4 instances.
        assert_eq!(collected.len(), 4);

        // Exactly one word boundary (at "で", start of second word "です")
        let positive_count = collected.iter().filter(|(_, label)| *label == 1).count();
        let negative_count = collected.iter().filter(|(_, label)| *label == -1).count();
        assert_eq!(positive_count, 1);
        assert_eq!(negative_count, 3);

        // Check that attributes contain expected keys
        let (attrs, _) = &collected[0];
        assert!(attrs.iter().any(|a| a.starts_with("UW")));
        assert!(attrs.iter().any(|a| a.starts_with("UC")));
    }

    #[test]
    fn test_add_corpus() {
        let mut segmenter = Segmenter::new(Language::Japanese, None);
        let sentence = "テスト です";
        segmenter.add_corpus(sentence);
        // Should not panic or add anything, just a smoke test
    }

    #[test]
    fn test_segment() {
        let sentence = "これはテストです。";

        let model_file =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join("RWCP.model");
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_path(&model_file).unwrap();

        let segmenter = Segmenter::new(Language::Japanese, Some(learner));

        let result = segmenter.segment(sentence);

        assert!(!result.is_empty());
        // "これはテストです。" segments into: "これ", "は", "テスト", "です", "。"
        // The RWCP model predicts word boundaries after these positions.
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], "これ");
        assert_eq!(result[1], "は");
        assert_eq!(result[2], "テスト");
        assert_eq!(result[3], "です");
        assert_eq!(result[4], "。");
    }

    #[test]
    fn test_add_sentence_empty() {
        let mut segmenter = Segmenter::new(Language::Japanese, None);
        segmenter.add_corpus("");
        // Should not panic or add anything
    }

    #[test]
    fn test_segment_empty_sentence() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let result = segmenter.segment("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_attributes() {
        let segmenter = Segmenter::new(Language::Japanese, None);

        let tags: Vec<&'static str> = vec!["U"; 7];

        let chars = vec![
            "B3".to_string(), // index 0
            "B2".to_string(), // index 1
            "B1".to_string(), // index 2
            "あ".to_string(), // index 3
            "い".to_string(), // index 4
            "う".to_string(), // index 5
            "E1".to_string(), // index 6
        ];

        let types: Vec<&'static str> = vec!["O", "O", "O", "O", "I", "I", "O"];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:い"));
        assert!(attrs.contains("UC4:I"));
        assert!(attrs.contains("UP3:U"));
        // Language-specific WC features (Japanese includes them)
        assert!(attrs.contains("WC1:あI")); // w3 + c4
        assert!(attrs.contains("WC2:Oい")); // c3 + w4
        assert!(attrs.contains("WC3:あO")); // w3 + c3
        assert!(attrs.contains("WC4:いI")); // w4 + c4
        // 38 base features (UW/BW/TW/UC/BC/TC/UP/BP/TP) + 4 WC features (Japanese-specific)
        assert_eq!(attrs.len(), 42);
    }

    #[test]
    fn test_collect_attributes_reuses_buffer() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec![
            "B3".to_string(),
            "B2".to_string(),
            "B1".to_string(),
            "あ".to_string(),
            "い".to_string(),
            "う".to_string(),
            "E1".to_string(),
        ];
        let types: Vec<&'static str> = vec!["O", "O", "O", "O", "I", "I", "O"];

        let mut buf: Vec<String> = Vec::new();
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut buf);
        assert_eq!(buf.len(), 42);
        // The slice contents must match the HashSet variant.
        let set = segmenter.get_attributes(4, &tags, &chars, &types);
        for attr in &buf {
            assert!(set.contains(attr), "missing from set: {}", attr);
        }
        // A second collection reuses the buffer and yields the same result.
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut buf);
        assert_eq!(buf.len(), 42);
    }

    #[test]
    #[should_panic]
    fn test_get_attributes_panics_index_too_low() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec![
            "B3".to_string(),
            "B2".to_string(),
            "B1".to_string(),
            "あ".to_string(),
            "い".to_string(),
            "う".to_string(),
            "E1".to_string(),
        ];
        let types: Vec<&'static str> = vec!["O"; 7];
        // i=2 is out of valid range [3, chars.len()-3); should panic on chars[i-3]
        let _ = segmenter.get_attributes(2, &tags, &chars, &types);
    }

    #[test]
    #[should_panic]
    fn test_get_attributes_panics_index_too_high() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec![
            "B3".to_string(),
            "B2".to_string(),
            "B1".to_string(),
            "あ".to_string(),
            "い".to_string(),
            "う".to_string(),
            "E1".to_string(),
        ];
        let types: Vec<&'static str> = vec!["O"; 7];
        // i=5 means i+2=7 which exceeds chars.len()=7; should panic on chars[i+2]
        let _ = segmenter.get_attributes(5, &tags, &chars, &types);
    }

    #[test]
    fn test_get_attributes_korean() {
        let segmenter = Segmenter::new(Language::Korean, None);

        let tags: Vec<&'static str> = vec!["U"; 7];

        let chars = vec![
            "B3".to_string(), // index 0
            "B2".to_string(), // index 1
            "B1".to_string(), // index 2
            "한".to_string(), // index 3
            "국".to_string(), // index 4
            "어".to_string(), // index 5
            "E1".to_string(), // index 6
        ];

        let types: Vec<&'static str> = vec!["O", "O", "O", "SF", "SF", "SN", "O"];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:국"));
        assert!(attrs.contains("UC4:SF"));
        // Korean does NOT include WC features
        assert!(!attrs.contains("WC1:한SF"));
        assert!(!attrs.contains("WC2:SF국"));
        // 38 base features only (Korean does not include WC word-character features)
        assert_eq!(attrs.len(), 38);
    }

    // --- 品詞推定関連テスト ---

    #[test]
    fn test_add_corpus_with_pos() {
        let mut segmenter = Segmenter::new(Language::Japanese, None);
        segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
        // pos_learnerが初期化される
        assert!(segmenter.pos_learner.is_some());
    }

    #[test]
    fn test_add_corpus_with_pos_writer() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let corpus = "テスト/NOUN です/AUX";
        let mut collected = Vec::new();

        segmenter.add_corpus_with_pos_writer(corpus, |attrs, label| {
            collected.push((attrs, label));
        });

        // "テストです" は5文字。ループは4..(8-3)=4..5で4回。
        // ただしlabels[0]はtags[3]に対応（先頭文字）で、ループはi=4から始まるのでlabel_idx=1から。
        // つまり "ス", "ト", "で", "す" の4文字分
        assert_eq!(collected.len(), 4);

        // "テスト/NOUN" → テ=B-NOUN, ス=O, ト=O
        // "です/AUX" → で=B-AUX, す=O
        // ループはi=4からなので: ス=O, ト=O, で=B-AUX, す=O
        let boundary_count = collected.iter().filter(|(_, l)| l.is_boundary()).count();
        assert_eq!(boundary_count, 1); // "で"のB-AUXのみ

        // B-AUXラベルの確認
        let b_aux = collected.iter().find(|(_, l)| l.is_boundary());
        assert!(b_aux.is_some());
        assert_eq!(b_aux.unwrap().1, SegmentLabel::B(Upos::AUX));
    }

    #[test]
    fn test_segment_with_pos() {
        let mut segmenter = Segmenter::new(Language::Japanese, None);

        // 学習データを複数回追加して学習
        for _ in 0..20 {
            segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
            segmenter.add_corpus_with_pos("私/PRON の/PART 猫/NOUN は/PART 可愛い/ADJ 。/PUNCT");
        }

        // Perceptronを学習
        let running = Arc::new(AtomicBool::new(true));
        segmenter.pos_learner.as_mut().unwrap().train(10, running);

        // 分割+品詞推定
        let result = segmenter.segment_with_pos("これはテストです。");
        assert!(!result.is_empty());

        // 結果が (単語, 品詞) のペアであることを確認
        for (word, pos) in &result {
            assert!(!word.is_empty());
            // 品詞はUposのいずれか
            let _ = pos.to_string();
        }
    }

    #[test]
    fn test_segment_with_pos_empty() {
        let segmenter = Segmenter::with_pos_learner(Language::Japanese, AveragedPerceptron::new());
        let result = segmenter.segment_with_pos("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_process_corpus_with_pos_empty() {
        let segmenter = Segmenter::new(Language::Japanese, None);
        let mut called = false;
        segmenter.add_corpus_with_pos_writer("", |_, _| {
            called = true;
        });
        assert!(!called);
    }
}
