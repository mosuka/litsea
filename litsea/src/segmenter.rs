use std::collections::HashSet;

use crate::adaboost::AdaBoost;
use crate::language::{CharTypePatterns, Language};

/// Segmenter struct for text segmentation using AdaBoost
/// It uses predefined patterns to classify characters and segment sentences into words.
pub struct Segmenter {
    pub language: Language,
    char_types: CharTypePatterns,
    pub learner: AdaBoost,
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
            char_types: language.char_type_patterns(),
            language,
            learner: learner.unwrap_or_else(|| AdaBoost::new(0.01, 100, 1)),
        }
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
    /// let char_type = segmenter.get_type("あ");
    /// assert_eq!(char_type, "I"); // Hiragana
    /// ```
    pub fn get_type(&self, ch: &str) -> &str {
        self.char_types.get_type(ch)
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
    /// let mut segmenter = Segmenter::new(Language::Japanese, None);
    /// segmenter.add_corpus_with_writer("テスト です", |attrs, label| {
    ///    println!("Attributes: {:?}, Label: {}", attrs, label);
    /// });
    /// ```
    ///
    /// This will process the corpus and call the writer function for each word, passing the attributes and label.
    ///
    /// # Returns
    /// Returns nothing.
    ///
    /// This method is useful for training the segmenter with a corpus of sentences, allowing it to learn how to segment text into words.
    pub fn add_corpus_with_writer<F>(&mut self, corpus: &str, mut writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        if corpus.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in corpus.split(' ') {
            if word.is_empty() {
                continue;
            }
            tags.push("B".to_string());
            for _ in 1..word.chars().count() {
                tags.push("O".to_string());
            }
            for ch in word.chars() {
                let s = ch.to_string();
                chars.push(s.clone());
                types.push(self.get_type(&s).to_string());
            }
        }
        if tags.len() < 4 {
            return;
        }
        tags[3] = "U".to_string();

        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        for i in 4..(chars.len() - 3) {
            let label = if tags[i] == "B" { 1 } else { -1 };
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            writer(attrs, label);
        }
    }

    /// Adds a corpus to the segmenter.
    ///
    /// # Arguments
    /// * `corpus` - A string slice representing the corpus to be added.
    ///
    /// This method processes the corpus, extracts features, and adds instances to the AdaBoost learner.
    /// If the corpus is empty, it does nothing.
    /// # Note
    /// The method constructs attributes based on the characters and their types, and uses the AdaBoost learner to add instances.
    /// If the corpus is too short or does not contain enough characters, it will not add any instances.
    /// The attributes are constructed based on the surrounding characters and their types, allowing for rich feature extraction.
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
    ///
    /// # Returns
    /// Returns nothing.
    ///
    /// This method is useful for training the segmenter with a corpus of sentences, allowing it to learn how to segment text into words.
    pub fn add_corpus(&mut self, corpus: &str) {
        if corpus.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in corpus.split(' ') {
            if word.is_empty() {
                continue;
            }
            tags.push("B".to_string());
            for _ in 1..word.chars().count() {
                tags.push("O".to_string());
            }
            for ch in word.chars() {
                let s = ch.to_string();
                chars.push(s.clone());
                types.push(self.get_type(&s).to_string());
            }
        }
        if tags.len() < 4 {
            return;
        }
        tags[3] = "U".to_string();

        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        for i in 4..(chars.len() - 3) {
            let label = if tags[i] == "B" { 1 } else { -1 };
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            // Call the learner for each instance; doing so individually avoids borrowing conflicts.
            self.learner.add_instance(attrs, label);
        }
    }

    /// Segments a sentence and segments it into words.
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
    /// # tokio_test::block_on(async {
    /// let model_file =
    ///     PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources").join("RWCP.model");
    /// let mut learner = AdaBoost::new(0.01, 100, 1);
    /// learner.load_model(model_file.to_str().unwrap()).await.unwrap();
    ///
    /// let segmenter = Segmenter::new(Language::Japanese, Some(learner));
    /// let result = segmenter.segment("これはテストです。");
    /// assert_eq!(result, vec!["これ", "は", "テスト", "です", "。"]);
    /// # });
    /// ```
    /// This will segment the sentence into words and return them as a vector of strings.
    pub fn segment(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let learner = &self.learner;
        let mut tags = vec!["U".to_string(); 4];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for ch in sentence.chars() {
            let s = ch.to_string();
            chars.push(s.clone());
            types.push(self.get_type(&s).to_string());
        }
        chars.extend_from_slice(&["E1".into(), "E2".into(), "E3".into()]);
        types.extend_from_slice(&["O".into(), "O".into(), "O".into()]);

        let mut result = Vec::new();
        let mut word = chars[3].clone();
        for i in 4..(chars.len() - 3) {
            let label = learner.predict(self.get_attributes(i, &tags, &chars, &types));
            if label >= 0 {
                result.push(word.clone());
                word.clear();
                tags.push("B".to_string());
            } else {
                tags.push("O".to_string());
            }
            word += &chars[i];
        }
        result.push(word);
        result
    }

    /// Gets the attributes for a specific index in the character and type arrays.
    ///
    /// # Arguments
    /// * `i` - The index for which to get the attributes.
    /// * `tags` - A slice of strings representing the tags for each character.
    /// * `chars` - A slice of strings representing the characters in the sentence.
    /// * `types` - A slice of strings representing the types of each character.
    ///
    /// # Returns
    /// A HashSet of strings representing the attributes for the specified index.
    ///
    /// # Note
    /// The attributes are constructed based on the surrounding characters and their types, allowing for rich feature extraction.
    /// This method is used internally by the segmenter to create features for each character in the sentence.
    ///
    /// This will return a set of attributes for the character at index 4, which is "い" in this case.
    fn get_attributes(
        &self,
        i: usize,
        tags: &[String],
        chars: &[String],
        types: &[String],
    ) -> HashSet<String> {
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

        let mut attrs: HashSet<String> = [
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

        // Language-specific features: char + char-type mixed features for Japanese and Chinese.
        // Korean is excluded because its uniform character types (SN/SF only) make these features noise.
        match self.language {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    #[test]
    fn test_get_type_japanese() {
        let segmenter = Segmenter::new(Language::Japanese, None);

        assert_eq!(segmenter.get_type("あ"), "I"); // Hiragana
        assert_eq!(segmenter.get_type("漢"), "H"); // Kanji
        assert_eq!(segmenter.get_type("。"), "P"); // Punctuation
        assert_eq!(segmenter.get_type("A"), "A"); // Latin
        assert_eq!(segmenter.get_type("1"), "N"); // Digit
        assert_eq!(segmenter.get_type("@"), "O"); // Not matching any pattern
    }

    #[test]
    fn test_get_type_chinese() {
        let segmenter = Segmenter::new(Language::Chinese, None);

        assert_eq!(segmenter.get_type("的"), "F"); // Function word
        assert_eq!(segmenter.get_type("中"), "C"); // CJK Unified
        assert_eq!(segmenter.get_type("国"), "C"); // CJK Unified
        assert_eq!(segmenter.get_type("。"), "P"); // Punctuation
        assert_eq!(segmenter.get_type("A"), "A"); // Latin
        assert_eq!(segmenter.get_type("5"), "N"); // Digit
        assert_eq!(segmenter.get_type("@"), "O"); // Other
    }

    #[test]
    fn test_get_type_korean() {
        let segmenter = Segmenter::new(Language::Korean, None);

        assert_eq!(segmenter.get_type("는"), "E"); // Particle (topic marker)
        assert_eq!(segmenter.get_type("가"), "SN"); // Hangul Syllable without 받침
        assert_eq!(segmenter.get_type("한"), "SF"); // Hangul Syllable with 받침
        assert_eq!(segmenter.get_type("ㄱ"), "G"); // Compatibility Jamo
        assert_eq!(segmenter.get_type("漢"), "H"); // Hanja
        assert_eq!(segmenter.get_type("A"), "A"); // Latin
        assert_eq!(segmenter.get_type("5"), "N"); // Digit
        assert_eq!(segmenter.get_type("@"), "O"); // Other
    }

    #[test]
    fn test_add_corpus_with_writer() {
        let mut segmenter = Segmenter::new(Language::Japanese, None);
        let sentence = "テスト です";
        let mut collected = Vec::new();

        segmenter.add_corpus_with_writer(sentence, |attrs, label| {
            collected.push((attrs, label));
        });

        // There should be as many instances as there are characters (excluding padding)
        assert!(!collected.is_empty());

        // Check that labels are either 1 or -1
        for (_, label) in &collected {
            assert!(*label == 1 || *label == -1);
        }

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

    #[tokio::test]
    async fn test_segment() {
        let sentence = "これはテストです。";

        let model_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../resources")
            .join("RWCP.model");
        let mut learner = AdaBoost::new(0.01, 100, 1);
        learner.load_model(model_file.to_str().unwrap()).await.unwrap();

        let segmenter = Segmenter::new(Language::Japanese, Some(learner));

        let result = segmenter.segment(sentence);

        assert!(!result.is_empty());
        assert_eq!(result.len(), 5); // Adjust based on expected segmentation
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

        let tags = vec!["U".to_string(); 7];

        let chars = vec![
            "B3".to_string(), // index 0
            "B2".to_string(), // index 1
            "B1".to_string(), // index 2
            "あ".to_string(), // index 3
            "い".to_string(), // index 4
            "う".to_string(), // index 5
            "E1".to_string(), // index 6
        ];

        let types = vec![
            "O".to_string(), // index 0
            "O".to_string(), // index 1
            "O".to_string(), // index 2
            "O".to_string(), // index 3
            "I".to_string(), // index 4
            "I".to_string(), // index 5
            "O".to_string(), // index 6
        ];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:い"));
        assert!(attrs.contains("UC4:I"));
        assert!(attrs.contains("UP3:U"));
        // Language-specific WC features (Japanese includes them)
        assert!(attrs.contains("WC1:あI")); // w3 + c4
        assert!(attrs.contains("WC2:Oい")); // c3 + w4
        assert!(attrs.contains("WC3:あO")); // w3 + c3
        assert!(attrs.contains("WC4:いI")); // w4 + c4
        assert_eq!(attrs.len(), 42); // 38 base + 4 WC
    }

    #[test]
    fn test_get_attributes_korean() {
        let segmenter = Segmenter::new(Language::Korean, None);

        let tags = vec!["U".to_string(); 7];

        let chars = vec![
            "B3".to_string(), // index 0
            "B2".to_string(), // index 1
            "B1".to_string(), // index 2
            "한".to_string(), // index 3
            "국".to_string(), // index 4
            "어".to_string(), // index 5
            "E1".to_string(), // index 6
        ];

        let types = vec![
            "O".to_string(),  // index 0
            "O".to_string(),  // index 1
            "O".to_string(),  // index 2
            "SF".to_string(), // index 3
            "SF".to_string(), // index 4
            "SN".to_string(), // index 5
            "O".to_string(),  // index 6
        ];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:국"));
        assert!(attrs.contains("UC4:SF"));
        // Korean does NOT include WC features
        assert!(!attrs.contains("WC1:한SF"));
        assert!(!attrs.contains("WC2:SF국"));
        assert_eq!(attrs.len(), 38); // 38 base only
    }
}
