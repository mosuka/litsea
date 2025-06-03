use crate::adaboost::AdaBoost;
use regex::Regex;
use std::collections::HashSet;

/// Segmenter struct for text segmentation using AdaBoost
/// It uses predefined patterns to classify characters and segments sentences into words.
pub struct Segmenter {
    patterns: Vec<(Regex, &'static str)>,
    pub learner: AdaBoost,
}

impl Segmenter {
    /// creates a new instance of [`Segmenter`].
    ///
    /// # Arguments
    /// * `learner` - An optional AdaBoost instance. If None, a default AdaBoost instance is created.
    ///
    /// # Returns
    /// A new Segmenter instance with the specified or default AdaBoost learner.
    pub fn new(learner: Option<AdaBoost>) -> Self {
        let patterns = vec![
            // (
            //     Regex::new(r"[一二三四五六七八九十百千万億兆]").unwrap(),
            //     "M",
            // ),
            // (Regex::new(r"[一-龠々〆ヵヶ]").unwrap(), "H"),
            // (Regex::new(r"[ぁ-ん]").unwrap(), "I"),
            // (Regex::new(r"[ァ-ヴーｱ-ﾝﾞｰ]").unwrap(), "K"),
            // (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
            // (Regex::new(r"[0-9０-９]").unwrap(), "N"),

            // Japanese Kanji numbers
            (Regex::new(r"[一二三四五六七八九十百千万億兆]").unwrap(), "M"),
            // Japanese Kanji
            (Regex::new(r"[一-龠々〆ヵヶ]").unwrap(), "J"),
            // Chinese Kanji (CJK Unified Ideographs)
            (Regex::new(r"[㐀-䶵一-鿿]").unwrap(), "M"),
            // Hangul (Korean)
            (Regex::new(r"[가-힣]").unwrap(), "K"),
            // Hiragana (Japanese)
            (Regex::new(r"[ぁ-ん]").unwrap(), "I"),
            // Katakana (Japanese)
            (Regex::new(r"[ァ-ヴーｱ-ﾝﾞﾟ]").unwrap(), "K"),
            // Latin alphabet (ASCII + full-width)
            (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
            // Numbers (ASCII + full-width)
            (Regex::new(r"[0-9０-９]").unwrap(), "N"),
            // Vietnamese Extended Latin
            (Regex::new(r"[À-ſ]").unwrap(), "V"),
            // Thai script
            (Regex::new(r"[ก-๛]").unwrap(), "T"),
        ];

        Segmenter {
            patterns,
            learner: learner.unwrap_or_else(|| AdaBoost::new(0.01, 100, 1)),
        }
    }

    /// Gets the type of a character based on predefined patterns.
    ///
    /// # Arguments
    /// * `ch` - A string slice representing a single character.
    ///
    /// # Returns
    /// A static string representing the type of the character, such as "M", "H", "I", "K", "A", "N", or "O" (for others).
    pub fn get_type(&self, ch: &str) -> &str {
        for (pattern, label) in &self.patterns {
            if pattern.is_match(ch) {
                return label;
            }
        }
        "O" // Other
    }

    /// Adds a sentence to the segmenter with a custom writer function.
    ///
    /// # Arguments
    /// * `sentence` - A string slice representing the sentence to be added.
    /// * `writer` - A closure that takes a `HashSet<String>` of attributes and a label (`i8`) as arguments.
    ///
    /// This closure is called for each instance created from the sentence.
    /// This method processes the sentence, extracts features, and calls the writer function for each instance.
    /// It constructs attributes based on the characters and their types, and uses the AdaBoost learner to add instances.
    pub fn add_sentence_with_writer<F>(&mut self, sentence: &str, mut writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        if sentence.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in sentence.split(' ') {
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

    /// Adds a sentence to the segmenter for training.
    ///
    /// # Arguments
    /// * `sentence` - A string slice representing the sentence to be added.
    ///
    /// This method processes the sentence, extracts features, and adds them to the AdaBoost learner.
    /// It constructs attributes based on the characters and their types, and uses the AdaBoost learner to add instances.
    /// If the sentence is empty or too short, it does nothing.
    pub fn add_sentence(&mut self, sentence: &str) {
        if sentence.is_empty() {
            return;
        }
        let mut tags = vec!["U".to_string(); 3];
        let mut chars = vec!["B3".to_string(), "B2".to_string(), "B1".to_string()];
        let mut types = vec!["O".to_string(); 3];

        for word in sentence.split(' ') {
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

    /// Parses a sentence and segments it into words.
    ///
    /// # Arguments
    /// * `sentence` - A string slice representing the sentence to be parsed.
    ///
    /// # Returns
    /// A vector of strings, where each string is a segmented word from the sentence.
    pub fn parse(&self, sentence: &str) -> Vec<String> {
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

        [
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
            format!("TW1:{}{}{}", w1, w2, w3),
            format!("TW2:{}{}{}", w2, w3, w4),
            format!("TW3:{}{}{}", w3, w4, w5),
            format!("TW4:{}{}{}", w4, w5, w6),
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
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn test_segmenter() {
        let model_file =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("./resources").join("RWCP.model");

        let mut learner = AdaBoost::new(0.01, 100, 1);
        learner.load_model(model_file.as_path()).unwrap();

        let mut segmenter = Segmenter::new(Some(learner));
        let sentence = "これはテストです。";
        segmenter.add_sentence(sentence);
        let result = segmenter.parse(sentence);
        assert!(!result.is_empty());
        assert_eq!(result.len(), 5); // Adjust based on expected segmentation
    }

    #[test]
    fn test_get_type() {
        let segmenter = Segmenter::new(None);
        assert_eq!(segmenter.get_type("あ"), "I"); // Hiragana
        assert_eq!(segmenter.get_type("漢"), "H"); // Kanji
        assert_eq!(segmenter.get_type("A"), "A"); // Latin
        assert_eq!(segmenter.get_type("1"), "N"); // Digit
    }
}
