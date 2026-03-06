use std::cell::RefCell;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::language::Language;
use crate::segmenter::Segmenter;
use crate::upos::SegmentLabel;

/// Extractor struct for processing text data and extracting features.
/// It reads sentences from a corpus file, segments them into words,
/// and writes the extracted features to a specified output file.
pub struct Extractor {
    segmenter: Segmenter,
}

impl Default for Extractor {
    /// Creates a new instance of [`Extractor`] with default settings (Japanese).
    ///
    /// # Returns
    /// Returns a new instance of `Extractor`.
    fn default() -> Self {
        Self::new(Language::default())
    }
}

impl Extractor {
    /// Creates a new instance of [`Extractor`].
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    ///
    /// # Returns
    /// Returns a new instance of `Extractor` with a new `Segmenter` for the specified language.
    pub fn new(language: Language) -> Self {
        Extractor {
            segmenter: Segmenter::new(language, None),
        }
    }

    /// Extracts features from a corpus file and writes them to a specified output file.
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the input corpus file containing sentences.
    /// * `features_path` - The path to the output file where extracted features will be written.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    pub fn extract(
        &mut self,
        corpus_path: &Path,
        features_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        // Read sentences from the corpus file.
        // Each line is treated as a separate sentence.
        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);

        // Create a file to write the features
        let features_file = File::create(features_path)?;
        let mut features = io::BufWriter::new(features_file);

        // Capture write errors from the closure via RefCell
        let write_error: RefCell<Option<io::Error>> = RefCell::new(None);

        // Learner function to write features
        // It takes a set of attributes and a label, and writes them to the output file
        let mut learner = |attributes: HashSet<String>, label: i8| {
            if write_error.borrow().is_some() {
                return;
            }
            let mut attrs: Vec<String> = attributes.into_iter().collect();
            attrs.sort();
            let mut line = vec![label.to_string()];
            line.extend(attrs);
            if let Err(e) = writeln!(features, "{}", line.join("\t")) {
                *write_error.borrow_mut() = Some(e);
            }
        };

        for line in corpus.lines() {
            let line = line?;
            let line = line.trim();
            if !line.is_empty() {
                self.segmenter.add_corpus_with_writer(line, &mut learner);
            }
            // Stop processing further lines if a write error occurred.
            if write_error.borrow().is_some() {
                break;
            }
        }

        if let Some(e) = write_error.into_inner() {
            return Err(Box::new(e));
        }

        Ok(())
    }

    /// 品詞付きコーパスから特徴量を抽出してファイルに書き出す。
    ///
    /// コーパスフォーマット: 各行が "単語/品詞 単語/品詞 ..." 形式。
    /// 出力フォーマット: 各行が "ラベル\t特徴1\t特徴2\t..." 形式。
    /// ラベルは "B-NOUN", "B-VERB", ..., "O" のSegmentLabel文字列。
    ///
    /// # Arguments
    /// * `corpus_path` - 品詞付きコーパスファイルのパス
    /// * `features_path` - 特徴量出力ファイルのパス
    pub fn extract_with_pos(
        &mut self,
        corpus_path: &Path,
        features_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);

        let features_file = File::create(features_path)?;
        let mut features = io::BufWriter::new(features_file);

        let write_error: RefCell<Option<io::Error>> = RefCell::new(None);

        let mut writer = |attributes: HashSet<String>, label: SegmentLabel| {
            if write_error.borrow().is_some() {
                return;
            }
            let mut attrs: Vec<String> = attributes.into_iter().collect();
            attrs.sort();
            let mut line = vec![label.to_string()];
            line.extend(attrs);
            if let Err(e) = writeln!(features, "{}", line.join("\t")) {
                *write_error.borrow_mut() = Some(e);
            }
        };

        for line in corpus.lines() {
            let line = line?;
            let line = line.trim();
            if !line.is_empty() {
                self.segmenter.add_corpus_with_pos_writer(line, &mut writer);
            }
            if write_error.borrow().is_some() {
                break;
            }
        }

        if let Some(e) = write_error.into_inner() {
            return Err(Box::new(e));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::{Read, Write};

    use tempfile::NamedTempFile;

    #[test]
    fn test_extract() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary file to simulate the corpus input
        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "これ は テスト です 。")?;
        writeln!(corpus_file, "別 の 文 も あり ます 。")?;
        corpus_file.as_file().sync_all()?;

        // Create a temporary file for the features output
        let features_file = NamedTempFile::new()?;

        // Create an instance of Extractor and extract features
        let mut extractor = Extractor::default();
        extractor.extract(corpus_file.path(), features_file.path())?;

        // Read the output from the features file
        let mut output = String::new();
        File::open(features_file.path())?.read_to_string(&mut output)?;

        // Check if the output is not empty
        assert!(!output.is_empty(), "Extracted features should not be empty");

        // Validate the output format line by line
        for line in output.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            // Each line must have at least a label and one feature
            assert!(fields.len() >= 2, "Line should have label + features: {line}");
            // First field is the label: must be "1" (boundary) or "-1" (non-boundary)
            let label = fields[0];
            assert!(label == "1" || label == "-1", "Label should be 1 or -1, got: {label}");
            // Remaining fields are feature names (non-empty strings)
            for feat in &fields[1..] {
                assert!(!feat.is_empty(), "Feature name should not be empty");
            }
        }

        Ok(())
    }

    #[test]
    fn test_extract_with_pos() -> Result<(), Box<dyn std::error::Error>> {
        // 品詞付きコーパスの作成
        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT")?;
        writeln!(corpus_file, "私/PRON の/PART 猫/NOUN 。/PUNCT")?;
        corpus_file.as_file().sync_all()?;

        let features_file = NamedTempFile::new()?;

        let mut extractor = Extractor::default();
        extractor.extract_with_pos(corpus_file.path(), features_file.path())?;

        let mut output = String::new();
        File::open(features_file.path())?.read_to_string(&mut output)?;

        assert!(!output.is_empty(), "Extracted features should not be empty");

        // ラベルが SegmentLabel 形式であることを確認
        for line in output.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            assert!(fields.len() >= 2, "Line should have label + features: {line}");
            let label = fields[0];
            // ラベルは "O" または "B-品詞" のいずれか
            assert!(
                label == "O" || label.starts_with("B-"),
                "Label should be 'O' or 'B-<POS>', got: {label}"
            );
            // B-品詞の場合、品詞がUPOSタグであることを確認
            if let Some(pos) = label.strip_prefix("B-") {
                assert!(
                    pos.parse::<crate::upos::Upos>().is_ok(),
                    "Invalid UPOS tag in label: {label}"
                );
            }
        }

        Ok(())
    }
}
