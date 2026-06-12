use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::error::Result;
use crate::language::Language;
use crate::segmenter::Segmenter;

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
    pub fn extract(&mut self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_with_writer(line, |attrs, label| {
                rows.push(Self::format_row(attrs, label));
            });
        })
    }

    /// Extracts features from a POS-tagged corpus and writes them to a file.
    ///
    /// Corpus format: each line is "word/POS word/POS ...".
    /// Output format: each line is "label\tfeature1\tfeature2\t...".
    /// Labels are SegmentLabel strings: "B-NOUN", "B-VERB", ..., "O".
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the POS-tagged corpus file
    /// * `features_path` - The path to the features output file
    pub fn extract_with_pos(&mut self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_with_pos_writer(line, |attrs, label| {
                rows.push(Self::format_row(attrs, label));
            });
        })
    }

    /// Shared extraction pipeline: reads the corpus line by line, lets
    /// `process_line` convert each non-empty line into formatted feature rows,
    /// and writes the rows to the features file.
    fn write_features<P>(
        corpus_path: &Path,
        features_path: &Path,
        mut process_line: P,
    ) -> Result<()>
    where
        P: FnMut(&str, &mut Vec<String>),
    {
        // Read sentences from the corpus file.
        // Each line is treated as a separate sentence.
        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);

        // Create a file to write the features
        let features_file = File::create(features_path)?;
        let mut features = io::BufWriter::new(features_file);

        let mut rows: Vec<String> = Vec::new();
        for line in corpus.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            process_line(line, &mut rows);
            for row in rows.drain(..) {
                writeln!(features, "{}", row)?;
            }
        }

        Ok(())
    }

    /// Formats one feature row: the label followed by the sorted attributes,
    /// tab-separated.
    fn format_row(attributes: HashSet<String>, label: impl fmt::Display) -> String {
        let mut attrs: Vec<String> = attributes.into_iter().collect();
        attrs.sort();
        let mut row = label.to_string();
        for attr in attrs {
            row.push('\t');
            row.push_str(&attr);
        }
        row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::io::{Read, Write};

    use tempfile::NamedTempFile;

    #[test]
    fn test_extract() -> Result<()> {
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
    fn test_extract_with_pos() -> Result<()> {
        // Create a POS-tagged corpus
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

        // Verify the labels follow the SegmentLabel format
        for line in output.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            assert!(fields.len() >= 2, "Line should have label + features: {line}");
            let label = fields[0];
            // The label is either "O" or "B-<POS>"
            assert!(
                label == "O" || label.starts_with("B-"),
                "Label should be 'O' or 'B-<POS>', got: {label}"
            );
            // For B-<POS>, verify the POS is a valid UPOS tag
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
