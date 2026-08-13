//! Feature extraction from training corpora.
//!
//! Defines [`Extractor`], which converts a pre-segmented corpus file (or,
//! for POS tagging, a "word/POS"-tagged corpus) into the tab-separated
//! label + feature rows consumed by the trainers
//! ([`Trainer`](crate::trainer::Trainer) / [`PosTrainer`](crate::trainer::PosTrainer)).

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use crate::error::Result;
use crate::language::Language;
use crate::segmenter::Segmenter;

/// Extractor struct for processing text data and extracting features.
/// It reads pre-segmented sentences from a corpus file (the word boundaries
/// are given by the corpus itself) and writes the extracted training
/// features to a specified output file. The internal `Segmenter` is used
/// only as the feature generator; its model is never used to segment.
#[derive(Debug)]
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
            segmenter: Segmenter::new(language),
        }
    }

    /// Extracts features from a corpus file and writes them to a specified output file.
    ///
    /// Corpus format: one sentence per line, each line consisting of
    /// space-separated words (the words define the boundary labels).
    /// Output format: each line is "label\tfeature1\tfeature2\t..." with
    /// label 1 (word start) or -1 (continuation).
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the input corpus file containing sentences.
    /// * `features_path` - The path to the output file where extracted features will be written.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus file cannot be opened or read, or
    /// if the features file cannot be created or written.
    pub fn extract(&self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_with_writer(line, |attrs, label| {
                rows.push(Self::format_row(attrs, label));
            });
        })
    }

    /// Extracts features from a tab-separated corpus file and writes them to
    /// a specified output file.
    ///
    /// Corpus format: one sentence per line, tokens separated by tab
    /// characters. A token may be a literal space `" "` (e.g. the
    /// inter-eojeol space in Korean), which preserves the original spacing of
    /// the sentence in the training text so the model can learn from space
    /// characters as boundary context (issue #152). Output format is the same
    /// as [`extract`](Self::extract): one `label\tfeature\t...` row per
    /// character position, with features in sorted order.
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the input tab-separated corpus file.
    /// * `features_path` - The path to the output file where extracted features will be written.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an error if the corpus file cannot be read or the features
    /// file cannot be created or written to.
    pub fn extract_tsv(&self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_tsv_with_writer(line, |attrs, label| {
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
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus file cannot be opened or read, or
    /// if the features file cannot be created or written.
    pub fn extract_with_pos(&self, corpus_path: &Path, features_path: &Path) -> Result<()> {
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
        let extractor = Extractor::default();
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
    fn test_extract_tsv() -> Result<()> {
        use crate::language::Language;

        // Space-preserving TSV corpus: tab-separated tokens, with the
        // inter-eojeol space as its own token.
        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "나는\t \t봄\t.")?;
        writeln!(corpus_file, "한국어\t \t분석기\t.")?;
        corpus_file.as_file().sync_all()?;

        let features_file = NamedTempFile::new()?;

        let extractor = Extractor::new(Language::Korean);
        extractor.extract_tsv(corpus_file.path(), features_file.path())?;

        let mut output = String::new();
        File::open(features_file.path())?.read_to_string(&mut output)?;

        assert!(!output.is_empty(), "Extracted features should not be empty");

        // Sentence 1: "나는 봄." = 5 chars -> 4 rows; sentence 2:
        // "한국어 분석기." = 8 chars -> 7 rows.
        assert_eq!(output.lines().count(), 11);

        let mut boundary_labels = 0usize;
        let mut space_features = 0usize;
        for line in output.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            assert!(fields.len() >= 2, "Line should have label + features: {line}");
            let label = fields[0];
            assert!(label == "1" || label == "-1", "Label should be 1 or -1, got: {label}");
            if label == "1" {
                boundary_labels += 1;
            }
            // The preserved space must show up inside character-context
            // features (e.g. "UW3: ").
            if fields[1..].iter().any(|f| f.starts_with("UW") && f.ends_with(' ')) {
                space_features += 1;
            }
        }
        // Boundaries: sentence 1 has 3 (space, 봄, '.'); sentence 2 has 3
        // (space, 분석기 start, '.').
        assert_eq!(boundary_labels, 6);
        assert!(space_features > 0, "expected space characters inside UW context features");

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

        let extractor = Extractor::default();
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
