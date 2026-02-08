use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

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
    pub fn extract(
        &mut self,
        corpus_path: &Path,
        features_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        // Read sentences from stdin
        // Each line is treated as a separate sentence
        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);

        // Create a file to write the features
        let features_file = File::create(features_path)?;
        let mut features = io::BufWriter::new(features_file);

        // learner function to write features
        // This function will be called for each word in the input sentences
        // It takes a set of attributes and a label, and writes them to stdout
        let mut learner = |attributes: HashSet<String>, label: i8| {
            let mut attrs: Vec<String> = attributes.into_iter().collect();
            attrs.sort();
            let mut line = vec![label.to_string()];
            line.extend(attrs);
            writeln!(features, "{}", line.join("\t")).expect("Failed to write features");
        };

        for line in corpus.lines() {
            match line {
                Ok(line) => {
                    let line = line.trim();
                    if !line.is_empty() {
                        self.segmenter.add_corpus_with_writer(line, &mut learner);
                    }
                }
                Err(err) => {
                    eprintln!("Error reading input: {}", err);
                }
            }
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

        // Check if the output contains tab-separated values
        assert!(output.contains("\t"), "Output should contain tab-separated values");

        Ok(())
    }
}
