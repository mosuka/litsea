//! Feature extraction from training corpora.
//!
//! Defines [`Extractor`], which converts a pre-segmented corpus file (or,
//! for POS tagging, a "word/POS"-tagged corpus) into the tab-separated
//! label + feature rows consumed by the trainers
//! ([`Trainer`](crate::trainer::Trainer) / [`PerceptronTrainer`](crate::trainer::PerceptronTrainer)).
//! [`Extractor::extract_two_stage`] extracts the same kind of rows for the
//! two-stage architecture instead, splitting them across the three files
//! read by [`TwoStageTrainer`](crate::trainer::TwoStageTrainer).

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

use rustc_hash::FxHashMap;

use crate::error::Result;
use crate::evaluation::parse_gold_pos_line;
use crate::language::Language;
use crate::segmenter::Segmenter;
use crate::two_stage::{TwoStageFeatureSet, sort_lexicon_entry, two_stage_paths, write_lexicon};
use crate::upos::{SegmentLabel, Upos};
use crate::word_features::write_word_features;

/// Extractor struct for processing text data and extracting features.
/// It reads pre-segmented sentences from a corpus file (the word boundaries
/// are given by the corpus itself) and writes the extracted training
/// features to one or more output files (a single features file for
/// [`extract`](Self::extract)/[`extract_tsv`](Self::extract_tsv), or three
/// files for [`extract_two_stage`](Self::extract_two_stage)). The internal
/// `Segmenter` is used only as the feature generator; its model is never
/// used to segment.
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
    /// label 1 (word start) or -1 (continuation); one row per character
    /// position of the sentence except the first (whose boundary label is
    /// degenerate: it always starts a word).
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
    /// character position except the first, with features in sorted order.
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

    /// Extracts features like [`extract`](Self::extract), but drops the 16
    /// tag-dependent templates (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`), which read
    /// the boundary decisions at the previous one to three positions.
    ///
    /// A model trained on the resulting features is *pointwise*: every
    /// position's score depends only on the input text, so `segment()`
    /// skips its sequential scoring pass entirely (issue #183). The
    /// bundled `korean.model` is trained this way; for Japanese and
    /// Chinese the tag features carry some quality (see the measured
    /// trade-off in the Pre-trained Models documentation), so this is an
    /// explicit speed-over-quality choice there.
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the input corpus file (the
    ///   space-separated format of [`extract`](Self::extract)).
    /// * `features_path` - The path to the output features file.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus file cannot be opened or read, or
    /// if the features file cannot be created or written.
    pub fn extract_tag_free(&self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_with_writer(line, |mut attrs, label| {
                attrs.retain(|a| !crate::packed_model::is_tag_dependent_feature(a));
                rows.push(Self::format_row(attrs, label));
            });
        })
    }

    /// Extracts features like [`extract_tsv`](Self::extract_tsv) (the
    /// space-preserving tab-separated corpus format), but drops the 16
    /// tag-dependent templates — the tag-free variant of
    /// [`extract_tag_free`](Self::extract_tag_free) for TSV corpora.
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the input tab-separated corpus file.
    /// * `features_path` - The path to the output features file.
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus file cannot be opened or read, or
    /// if the features file cannot be created or written.
    pub fn extract_tsv_tag_free(&self, corpus_path: &Path, features_path: &Path) -> Result<()> {
        let segmenter = &self.segmenter;
        Self::write_features(corpus_path, features_path, |line, rows| {
            segmenter.add_corpus_tsv_with_writer(line, |mut attrs, label| {
                attrs.retain(|a| !crate::packed_model::is_tag_dependent_feature(a));
                rows.push(Self::format_row(attrs, label));
            });
        })
    }

    /// Extracts two-stage training features (issue #147) from a POS-tagged
    /// corpus in a single pass: stage-1 boundary features (2-class labels
    /// `B`/`O`, using the same character-level feature templates as
    /// segmentation), stage-2 word-level features (UPOS labels, using the
    /// templates selected by `feature_set`), and the candidate-tag lexicon.
    ///
    /// Corpus format: `"word/POS word/POS ..."`, parsed with
    /// [`crate::evaluation::parse_gold_pos_line`] (last-`/`-wins, a
    /// slash-less token gets [`Upos::X`]).
    ///
    /// # Arguments
    /// * `corpus_path` - The path to the POS-tagged corpus file.
    /// * `output_prefix` - Base path for the three output files, written as
    ///   `{output_prefix}.stage1`, `{output_prefix}.stage2`, and
    ///   `{output_prefix}.lexicon`: stage-1 boundary features
    ///   (`label\tfeature\t...` rows, label `B` or `O`), stage-2 word-level
    ///   features (`label\tfeature\t...` rows, label a UPOS tag), and the
    ///   candidate-tag lexicon in the `litsea-two-stage` model's
    ///   `[lexicon]` section format (`surface\tTAG:count[,TAG:count...]`,
    ///   most-frequent-first). [`crate::trainer::TwoStageTrainer::new`]
    ///   reads the same three paths from the same prefix.
    /// * `feature_set` - Which stage-2 word templates to write; see
    ///   [`TwoStageFeatureSet`].
    ///
    /// # Returns
    /// Returns a Result indicating success or failure.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus file cannot be opened or read, or
    /// if any output file cannot be created or written.
    pub fn extract_two_stage(
        &self,
        corpus_path: &Path,
        output_prefix: &Path,
        feature_set: TwoStageFeatureSet,
    ) -> Result<()> {
        let segmenter = &self.segmenter;
        let language = segmenter.language();
        let (stage1_path, stage2_path, lexicon_path) = two_stage_paths(output_prefix);

        let corpus_file = File::open(corpus_path)?;
        let corpus = io::BufReader::new(corpus_file);
        let mut stage1_out = io::BufWriter::new(File::create(stage1_path)?);
        let mut stage2_out = io::BufWriter::new(File::create(stage2_path)?);
        let mut lexicon: FxHashMap<String, FxHashMap<Upos, u32>> = FxHashMap::default();

        let mut stage1_rows: Vec<String> = Vec::new();
        let mut stage2_feats: Vec<String> = Vec::new();
        for line in corpus.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Stage 1: character-level attribute generation over the
            // POS-tagged corpus, with the label collapsed to the boundary
            // class.
            segmenter.add_corpus_with_pos_writer(line, |attrs, label| {
                let boundary = match label {
                    SegmentLabel::B(_) => "B",
                    SegmentLabel::O => "O",
                };
                stage1_rows.push(Self::format_row(attrs, boundary));
            });
            for row in stage1_rows.drain(..) {
                writeln!(stage1_out, "{}", row)?;
            }

            // Stage 2 + lexicon: one row per word, keyed by its UPOS tag.
            let tokens = parse_gold_pos_line(line);
            let sent: Vec<char> = tokens.iter().flat_map(|(w, _)| w.chars()).collect();
            let type_ids: Vec<u8> = sent.iter().map(|&c| language.char_type_id(c)).collect();
            let mut start = 0usize;
            for (surface, tag) in &tokens {
                let wlen = surface.chars().count();
                if wlen == 0 {
                    continue;
                }
                let end = start + wlen;
                stage2_feats.clear();
                write_word_features(
                    language,
                    &sent,
                    &type_ids,
                    start,
                    end,
                    |tid| feature_set.includes(tid),
                    &mut |f| stage2_feats.push(f),
                );
                writeln!(stage2_out, "{}\t{}", tag, stage2_feats.join("\t"))?;
                *lexicon.entry(surface.clone()).or_default().entry(*tag).or_insert(0) += 1;
                start = end;
            }
        }
        stage1_out.flush()?;
        stage2_out.flush()?;

        let lexicon: FxHashMap<String, Vec<(Upos, u32)>> = lexicon
            .into_iter()
            .map(|(surface, counts)| {
                let mut entry: Vec<(Upos, u32)> = counts.into_iter().collect();
                sort_lexicon_entry(&mut entry);
                (surface, entry)
            })
            .collect();
        let mut lexicon_out = io::BufWriter::new(File::create(lexicon_path)?);
        write_lexicon(&lexicon, &mut lexicon_out)?;
        lexicon_out.flush()?;

        Ok(())
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
    fn test_extract_tag_free() -> Result<()> {
        use crate::packed_model::is_tag_dependent_feature;

        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "これ は テスト です 。")?;
        writeln!(corpus_file, "別 の 文 も あり ます 。")?;
        corpus_file.as_file().sync_all()?;

        let full_file = NamedTempFile::new()?;
        let tag_free_file = NamedTempFile::new()?;
        let extractor = Extractor::default();
        extractor.extract(corpus_file.path(), full_file.path())?;
        extractor.extract_tag_free(corpus_file.path(), tag_free_file.path())?;

        let mut full = String::new();
        File::open(full_file.path())?.read_to_string(&mut full)?;
        let mut tag_free = String::new();
        File::open(tag_free_file.path())?.read_to_string(&mut tag_free)?;

        // The tag-free output must be exactly the full output with the
        // tag-dependent columns removed: same rows, same labels, same
        // remaining features in the same (sorted) order.
        let full_rows: Vec<&str> = full.lines().collect();
        let tag_free_rows: Vec<&str> = tag_free.lines().collect();
        assert_eq!(full_rows.len(), tag_free_rows.len());
        for (full_row, tag_free_row) in full_rows.iter().zip(&tag_free_rows) {
            let expected: Vec<&str> = full_row
                .split('\t')
                .enumerate()
                .filter(|(i, f)| *i == 0 || !is_tag_dependent_feature(f))
                .map(|(_, f)| f)
                .collect();
            let actual: Vec<&str> = tag_free_row.split('\t').collect();
            assert_eq!(actual, expected, "tag-free row diverged: {tag_free_row}");
            assert!(
                actual.iter().skip(1).all(|f| !is_tag_dependent_feature(f)),
                "tag-dependent feature left in: {tag_free_row}"
            );
        }
        // Sanity: the filter actually removed something.
        assert!(full.len() > tag_free.len());
        Ok(())
    }

    #[test]
    fn test_extract_tsv_tag_free() -> Result<()> {
        use crate::language::Language;
        use crate::packed_model::is_tag_dependent_feature;

        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "나는\t \t봄\t.")?;
        corpus_file.as_file().sync_all()?;

        let features_file = NamedTempFile::new()?;
        let extractor = Extractor::new(Language::Korean);
        extractor.extract_tsv_tag_free(corpus_file.path(), features_file.path())?;

        let mut output = String::new();
        File::open(features_file.path())?.read_to_string(&mut output)?;
        // "나는 봄." = 5 chars -> 4 rows, same as extract_tsv.
        assert_eq!(output.lines().count(), 4);
        for line in output.lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            assert!(fields.len() >= 2, "Line should have label + features: {line}");
            assert!(fields[0] == "1" || fields[0] == "-1");
            assert!(
                fields[1..].iter().all(|f| !is_tag_dependent_feature(f)),
                "tag-dependent feature left in: {line}"
            );
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
    fn test_extract_two_stage() -> Result<()> {
        use std::collections::HashMap;

        // "これ" and "テスト" and "。" each repeat with the same tag, so the
        // lexicon should accumulate their counts.
        let mut corpus_file = NamedTempFile::new()?;
        writeln!(corpus_file, "これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT")?;
        writeln!(corpus_file, "これ/PRON も/PART テスト/NOUN 。/PUNCT")?;
        corpus_file.as_file().sync_all()?;

        let dir = tempfile::tempdir()?;
        let prefix = dir.path().join("out");

        let extractor = Extractor::default();
        extractor.extract_two_stage(corpus_file.path(), &prefix, TwoStageFeatureSet::Fast)?;

        let mut stage1 = String::new();
        File::open(dir.path().join("out.stage1"))?.read_to_string(&mut stage1)?;
        let mut stage2 = String::new();
        File::open(dir.path().join("out.stage2"))?.read_to_string(&mut stage2)?;
        let mut lexicon = String::new();
        File::open(dir.path().join("out.lexicon"))?.read_to_string(&mut lexicon)?;

        // Stage 1: one row per character (the POS corpus pipeline emits a
        // row at the first position too), label collapsed to the boundary
        // class.
        // Line 1 has 9 chars ("これはテストです。"), line 2 has 7
        // ("これもテスト。").
        let stage1_lines: Vec<&str> = stage1.lines().collect();
        assert_eq!(stage1_lines.len(), 16);
        for line in &stage1_lines {
            let label = line.split('\t').next().unwrap();
            assert!(label == "B" || label == "O", "unexpected stage-1 label: {label}");
        }

        // Stage 2: one row per word, label a UPOS tag; the Fast feature set
        // excludes FC/LC, so no "FC:"/"LC:" feature should appear, but WS:
        // (always included in Fast) should.
        let stage2_lines: Vec<&str> = stage2.lines().collect();
        assert_eq!(stage2_lines.len(), 9);
        let mut tag_counts: HashMap<&str, usize> = HashMap::new();
        for line in &stage2_lines {
            let mut fields = line.split('\t');
            let label = fields.next().unwrap();
            assert!(label.parse::<crate::upos::Upos>().is_ok(), "not a UPOS tag: {label}");
            *tag_counts.entry(label).or_insert(0) += 1;
            let feats: Vec<&str> = fields.collect();
            assert!(feats.iter().any(|f| f.starts_with("WS:")), "missing WS feature in {line}");
            assert!(
                !feats.iter().any(|f| f.starts_with("FC:") || f.starts_with("LC:")),
                "Fast feature set should not emit FC:/LC: in {line}"
            );
        }
        assert_eq!(tag_counts.get("PRON"), Some(&2));
        assert_eq!(tag_counts.get("NOUN"), Some(&2));
        assert_eq!(tag_counts.get("PUNCT"), Some(&2));
        assert_eq!(tag_counts.get("PART"), Some(&2));
        assert_eq!(tag_counts.get("AUX"), Some(&1));

        // Lexicon: one line per unique surface, counts accumulated across
        // both occurrences of "これ"/"テスト"/"。".
        let mut entries: HashMap<&str, &str> = HashMap::new();
        for line in lexicon.lines() {
            let (surface, tags) = line.split_once('\t').unwrap();
            entries.insert(surface, tags);
        }
        assert_eq!(entries.len(), 6);
        assert_eq!(entries.get("これ"), Some(&"PRON:2"));
        assert_eq!(entries.get("テスト"), Some(&"NOUN:2"));
        assert_eq!(entries.get("。"), Some(&"PUNCT:2"));
        assert_eq!(entries.get("は"), Some(&"PART:1"));
        assert_eq!(entries.get("も"), Some(&"PART:1"));
        assert_eq!(entries.get("です"), Some(&"AUX:1"));

        Ok(())
    }
}
