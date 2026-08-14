//! Two-stage model container and file format (`litsea-two-stage v1`).
//!
//! A two-stage model performs word segmentation with a binary boundary
//! classifier (stage 1, an [`AdaBoost`]-format scalar-weight model) and then
//! assigns a POS tag to each word (stage 2) using a lexicon of candidate
//! tags plus a global multiclass [`AveragedPerceptron`] over word-level
//! features. This module defines the on-disk container that bundles the
//! three parts into a single file, and the [`TwoStageLearner`] type that
//! holds them in memory. The runtime that consumes the learner lives in the
//! segmenter (see issue #147 for the architecture).
//!
//! # File format
//!
//! A plain-text file with a fixed magic first line and marker-delimited
//! sections in a fixed order:
//!
//! ```text
//! litsea-two-stage v1
//! [params]                    <- optional
//! dominance\t0.99
//! [stage1]
//! <AdaBoost model format: "feature\tweight" lines + one bias line>
//! [lexicon]
//! <surface>\t<TAG>:<count>[,<TAG>:<count>...]
//! [stage2]
//! <averaged-perceptron model format: class count, class names, weights>
//! ```
//!
//! - The `[stage1]` and `[stage2]` sections embed the existing model
//!   formats verbatim and are parsed by the existing loaders.
//! - Lexicon lines map a word surface to the UPOS tags observed for it in
//!   the training corpus with their occurrence counts, most frequent first
//!   (ties broken by tag name). Surfaces may contain any character except
//!   tab and newline and are not trimmed, so whitespace tokens (e.g. the
//!   Korean space token) stay representable.
//! - `dominance` is the classifier-skip threshold: at inference time a
//!   known surface whose most frequent tag covers at least this fraction of
//!   its training occurrences is tagged without invoking the stage-2
//!   classifier. It defaults to [`DEFAULT_DOMINANCE`] when the `[params]`
//!   section is absent.
//!
//! Section markers cannot collide with content lines: every weight and
//! lexicon line contains a tab, and the only single-token content lines are
//! the AdaBoost bias (numeric), the perceptron class count (numeric), and
//! the perceptron class names (validated to be UPOS tags).
//!
//! The first line of the existing AdaBoost and averaged-perceptron formats
//! can never equal the magic line (it is neither a valid weight/bias line
//! nor a class count), so the format is purely additive: old files keep
//! loading with their loaders, and those loaders reject two-stage files
//! with `InvalidData`.

use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;
use std::str::FromStr;

use rustc_hash::FxHashMap;

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::perceptron::AveragedPerceptron;
use crate::upos::Upos;

/// Magic first line of the two-stage model format (version 1).
const MAGIC: &str = "litsea-two-stage v1";
/// Prefix shared by all (current and future) two-stage magic lines.
const MAGIC_PREFIX: &str = "litsea-two-stage ";
/// Marker line opening the optional parameter section.
const SECTION_PARAMS: &str = "[params]";
/// Marker line opening the embedded stage-1 (AdaBoost) section.
const SECTION_STAGE1: &str = "[stage1]";
/// Marker line opening the lexicon section.
const SECTION_LEXICON: &str = "[lexicon]";
/// Marker line opening the embedded stage-2 (averaged perceptron) section.
const SECTION_STAGE2: &str = "[stage2]";

/// Default classifier-skip dominance threshold, used when a model file has
/// no `[params]` section. The value comes from the #147 prototype sweep,
/// where 0.99 reduced classifier invocations by ~20% at a -0.02pt tagged-F1
/// cost.
pub const DEFAULT_DOMINANCE: f64 = 0.99;

/// The kind of model stored in a litsea model file, detected from its
/// content. Used to dispatch a file to the matching loader (CLI `--pos`
/// accepts both joint and two-stage models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// AdaBoost word-segmentation model (`feature\tweight` lines + bias).
    AdaBoost,
    /// Averaged-perceptron joint POS model (class count header).
    AveragedPerceptron,
    /// Two-stage model (`litsea-two-stage` magic line).
    TwoStage,
}

impl ModelKind {
    /// Detects the model kind from file content by inspecting the first
    /// line: a `litsea-two-stage` magic line means [`ModelKind::TwoStage`],
    /// a bare integer (the class-count header) means
    /// [`ModelKind::AveragedPerceptron`], and anything else is assumed to
    /// be [`ModelKind::AdaBoost`].
    ///
    /// Detection is a dispatch heuristic, not a validation: the matching
    /// loader still fully validates the content and reports malformed files
    /// as `InvalidData`.
    ///
    /// # Arguments
    /// * `content` - The model file content (or at least its first line).
    ///
    /// # Returns
    /// The detected [`ModelKind`].
    #[must_use]
    pub fn detect(content: &str) -> Self {
        let first = content.lines().next().unwrap_or("");
        if first.starts_with(MAGIC_PREFIX) {
            ModelKind::TwoStage
        } else if first.trim().parse::<usize>().is_ok() {
            ModelKind::AveragedPerceptron
        } else {
            ModelKind::AdaBoost
        }
    }
}

/// Lexicon entry type: the UPOS tags observed for one surface, with their
/// training-corpus occurrence counts, sorted most-frequent-first (ties
/// broken by tag name).
type LexiconEntry = Vec<(Upos, u32)>;

/// A two-stage segmentation + POS-tagging model: a stage-1 boundary
/// classifier, a candidate-tag lexicon, and a stage-2 word-level tagger.
///
/// This type owns the model data and its (de)serialization. It follows the
/// same API conventions as [`AdaBoost`] and [`AveragedPerceptron`]:
/// construct with [`new`](Self::new) (or [`from_parts`](Self::from_parts))
/// and fill it with [`load_model`](Self::load_model) /
/// [`load_model_from_path`](Self::load_model_from_path) /
/// [`load_model_from_reader`](Self::load_model_from_reader).
#[derive(Debug)]
pub struct TwoStageLearner {
    /// Stage-1 binary boundary classifier (scalar weights, AdaBoost format).
    stage1: AdaBoost,
    /// Stage-2 global multiclass tagger over word-level features.
    stage2: AveragedPerceptron,
    /// Word surface -> candidate tags with counts, most frequent first.
    lexicon: FxHashMap<String, LexiconEntry>,
    /// Classifier-skip dominance threshold, in `(0.5, 1.0]`.
    dominance: f64,
}

impl Default for TwoStageLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoStageLearner {
    /// Creates an empty learner with the default dominance threshold.
    ///
    /// An empty learner cannot be saved (both embedded sections reject
    /// empty models); fill it with a `load_model*` call or build a
    /// populated one with [`from_parts`](Self::from_parts).
    ///
    /// # Returns
    /// A new empty [`TwoStageLearner`].
    #[must_use]
    pub fn new() -> Self {
        TwoStageLearner {
            stage1: AdaBoost::default(),
            stage2: AveragedPerceptron::new(),
            lexicon: FxHashMap::default(),
            dominance: DEFAULT_DOMINANCE,
        }
    }

    /// Builds a learner from its three parts and a dominance threshold,
    /// validating the combination.
    ///
    /// Lexicon entries are normalized to the canonical order (count
    /// descending, ties by tag name ascending); they do not need to arrive
    /// sorted.
    ///
    /// # Arguments
    /// * `stage1` - The stage-1 boundary classifier.
    /// * `stage2` - The stage-2 tagger; every registered class name must be
    ///   a valid UPOS tag.
    /// * `lexicon` - `(surface, tags)` pairs; surfaces must be non-empty
    ///   and free of tab/newline, tag lists must be non-empty with positive
    ///   counts and no duplicate tag.
    /// * `dominance` - The classifier-skip threshold, in `(0.5, 1.0]`.
    ///
    /// # Returns
    /// The validated learner.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if the dominance threshold is
    /// out of range, the lexicon is empty or violates the rules above, a
    /// duplicate surface is supplied, or a stage-2 class name is not a
    /// UPOS tag.
    pub fn from_parts(
        stage1: AdaBoost,
        stage2: AveragedPerceptron,
        lexicon: impl IntoIterator<Item = (String, LexiconEntry)>,
        dominance: f64,
    ) -> Result<Self> {
        if !(dominance > 0.5 && dominance <= 1.0) {
            return Err(LitseaError::InvalidInput(format!(
                "dominance must be in (0.5, 1.0], got {}",
                dominance
            )));
        }
        for class in stage2.classes() {
            if Upos::from_str(class).is_err() {
                return Err(LitseaError::InvalidInput(format!(
                    "stage-2 class '{}' is not a UPOS tag",
                    class
                )));
            }
        }

        let mut map: FxHashMap<String, LexiconEntry> = FxHashMap::default();
        for (surface, mut entry) in lexicon {
            if surface.is_empty() || surface.contains('\t') || surface.contains('\n') {
                return Err(LitseaError::InvalidInput(format!(
                    "invalid lexicon surface: '{}'",
                    surface.escape_debug()
                )));
            }
            if entry.is_empty() {
                return Err(LitseaError::InvalidInput(format!(
                    "lexicon surface '{}' has no tags",
                    surface
                )));
            }
            for i in 0..entry.len() {
                let (tag, count) = entry[i];
                if count == 0 {
                    return Err(LitseaError::InvalidInput(format!(
                        "lexicon surface '{}' has a zero count for tag {}",
                        surface, tag
                    )));
                }
                if entry[..i].iter().any(|(t, _)| *t == tag) {
                    return Err(LitseaError::InvalidInput(format!(
                        "lexicon surface '{}' lists tag {} twice",
                        surface, tag
                    )));
                }
            }
            sort_lexicon_entry(&mut entry);
            if map.insert(surface.clone(), entry).is_some() {
                return Err(LitseaError::InvalidInput(format!(
                    "duplicate lexicon surface: '{}'",
                    surface
                )));
            }
        }
        if map.is_empty() {
            return Err(LitseaError::InvalidInput("lexicon must not be empty".to_string()));
        }

        Ok(TwoStageLearner {
            stage1,
            stage2,
            lexicon: map,
            dominance,
        })
    }

    /// Decomposes the learner into its parts. Crate-private: used by the
    /// segmenter runtime, which installs stage-1 as its boundary learner
    /// and compiles the rest into packed tagging tables.
    ///
    /// # Returns
    /// `(stage1, stage2, lexicon, dominance)`.
    pub(crate) fn into_parts(
        self,
    ) -> (AdaBoost, AveragedPerceptron, FxHashMap<String, LexiconEntry>, f64) {
        (self.stage1, self.stage2, self.lexicon, self.dominance)
    }

    /// Returns the stage-1 boundary classifier.
    #[must_use]
    pub fn stage1(&self) -> &AdaBoost {
        &self.stage1
    }

    /// Returns the stage-2 word-level tagger.
    #[must_use]
    pub fn stage2(&self) -> &AveragedPerceptron {
        &self.stage2
    }

    /// Returns the classifier-skip dominance threshold, in `(0.5, 1.0]`.
    #[must_use]
    pub fn dominance(&self) -> f64 {
        self.dominance
    }

    /// Returns the number of surfaces in the lexicon.
    #[must_use]
    pub fn lexicon_len(&self) -> usize {
        self.lexicon.len()
    }

    /// Looks up the candidate tags observed for a surface.
    ///
    /// # Arguments
    /// * `surface` - The word surface to look up (exact match, not
    ///   trimmed).
    ///
    /// # Returns
    /// The `(tag, count)` candidates sorted most-frequent-first (ties by
    /// tag name), or `None` if the surface was not seen in training.
    #[must_use]
    pub fn lexicon_entry(&self, surface: &str) -> Option<&[(Upos, u32)]> {
        self.lexicon.get(surface).map(Vec::as_slice)
    }

    /// Saves the model to a file in the `litsea-two-stage v1` format.
    ///
    /// # Arguments
    /// * `path` - The path of the file to write the model to.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if any part is empty (see
    /// [`save_model_to_writer`](Self::save_model_to_writer)), or an I/O
    /// error if the file cannot be created or written.
    pub fn save_model(&self, path: &Path) -> Result<()> {
        let mut file = std::io::BufWriter::new(File::create(path)?);
        self.save_model_to_writer(&mut file)?;
        file.flush()?;
        Ok(())
    }

    /// Writes the model to an arbitrary writer in the `litsea-two-stage v1`
    /// format.
    ///
    /// The output is deterministic: the `[params]` section is always
    /// written, lexicon surfaces are sorted, and the embedded sections use
    /// the deterministic output of the underlying learners. The writer is
    /// not flushed.
    ///
    /// # Arguments
    /// * `writer` - The writer receiving the model text.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if the lexicon is empty or
    /// either embedded learner is empty (their writers reject empty
    /// models), or an I/O error if writing fails.
    pub fn save_model_to_writer<W: Write>(&self, writer: &mut W) -> Result<()> {
        if self.lexicon.is_empty() {
            return Err(LitseaError::InvalidInput("Cannot save an empty model".to_string()));
        }

        writeln!(writer, "{}", MAGIC)?;
        writeln!(writer, "{}", SECTION_PARAMS)?;
        writeln!(writer, "dominance\t{}", self.dominance)?;
        writeln!(writer, "{}", SECTION_STAGE1)?;
        self.stage1.save_model_to_writer(writer)?;
        writeln!(writer, "{}", SECTION_LEXICON)?;
        let mut surfaces: Vec<&String> = self.lexicon.keys().collect();
        surfaces.sort_unstable();
        for surface in surfaces {
            let tags = self.lexicon[surface]
                .iter()
                .map(|(tag, count)| format!("{}:{}", tag, count))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t{}", surface, tags)?;
        }
        writeln!(writer, "{}", SECTION_STAGE2)?;
        self.stage2.save_model_to_writer(writer)?;
        Ok(())
    }

    /// Loads a model from a URI.
    ///
    /// The URI can be a file path, a `file://` path, or an `http(s)://` URL
    /// (the latter requires the `remote_model` feature).
    /// For local files, prefer the synchronous
    /// [`load_model_from_path`](Self::load_model_from_path).
    ///
    /// # Arguments
    /// * `uri` - The URI of the model to load.
    ///
    /// # Errors
    /// Returns an error if the model bytes cannot be fetched from the URI
    /// or the content is malformed (see
    /// [`load_model_from_reader`](Self::load_model_from_reader)).
    pub async fn load_model(&mut self, uri: &str) -> Result<()> {
        let bytes = crate::model_io::read_model_bytes(uri).await?;
        self.load_model_from_reader(bytes.as_slice())
    }

    /// Loads a model from a local file path (synchronous).
    ///
    /// # Arguments
    /// * `path` - The path of the model file to load.
    ///
    /// # Errors
    /// Returns an I/O error if the file cannot be opened, or a parse error
    /// if the content is malformed (see
    /// [`load_model_from_reader`](Self::load_model_from_reader)).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_model_from_path(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path)?;
        self.load_model_from_reader(std::io::BufReader::new(file))
    }

    /// Loads a model from a buffered reader (synchronous).
    ///
    /// The learner is not modified on error.
    ///
    /// # Arguments
    /// * `reader` - The buffered reader providing the model content (the
    ///   format written by [`save_model`](Self::save_model)).
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidData`] if the magic line is missing or
    /// names an unsupported version, sections are missing, duplicated, out
    /// of order, or malformed (including the embedded stage-1/stage-2
    /// content, reported with the section name), the lexicon is empty or
    /// violates the format, or a parameter is unknown or out of range.
    /// I/O errors from the reader are also propagated.
    pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        /// The section currently being collected.
        #[derive(PartialEq, Clone, Copy)]
        enum Section {
            Start,
            Params,
            Stage1,
            Lexicon,
            Stage2,
        }

        let mut lines = reader.lines();
        let first = lines
            .next()
            .ok_or_else(|| LitseaError::InvalidData("Empty model file".to_string()))??;
        if first != MAGIC {
            if first.starts_with(MAGIC_PREFIX) {
                return Err(LitseaError::InvalidData(format!(
                    "unsupported two-stage model version: '{}'",
                    first
                )));
            }
            return Err(LitseaError::InvalidData(format!(
                "missing '{}' magic line (found '{}')",
                MAGIC, first
            )));
        }

        let mut section = Section::Start;
        let mut params_lines: Vec<String> = Vec::new();
        let mut stage1_lines: Vec<String> = Vec::new();
        let mut lexicon_lines: Vec<String> = Vec::new();
        let mut stage2_lines: Vec<String> = Vec::new();
        for line in lines {
            let line = line?;
            match line.as_str() {
                SECTION_PARAMS if section == Section::Start => section = Section::Params,
                SECTION_STAGE1 if matches!(section, Section::Start | Section::Params) => {
                    section = Section::Stage1;
                }
                SECTION_LEXICON if section == Section::Stage1 => section = Section::Lexicon,
                SECTION_STAGE2 if section == Section::Lexicon => section = Section::Stage2,
                SECTION_PARAMS | SECTION_STAGE1 | SECTION_LEXICON | SECTION_STAGE2 => {
                    return Err(LitseaError::InvalidData(format!(
                        "section marker '{}' is duplicated or out of order",
                        line
                    )));
                }
                _ => match section {
                    Section::Start => {
                        return Err(LitseaError::InvalidData(format!(
                            "expected '{}' or '{}' after the magic line, found '{}'",
                            SECTION_PARAMS, SECTION_STAGE1, line
                        )));
                    }
                    Section::Params => params_lines.push(line),
                    Section::Stage1 => stage1_lines.push(line),
                    Section::Lexicon => lexicon_lines.push(line),
                    Section::Stage2 => stage2_lines.push(line),
                },
            }
        }
        if section != Section::Stage2 {
            return Err(LitseaError::InvalidData(
                "missing section: the file must contain [stage1], [lexicon] and [stage2]"
                    .to_string(),
            ));
        }

        let dominance = parse_params(&params_lines)?;
        let mut stage1 = AdaBoost::default();
        stage1
            .load_model_from_reader(stage1_lines.join("\n").as_bytes())
            .map_err(|e| in_section(SECTION_STAGE1, e))?;
        let lexicon = parse_lexicon(&lexicon_lines)?;
        let mut stage2 = AveragedPerceptron::new();
        stage2
            .load_model_from_reader(stage2_lines.join("\n").as_bytes())
            .map_err(|e| in_section(SECTION_STAGE2, e))?;
        for class in stage2.classes() {
            if Upos::from_str(class).is_err() {
                return Err(LitseaError::InvalidData(format!(
                    "{} section: class '{}' is not a UPOS tag",
                    SECTION_STAGE2, class
                )));
            }
        }

        self.stage1 = stage1;
        self.stage2 = stage2;
        self.lexicon = lexicon;
        self.dominance = dominance;
        Ok(())
    }
}

/// Sorts a lexicon entry into the canonical order: count descending, ties
/// by tag name ascending.
fn sort_lexicon_entry(entry: &mut LexiconEntry) {
    entry.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.to_string().cmp(&b.0.to_string())));
}

/// Prefixes an `InvalidData` message with the section it occurred in; other
/// error kinds (I/O) pass through unchanged.
fn in_section(section: &str, e: LitseaError) -> LitseaError {
    match e {
        LitseaError::InvalidData(msg) => {
            LitseaError::InvalidData(format!("{} section: {}", section, msg))
        }
        other => other,
    }
}

/// Parses the `[params]` section lines and returns the dominance threshold
/// (the default when the section carries no `dominance` line).
///
/// # Errors
/// Returns [`LitseaError::InvalidData`] on an unknown key, a duplicate
/// `dominance` line, or a value that is unparsable or outside `(0.5, 1.0]`.
fn parse_params(lines: &[String]) -> Result<f64> {
    let mut dominance: Option<f64> = None;
    for line in lines {
        let Some((key, value)) = line.split_once('\t') else {
            return Err(LitseaError::InvalidData(format!(
                "{} section: invalid line '{}' (expected 'key\\tvalue')",
                SECTION_PARAMS, line
            )));
        };
        match key {
            "dominance" => {
                if dominance.is_some() {
                    return Err(LitseaError::InvalidData(format!(
                        "{} section: duplicate 'dominance'",
                        SECTION_PARAMS
                    )));
                }
                let v: f64 = value.parse().map_err(|e| {
                    LitseaError::InvalidData(format!(
                        "{} section: invalid dominance '{}': {}",
                        SECTION_PARAMS, value, e
                    ))
                })?;
                if !(v > 0.5 && v <= 1.0) {
                    return Err(LitseaError::InvalidData(format!(
                        "{} section: dominance must be in (0.5, 1.0], got {}",
                        SECTION_PARAMS, value
                    )));
                }
                dominance = Some(v);
            }
            _ => {
                return Err(LitseaError::InvalidData(format!(
                    "{} section: unknown parameter '{}'",
                    SECTION_PARAMS, key
                )));
            }
        }
    }
    Ok(dominance.unwrap_or(DEFAULT_DOMINANCE))
}

/// Parses the `[lexicon]` section lines into the in-memory lexicon,
/// normalizing each entry to the canonical order.
///
/// # Errors
/// Returns [`LitseaError::InvalidData`] on a line without a tab, an empty
/// surface, a malformed `TAG:count` element, an unknown tag, a zero count,
/// a duplicate tag within a line, a duplicate surface, or an empty section.
fn parse_lexicon(lines: &[String]) -> Result<FxHashMap<String, LexiconEntry>> {
    let mut lexicon: FxHashMap<String, LexiconEntry> = FxHashMap::default();
    for line in lines {
        let Some((surface, tags_str)) = line.split_once('\t') else {
            return Err(LitseaError::InvalidData(format!(
                "{} section: invalid line '{}' (expected 'surface\\ttags')",
                SECTION_LEXICON, line
            )));
        };
        if surface.is_empty() {
            return Err(LitseaError::InvalidData(format!(
                "{} section: empty surface in line '{}'",
                SECTION_LEXICON, line
            )));
        }
        if tags_str.contains('\t') {
            return Err(LitseaError::InvalidData(format!(
                "{} section: unexpected tab in tag list '{}'",
                SECTION_LEXICON, tags_str
            )));
        }
        let mut entry: LexiconEntry = Vec::new();
        for part in tags_str.split(',') {
            let Some((tag_str, count_str)) = part.split_once(':') else {
                return Err(LitseaError::InvalidData(format!(
                    "{} section: invalid tag element '{}' (expected 'TAG:count')",
                    SECTION_LEXICON, part
                )));
            };
            let tag = Upos::from_str(tag_str).map_err(|e| {
                LitseaError::InvalidData(format!("{} section: {}", SECTION_LEXICON, e))
            })?;
            let count: u32 = count_str.parse().map_err(|e| {
                LitseaError::InvalidData(format!(
                    "{} section: invalid count '{}': {}",
                    SECTION_LEXICON, count_str, e
                ))
            })?;
            if count == 0 {
                return Err(LitseaError::InvalidData(format!(
                    "{} section: zero count for tag {} of surface '{}'",
                    SECTION_LEXICON, tag, surface
                )));
            }
            if entry.iter().any(|(t, _)| *t == tag) {
                return Err(LitseaError::InvalidData(format!(
                    "{} section: surface '{}' lists tag {} twice",
                    SECTION_LEXICON, surface, tag
                )));
            }
            entry.push((tag, count));
        }
        sort_lexicon_entry(&mut entry);
        if lexicon.insert(surface.to_string(), entry).is_some() {
            return Err(LitseaError::InvalidData(format!(
                "{} section: duplicate surface '{}'",
                SECTION_LEXICON, surface
            )));
        }
    }
    if lexicon.is_empty() {
        return Err(LitseaError::InvalidData(format!(
            "{} section: the lexicon must not be empty",
            SECTION_LEXICON
        )));
    }
    Ok(lexicon)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// A minimal valid stage-1 (AdaBoost format) section body.
    const STAGE1: &str = "UW1:a\t0.5\nUW2:b\t-0.25\n-0.125";
    /// A minimal valid stage-2 (averaged perceptron format) section body.
    const STAGE2: &str = "2\nNOUN\nVERB\nL1:x\tNOUN\t0.5\nWS:run\tVERB\t1";

    /// Builds a valid two-stage model file from the given section bodies.
    fn model_text(params: Option<&str>, stage1: &str, lexicon: &str, stage2: &str) -> String {
        let mut s = String::new();
        s.push_str(MAGIC);
        s.push('\n');
        if let Some(p) = params {
            s.push_str("[params]\n");
            if !p.is_empty() {
                s.push_str(p);
                s.push('\n');
            }
        }
        s.push_str("[stage1]\n");
        s.push_str(stage1);
        s.push('\n');
        s.push_str("[lexicon]\n");
        s.push_str(lexicon);
        s.push('\n');
        s.push_str("[stage2]\n");
        s.push_str(stage2);
        s.push('\n');
        s
    }

    /// A valid model file exercising params, a multi-tag lexicon entry, and
    /// both embedded sections.
    fn valid_model() -> String {
        model_text(Some("dominance\t0.9"), STAGE1, "run\tVERB:7,NOUN:2\nは\tADP:10", STAGE2)
    }

    fn load(text: &str) -> Result<TwoStageLearner> {
        let mut learner = TwoStageLearner::new();
        learner.load_model_from_reader(text.as_bytes())?;
        Ok(learner)
    }

    #[test]
    fn test_round_trip_and_determinism() {
        let learner = load(&valid_model()).unwrap();
        assert_eq!(learner.dominance(), 0.9);
        assert_eq!(learner.lexicon_len(), 2);
        assert_eq!(learner.lexicon_entry("run"), Some(&[(Upos::VERB, 7), (Upos::NOUN, 2)][..]));
        assert_eq!(learner.lexicon_entry("は"), Some(&[(Upos::ADP, 10)][..]));
        assert_eq!(learner.lexicon_entry("missing"), None);
        assert_eq!(learner.stage2().classes(), ["NOUN", "VERB"]);

        let mut first = Vec::new();
        learner.save_model_to_writer(&mut first).unwrap();
        let reloaded = load(std::str::from_utf8(&first).unwrap()).unwrap();
        let mut second = Vec::new();
        reloaded.save_model_to_writer(&mut second).unwrap();
        assert_eq!(first, second, "save -> load -> save must be byte-identical");
    }

    #[test]
    fn test_lexicon_entry_order_is_normalized() {
        // Counts arriving unsorted (and tied) are normalized: count
        // descending, ties by tag name ascending.
        let text = model_text(None, STAGE1, "run\tNOUN:2,VERB:7,ADP:7", STAGE2);
        let learner = load(&text).unwrap();
        assert_eq!(
            learner.lexicon_entry("run"),
            Some(&[(Upos::ADP, 7), (Upos::VERB, 7), (Upos::NOUN, 2)][..])
        );
        // The params section is optional; the default applies.
        assert_eq!(learner.dominance(), DEFAULT_DOMINANCE);
    }

    #[test]
    fn test_model_kind_detect() {
        assert_eq!(ModelKind::detect(&valid_model()), ModelKind::TwoStage);
        assert_eq!(ModelKind::detect("litsea-two-stage v2\n"), ModelKind::TwoStage);
        assert_eq!(ModelKind::detect(STAGE2), ModelKind::AveragedPerceptron);
        assert_eq!(ModelKind::detect(STAGE1), ModelKind::AdaBoost);
        assert_eq!(ModelKind::detect(""), ModelKind::AdaBoost);
    }

    #[test]
    fn test_existing_loaders_reject_two_stage_files() {
        let text = valid_model();
        let mut adaboost = AdaBoost::default();
        assert!(matches!(
            adaboost.load_model_from_reader(text.as_bytes()),
            Err(LitseaError::InvalidData(_))
        ));
        let mut perceptron = AveragedPerceptron::new();
        assert!(matches!(
            perceptron.load_model_from_reader(text.as_bytes()),
            Err(LitseaError::InvalidData(_))
        ));
    }

    #[test]
    fn test_load_rejects_bad_magic() {
        let missing = valid_model().replacen(MAGIC, "not-a-litsea-model", 1);
        assert!(
            matches!(load(&missing), Err(LitseaError::InvalidData(msg)) if msg.contains("magic"))
        );

        let future = valid_model().replacen(MAGIC, "litsea-two-stage v2", 1);
        assert!(matches!(
            load(&future),
            Err(LitseaError::InvalidData(msg)) if msg.contains("unsupported")
        ));

        assert!(matches!(load(""), Err(LitseaError::InvalidData(_))));
    }

    #[test]
    fn test_load_rejects_section_structure_errors() {
        // Missing [stage2] (file ends in the lexicon section).
        let truncated = valid_model().split("[stage2]").next().unwrap().to_string();
        assert!(matches!(
            load(&truncated),
            Err(LitseaError::InvalidData(msg)) if msg.contains("missing section")
        ));

        // Duplicate section marker.
        let dup = valid_model().replacen("[lexicon]", "[stage1]\n[lexicon]", 1);
        assert!(matches!(
            load(&dup),
            Err(LitseaError::InvalidData(msg)) if msg.contains("duplicated or out of order")
        ));

        // Out-of-order: [lexicon] before [stage1].
        let out_of_order = model_text(None, STAGE1, "run\tVERB:1", STAGE2)
            .replacen("[stage1]", "[TMP]", 1)
            .replacen("[lexicon]", "[stage1]", 1)
            .replacen("[TMP]", "[lexicon]", 1);
        assert!(matches!(out_of_order, ref s if load(s).is_err()));

        // Content between the magic line and the first section marker.
        let stray = valid_model().replacen("[params]", "stray\n[params]", 1);
        assert!(matches!(
            load(&stray),
            Err(LitseaError::InvalidData(msg)) if msg.contains("after the magic line")
        ));

        // [params] after [stage1] is out of order.
        let late_params = model_text(None, &format!("[params]\n{}", STAGE1), "run\tVERB:1", STAGE2);
        assert!(matches!(
            load(&late_params),
            Err(LitseaError::InvalidData(msg)) if msg.contains("duplicated or out of order")
        ));
    }

    #[test]
    fn test_load_rejects_bad_params() {
        for (params, expect) in [
            ("dominance\t0.5", "must be in"),
            ("dominance\t1.5", "must be in"),
            ("dominance\tabc", "invalid dominance"),
            ("dominance\t0.9\ndominance\t0.9", "duplicate"),
            ("verbosity\thigh", "unknown parameter"),
            ("dominance 0.9", "expected 'key"),
        ] {
            let text = model_text(Some(params), STAGE1, "run\tVERB:1", STAGE2);
            let result = load(&text);
            assert!(
                matches!(result, Err(LitseaError::InvalidData(ref msg)) if msg.contains(expect)),
                "params {:?}: expected error containing {:?}, got {:?}",
                params,
                expect,
                result
            );
        }
    }

    #[test]
    fn test_load_rejects_bad_lexicon() {
        for (lexicon, expect) in [
            ("no-tab-here", "expected 'surface"),
            ("\tVERB:1", "empty surface"),
            ("run\tVERB:1\textra", "unexpected tab"),
            ("run\tVERB", "expected 'TAG:count'"),
            ("run\tFOO:1", "FOO"),
            ("run\tVERB:0", "zero count"),
            ("run\tVERB:x", "invalid count"),
            ("run\tVERB:1,VERB:2", "twice"),
            ("run\tVERB:1,", "expected 'TAG:count'"),
            ("run\tVERB:1\nrun\tNOUN:1", "duplicate surface"),
        ] {
            let text = model_text(None, STAGE1, lexicon, STAGE2);
            let result = load(&text);
            assert!(
                matches!(result, Err(LitseaError::InvalidData(ref msg)) if msg.contains(expect)),
                "lexicon {:?}: expected error containing {:?}, got {:?}",
                lexicon,
                expect,
                result
            );
        }

        // An empty lexicon section is rejected.
        let text = valid_model().replace("run\tVERB:7,NOUN:2\nは\tADP:10\n", "");
        assert!(matches!(
            load(&text),
            Err(LitseaError::InvalidData(msg)) if msg.contains("must not be empty")
        ));
    }

    #[test]
    fn test_load_rejects_embedded_section_errors_with_context() {
        // Broken stage-1 content (no bias line) is reported with its section.
        let text = model_text(None, "UW1:a\t0.5", "run\tVERB:1", STAGE2);
        assert!(matches!(
            load(&text),
            Err(LitseaError::InvalidData(msg)) if msg.contains("[stage1] section:")
        ));

        // Broken stage-2 content is reported with its section.
        let text = model_text(None, STAGE1, "run\tVERB:1", "2\nNOUN\nVERB\nbroken-line");
        assert!(matches!(
            load(&text),
            Err(LitseaError::InvalidData(msg)) if msg.contains("[stage2] section:")
        ));

        // A stage-2 class that is not a UPOS tag is rejected.
        let text = model_text(None, STAGE1, "run\tVERB:1", "2\nFOO\nNOUN\nWS:x\tFOO\t1");
        assert!(matches!(
            load(&text),
            Err(LitseaError::InvalidData(msg)) if msg.contains("not a UPOS tag")
        ));
    }

    #[test]
    fn test_learner_not_modified_on_error() {
        let mut learner = load(&valid_model()).unwrap();
        let bad = model_text(None, STAGE1, "run\tFOO:1", STAGE2);
        assert!(learner.load_model_from_reader(bad.as_bytes()).is_err());
        // The previously loaded state is intact.
        assert_eq!(learner.dominance(), 0.9);
        assert_eq!(learner.lexicon_len(), 2);
        assert_eq!(learner.lexicon_entry("は"), Some(&[(Upos::ADP, 10)][..]));
    }

    #[test]
    fn test_from_parts_validation() {
        fn parts() -> (AdaBoost, AveragedPerceptron) {
            let mut stage1 = AdaBoost::default();
            stage1.load_model_from_reader(STAGE1.as_bytes()).unwrap();
            let mut stage2 = AveragedPerceptron::new();
            stage2.load_model_from_reader(STAGE2.as_bytes()).unwrap();
            (stage1, stage2)
        }
        let lex = |surface: &str, entry: LexiconEntry| vec![(surface.to_string(), entry)];

        // A valid combination normalizes the entry order.
        let (s1, s2) = parts();
        let learner = TwoStageLearner::from_parts(
            s1,
            s2,
            lex("run", vec![(Upos::NOUN, 2), (Upos::VERB, 7)]),
            0.99,
        )
        .unwrap();
        assert_eq!(learner.lexicon_entry("run"), Some(&[(Upos::VERB, 7), (Upos::NOUN, 2)][..]));

        // Dominance out of range.
        let (s1, s2) = parts();
        assert!(matches!(
            TwoStageLearner::from_parts(s1, s2, lex("run", vec![(Upos::VERB, 1)]), 0.5),
            Err(LitseaError::InvalidInput(msg)) if msg.contains("dominance")
        ));

        // Empty lexicon.
        let (s1, s2) = parts();
        assert!(matches!(
            TwoStageLearner::from_parts(s1, s2, Vec::new(), 0.99),
            Err(LitseaError::InvalidInput(msg)) if msg.contains("empty")
        ));

        // Tab in surface, empty surface, empty tags, zero count, duplicate tag.
        for (surface, entry, expect) in [
            ("a\tb", vec![(Upos::VERB, 1)], "invalid lexicon surface"),
            ("", vec![(Upos::VERB, 1)], "invalid lexicon surface"),
            ("run", vec![], "no tags"),
            ("run", vec![(Upos::VERB, 0)], "zero count"),
            ("run", vec![(Upos::VERB, 1), (Upos::VERB, 2)], "twice"),
        ] {
            let (s1, s2) = parts();
            let result = TwoStageLearner::from_parts(s1, s2, lex(surface, entry), 0.99);
            assert!(
                matches!(result, Err(LitseaError::InvalidInput(ref msg)) if msg.contains(expect)),
                "surface {:?}: expected error containing {:?}, got {:?}",
                surface,
                expect,
                result
            );
        }

        // Duplicate surface across items.
        let (s1, s2) = parts();
        let dup = vec![
            ("run".to_string(), vec![(Upos::VERB, 1)]),
            ("run".to_string(), vec![(Upos::NOUN, 1)]),
        ];
        assert!(matches!(
            TwoStageLearner::from_parts(s1, s2, dup, 0.99),
            Err(LitseaError::InvalidInput(msg)) if msg.contains("duplicate lexicon surface")
        ));

        // A stage-2 class that is not a UPOS tag.
        let (s1, _) = parts();
        let mut bad_stage2 = AveragedPerceptron::new();
        bad_stage2
            .load_model_from_reader("2\nFOO\nNOUN\nWS:x\tFOO\t1".as_bytes())
            .unwrap();
        assert!(matches!(
            TwoStageLearner::from_parts(s1, bad_stage2, lex("run", vec![(Upos::VERB, 1)]), 0.99),
            Err(LitseaError::InvalidInput(msg)) if msg.contains("not a UPOS tag")
        ));
    }

    #[test]
    fn test_save_rejects_empty_model() {
        let learner = TwoStageLearner::new();
        let mut buf = Vec::new();
        assert!(matches!(
            learner.save_model_to_writer(&mut buf),
            Err(LitseaError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_save_model_writer_matches_save_model() {
        // The path-based and writer-based savers of all three model types
        // must produce identical bytes.
        let learner = load(&valid_model()).unwrap();
        let dir = tempfile::tempdir().unwrap();

        let path = dir.path().join("two_stage.model");
        learner.save_model(&path).unwrap();
        let mut buf = Vec::new();
        learner.save_model_to_writer(&mut buf).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), buf);

        let path = dir.path().join("stage1.model");
        learner.stage1().save_model(&path).unwrap();
        let mut buf = Vec::new();
        learner.stage1().save_model_to_writer(&mut buf).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), buf);

        let path = dir.path().join("stage2.model");
        learner.stage2().save_model(&path).unwrap();
        let mut buf = Vec::new();
        learner.stage2().save_model_to_writer(&mut buf).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), buf);
    }

    #[tokio::test]
    async fn test_load_from_path_and_uri() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.two-stage");
        std::fs::write(&path, valid_model()).unwrap();

        let mut learner = TwoStageLearner::new();
        learner.load_model_from_path(&path).unwrap();
        assert_eq!(learner.lexicon_len(), 2);

        let mut learner = TwoStageLearner::new();
        learner.load_model(&format!("file://{}", path.display())).await.unwrap();
        assert_eq!(learner.lexicon_len(), 2);
    }
}
