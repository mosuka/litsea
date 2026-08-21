//! Feature extraction and model training for the bindings.
//!
//! `litsea`'s extractor and trainers are path-in / path-out and take a
//! `running: &AtomicBool`. This module keeps that shape but replaces the raw
//! flag with a [`CancelToken`] and folds the "which variant do I call"
//! decision (plain text vs TSV corpus, tag-free or not) into arguments, so
//! each binding exposes one method per operation instead of six.
//!
//! Not available on wasm32: every entry point reads and writes files.

use std::path::Path;
use std::str::FromStr;

use litsea::{
    BinaryMetrics, Extractor, Language, MulticlassMetrics, PerceptronTrainer, Trainer,
    TwoStageFeatureSet, TwoStageMetrics, TwoStageTrainer,
};

use crate::cancel::CancelToken;
use crate::error::{CoreError, CoreResult};

/// How a training corpus is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorpusFormat {
    /// Words separated by spaces, one sentence per line.
    #[default]
    PlainText,
    /// Tab-separated, space-preserving format (see the training guide).
    Tsv,
}

impl CorpusFormat {
    /// Selects the format from a boolean "is TSV" flag.
    ///
    /// Bindings usually expose a `tsv=True` keyword rather than an enum.
    ///
    /// # Arguments
    /// * `tsv` - `true` for [`CorpusFormat::Tsv`].
    ///
    /// # Returns
    /// The matching [`CorpusFormat`].
    #[must_use]
    pub fn from_tsv_flag(tsv: bool) -> Self {
        if tsv { CorpusFormat::Tsv } else { CorpusFormat::PlainText }
    }
}

/// Parses a two-stage feature-set name.
///
/// # Arguments
/// * `name` - `"full"`, `"balanced"`, or `"fast"` (case-insensitive).
///
/// # Returns
/// The parsed [`TwoStageFeatureSet`].
///
/// # Errors
/// Returns an [`crate::ErrorKind::InvalidArgument`] error naming the valid
/// values if `name` is not one of them.
pub fn parse_feature_set(name: &str) -> CoreResult<TwoStageFeatureSet> {
    TwoStageFeatureSet::from_str(name).map_err(|e| CoreError::invalid_argument(e.to_string()))
}

/// Extracts training features from a corpus.
#[derive(Debug)]
pub struct CoreExtractor {
    /// The wrapped extractor.
    inner: Extractor,
}

impl CoreExtractor {
    /// Creates an extractor for a language.
    ///
    /// # Arguments
    /// * `language` - The corpus language.
    ///
    /// # Returns
    /// The new [`CoreExtractor`].
    #[must_use]
    pub fn new(language: Language) -> Self {
        Self {
            inner: Extractor::new(language),
        }
    }

    /// Extracts boundary-classification features.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the training corpus.
    /// * `features_path` - Path of the features file to write.
    /// * `format` - The corpus layout.
    /// * `tag_free` - Omit tag-dependent feature templates, producing the
    ///   smaller, faster models the pointwise fast path can use.
    ///
    /// # Returns
    /// `()` once the features file has been written.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus cannot be read or the features
    /// file cannot be written.
    pub fn extract(
        &self,
        corpus_path: &Path,
        features_path: &Path,
        format: CorpusFormat,
        tag_free: bool,
    ) -> CoreResult<()> {
        let result = match (format, tag_free) {
            (CorpusFormat::PlainText, false) => self.inner.extract(corpus_path, features_path),
            (CorpusFormat::PlainText, true) => {
                self.inner.extract_tag_free(corpus_path, features_path)
            }
            (CorpusFormat::Tsv, false) => self.inner.extract_tsv(corpus_path, features_path),
            (CorpusFormat::Tsv, true) => {
                self.inner.extract_tsv_tag_free(corpus_path, features_path)
            }
        };
        result.map_err(CoreError::from)
    }

    /// Extracts two-stage (segmentation + POS) features.
    ///
    /// Writes `{output_prefix}.stage1`, `.stage2`, and `.lexicon`.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the POS-tagged training corpus.
    /// * `output_prefix` - Prefix for the three output files.
    /// * `feature_set` - Which feature templates stage 2 uses.
    /// * `format` - The corpus layout.
    ///
    /// # Returns
    /// `()` once all three files have been written.
    ///
    /// # Errors
    /// Returns an I/O error if the corpus cannot be read or the outputs
    /// cannot be written.
    pub fn extract_two_stage(
        &self,
        corpus_path: &Path,
        output_prefix: &Path,
        feature_set: TwoStageFeatureSet,
        format: CorpusFormat,
    ) -> CoreResult<()> {
        let result = match format {
            CorpusFormat::PlainText => {
                self.inner.extract_two_stage(corpus_path, output_prefix, feature_set)
            }
            CorpusFormat::Tsv => {
                self.inner.extract_two_stage_tsv(corpus_path, output_prefix, feature_set)
            }
        };
        result.map_err(CoreError::from)
    }
}

/// Trains a segmentation model (AdaBoost format).
#[derive(Debug)]
pub struct CoreTrainer {
    /// The wrapped trainer.
    inner: Trainer,
}

impl CoreTrainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `threshold` - Early-stopping threshold for weak classifiers.
    /// * `num_iterations` - Maximum number of boosting iterations.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new [`CoreTrainer`].
    ///
    /// # Errors
    /// Returns an error if the features file cannot be read or parsed.
    pub fn new(threshold: f64, num_iterations: usize, features_path: &Path) -> CoreResult<Self> {
        Ok(Self {
            inner: Trainer::new(threshold, num_iterations, features_path)?,
        })
    }

    /// Loads an existing model to continue training from it.
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `()` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Returns an error if the model cannot be fetched or parsed.
    pub async fn load_model(&mut self, model_uri: &str) -> CoreResult<()> {
        self.inner.load_model(model_uri).await.map_err(CoreError::from)
    }

    /// Blocking variant of [`CoreTrainer::load_model`].
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `()` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Returns an error if called from inside an async runtime, or if the
    /// model cannot be fetched or parsed.
    pub fn load_model_blocking(&mut self, model_uri: &str) -> CoreResult<()> {
        crate::runtime::block_on(self.inner.load_model(model_uri))?.map_err(CoreError::from)
    }

    /// Trains the model and writes it to `model_path`.
    ///
    /// Cancelling `cancel` stops training early; the partially trained model
    /// is still written and its metrics returned.
    ///
    /// # Arguments
    /// * `cancel` - Cancellation token checked once per boosting iteration.
    /// * `model_path` - Path of the model file to write.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Returns an I/O error if the model cannot be written.
    pub fn train(&mut self, cancel: &CancelToken, model_path: &Path) -> CoreResult<BinaryMetrics> {
        self.inner.train(cancel.running_flag(), model_path).map_err(CoreError::from)
    }
}

/// Trains a label-agnostic Averaged Perceptron model.
#[derive(Debug)]
pub struct CorePerceptronTrainer {
    /// The wrapped trainer.
    inner: PerceptronTrainer,
}

impl CorePerceptronTrainer {
    /// Loads a features file and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `features_path` - Path to the features file.
    ///
    /// # Returns
    /// The new [`CorePerceptronTrainer`].
    ///
    /// # Errors
    /// Returns an error if the features file cannot be read or parsed.
    pub fn new(num_epochs: usize, features_path: &Path) -> CoreResult<Self> {
        Ok(Self {
            inner: PerceptronTrainer::new(num_epochs, features_path)?,
        })
    }

    /// Loads an existing model to continue training from it.
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `()` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Returns an error if the model cannot be fetched or parsed.
    pub async fn load_model(&mut self, model_uri: &str) -> CoreResult<()> {
        self.inner.load_model(model_uri).await.map_err(CoreError::from)
    }

    /// Blocking variant of [`CorePerceptronTrainer::load_model`].
    ///
    /// # Arguments
    /// * `model_uri` - Path, `file://` path, or `http(s)://` URL.
    ///
    /// # Returns
    /// `()` once the model has been merged into the learner.
    ///
    /// # Errors
    /// Returns an error if called from inside an async runtime, or if the
    /// model cannot be fetched or parsed.
    pub fn load_model_blocking(&mut self, model_uri: &str) -> CoreResult<()> {
        crate::runtime::block_on(self.inner.load_model(model_uri))?.map_err(CoreError::from)
    }

    /// Trains the model and writes it to `model_path`.
    ///
    /// # Arguments
    /// * `cancel` - Cancellation token checked per epoch and per instance.
    /// * `model_path` - Path of the model file to write.
    ///
    /// # Returns
    /// The training metrics.
    ///
    /// # Errors
    /// Returns an I/O error if the model cannot be written.
    pub fn train(
        &mut self,
        cancel: &CancelToken,
        model_path: &Path,
    ) -> CoreResult<MulticlassMetrics> {
        self.inner.train(cancel.running_flag(), model_path).map_err(CoreError::from)
    }
}

/// Trains a two-stage segmentation + POS model.
#[derive(Debug)]
pub struct CoreTwoStageTrainer {
    /// The wrapped trainer, taken by [`CoreTwoStageTrainer::train`].
    ///
    /// `TwoStageTrainer::train` consumes `self` because stage 1 is collapsed
    /// into an AdaBoost model and cannot be retrained in place, so the
    /// handle bindings hold must be able to give it up exactly once.
    inner: Option<TwoStageTrainer>,
}

impl CoreTwoStageTrainer {
    /// Loads a two-stage features prefix and prepares training.
    ///
    /// # Arguments
    /// * `num_epochs` - Number of passes over the training data.
    /// * `dominance` - Lexicon dominance threshold, in `(0.5, 1.0]`.
    /// * `features_prefix` - Prefix of the `.stage1` / `.stage2` /
    ///   `.lexicon` files written by
    ///   [`CoreExtractor::extract_two_stage`].
    ///
    /// # Returns
    /// The new [`CoreTwoStageTrainer`].
    ///
    /// # Errors
    /// Returns an error if `dominance` is out of range or the feature files
    /// cannot be read or parsed.
    pub fn new(num_epochs: usize, dominance: f64, features_prefix: &Path) -> CoreResult<Self> {
        Ok(Self {
            inner: Some(TwoStageTrainer::new(num_epochs, dominance, features_prefix)?),
        })
    }

    /// Returns whether this trainer can still be used.
    ///
    /// # Returns
    /// `false` once [`CoreTwoStageTrainer::train`] has been called.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.inner.is_some()
    }

    /// Trains both stages and writes the model to `model_path`.
    ///
    /// Consumes the trainer: a second call returns an error rather than
    /// silently retraining nothing.
    ///
    /// # Arguments
    /// * `cancel` - Cancellation token, passed to both stages in turn.
    /// * `model_path` - Path of the model file to write.
    ///
    /// # Returns
    /// The metrics of both stages.
    ///
    /// # Errors
    /// Returns an [`crate::ErrorKind::InvalidArgument`] error if the trainer
    /// has already been used, or an I/O error if the model cannot be
    /// written.
    pub fn train(
        &mut self,
        cancel: &CancelToken,
        model_path: &Path,
    ) -> CoreResult<TwoStageMetrics> {
        let trainer = self.inner.take().ok_or_else(|| {
            CoreError::invalid_argument(
                "this two-stage trainer has already been used; create a new one to train again",
            )
        })?;

        trainer.train(cancel.running_flag(), model_path).map_err(CoreError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;
    use crate::segmenter::CoreSegmenter;

    /// A tiny space-separated corpus, repeated so training has something to
    /// learn from.
    fn write_corpus(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("corpus.txt");
        let sentences = [
            "すもも も もも も もも の うち",
            "隣 の 客 は よく 柿 食う 客 だ",
            "東京 都 から 神奈川 県 へ 引っ越し た",
        ];
        let mut content = String::new();
        for _ in 0..20 {
            for sentence in sentences {
                content.push_str(sentence);
                content.push('\n');
            }
        }
        fs::write(&path, content).unwrap();
        path
    }

    /// The same corpus with UPOS tags, for two-stage training.
    fn write_pos_corpus(dir: &TempDir) -> std::path::PathBuf {
        let path = dir.path().join("corpus_pos.txt");
        let sentences = [
            "すもも/NOUN も/ADP もも/NOUN も/ADP もも/NOUN の/ADP うち/NOUN",
            "隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX",
            "東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX",
        ];
        let mut content = String::new();
        for _ in 0..20 {
            for sentence in sentences {
                content.push_str(sentence);
                content.push('\n');
            }
        }
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_extract_then_train_produces_a_loadable_model() {
        let dir = TempDir::new().unwrap();
        let corpus = write_corpus(&dir);
        let features = dir.path().join("features.txt");
        let model = dir.path().join("trained.model");

        CoreExtractor::new(Language::Japanese)
            .extract(&corpus, &features, CorpusFormat::PlainText, false)
            .unwrap();
        assert!(features.exists());

        let metrics = CoreTrainer::new(0.01, 20, &features)
            .unwrap()
            .train(&CancelToken::new(), &model)
            .unwrap();
        assert!(metrics.num_instances > 0);

        // The written model must be loadable, and recognized as
        // segmentation-only.
        let segmenter = CoreSegmenter::from_path(Language::Japanese, &model).unwrap();
        assert!(!segmenter.has_pos());
        assert!(!segmenter.segment("すもももももももものうち").is_empty());
    }

    #[test]
    fn test_tag_free_extraction_writes_fewer_features() {
        let dir = TempDir::new().unwrap();
        let corpus = write_corpus(&dir);
        let extractor = CoreExtractor::new(Language::Japanese);

        let full = dir.path().join("full.txt");
        let tag_free = dir.path().join("tag_free.txt");
        extractor.extract(&corpus, &full, CorpusFormat::PlainText, false).unwrap();
        extractor.extract(&corpus, &tag_free, CorpusFormat::PlainText, true).unwrap();

        assert!(
            fs::metadata(&tag_free).unwrap().len() < fs::metadata(&full).unwrap().len(),
            "tag-free extraction must drop the tag-dependent templates"
        );
    }

    #[test]
    fn test_cancelled_training_still_writes_a_model() {
        let dir = TempDir::new().unwrap();
        let corpus = write_corpus(&dir);
        let features = dir.path().join("features.txt");
        let model = dir.path().join("cancelled.model");

        CoreExtractor::new(Language::Japanese)
            .extract(&corpus, &features, CorpusFormat::PlainText, false)
            .unwrap();

        let cancel = CancelToken::new();
        cancel.cancel();

        // Cancellation is cooperative, not an error: training returns
        // metrics for the partially trained model and still saves it.
        let metrics =
            CoreTrainer::new(0.01, 1000, &features).unwrap().train(&cancel, &model).unwrap();
        assert!(metrics.num_instances > 0);
        assert!(model.exists());
    }

    #[test]
    fn test_two_stage_training_round_trip() {
        let dir = TempDir::new().unwrap();
        let corpus = write_pos_corpus(&dir);
        let prefix = dir.path().join("features");
        let model = dir.path().join("two_stage.model");

        CoreExtractor::new(Language::Japanese)
            .extract_two_stage(
                &corpus,
                &prefix,
                parse_feature_set("fast").unwrap(),
                CorpusFormat::PlainText,
            )
            .unwrap();
        for suffix in ["stage1", "stage2", "lexicon"] {
            assert!(
                dir.path().join(format!("features.{suffix}")).exists(),
                "missing features.{suffix}"
            );
        }

        let mut trainer = CoreTwoStageTrainer::new(3, 0.99, &prefix).unwrap();
        assert!(trainer.is_available());
        let metrics = trainer.train(&CancelToken::new(), &model).unwrap();
        assert!(metrics.stage1.num_instances > 0);
        assert!(metrics.stage2.num_instances > 0);

        let segmenter = CoreSegmenter::from_path(Language::Japanese, &model).unwrap();
        assert!(segmenter.has_pos(), "a two-stage model must be detected as POS-capable");
        let tokens = segmenter.segment_with_pos("すもももももももものうち").unwrap();
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|t| t.pos.is_some()));
    }

    #[test]
    fn test_two_stage_trainer_cannot_be_reused() {
        let dir = TempDir::new().unwrap();
        let corpus = write_pos_corpus(&dir);
        let prefix = dir.path().join("features");
        let model = dir.path().join("two_stage.model");

        CoreExtractor::new(Language::Japanese)
            .extract_two_stage(
                &corpus,
                &prefix,
                TwoStageFeatureSet::default(),
                CorpusFormat::PlainText,
            )
            .unwrap();

        let mut trainer = CoreTwoStageTrainer::new(1, 0.99, &prefix).unwrap();
        trainer.train(&CancelToken::new(), &model).unwrap();
        assert!(!trainer.is_available());

        let error = trainer.train(&CancelToken::new(), &model).unwrap_err();
        assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);
        assert!(
            error.message().contains("already been used"),
            "unexpected message: {}",
            error.message()
        );
    }

    #[test]
    fn test_perceptron_trainer_trains_from_two_stage_features() {
        let dir = TempDir::new().unwrap();
        let corpus = write_pos_corpus(&dir);
        let prefix = dir.path().join("features");
        let model = dir.path().join("perceptron.model");

        CoreExtractor::new(Language::Japanese)
            .extract_two_stage(
                &corpus,
                &prefix,
                TwoStageFeatureSet::default(),
                CorpusFormat::PlainText,
            )
            .unwrap();

        let stage2 = dir.path().join("features.stage2");
        let metrics = CorePerceptronTrainer::new(2, &stage2)
            .unwrap()
            .train(&CancelToken::new(), &model)
            .unwrap();

        assert!(metrics.num_instances > 0);
        assert!(model.exists());
    }

    #[test]
    fn test_parse_feature_set() {
        assert_eq!(parse_feature_set("FAST").unwrap(), TwoStageFeatureSet::Fast);
        assert_eq!(parse_feature_set("full").unwrap(), TwoStageFeatureSet::Full);
        let error = parse_feature_set("turbo").unwrap_err();
        assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);
        assert!(error.message().contains("turbo"));
    }

    #[test]
    fn test_corpus_format_from_flag() {
        assert_eq!(CorpusFormat::from_tsv_flag(true), CorpusFormat::Tsv);
        assert_eq!(CorpusFormat::from_tsv_flag(false), CorpusFormat::PlainText);
        assert_eq!(CorpusFormat::default(), CorpusFormat::PlainText);
    }
}
