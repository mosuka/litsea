//! Command-line interface for litsea.
//!
//! Provides four subcommands: `extract` (turn a corpus into training
//! features, or, with `--two-stage`, into the three feature files consumed
//! by two-stage training), `train` (train an AdaBoost segmentation model,
//! or, with `--pos`, an Averaged Perceptron POS model, or, with
//! `--two-stage`, a two-stage boundary+lexicon POS model), `segment`
//! (segment sentences from standard input with a trained model), and
//! `evaluate` (measure held-out quality against a gold corpus).

use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Args, Parser, Subcommand};

use litsea::version;
use litsea::{
    AdaBoost, AnyPosModel, Extractor, Language, PosTrainer, Segmenter, Trainer, TwoStageFeatureSet,
    TwoStageTrainer, evaluation,
};

/// Arguments for the extract command.
#[derive(Debug, Args)]
#[command(about = "Extract features from a corpus")]
struct ExtractArgs {
    /// Language of the corpus (japanese, chinese, or korean)
    #[arg(short, long, default_value = "japanese", value_parser = Language::from_str)]
    language: Language,

    /// Extract features from a POS-tagged corpus (format: "word/POS word/POS ...")
    #[arg(long)]
    pos: bool,

    /// Corpus format: "space" (space-separated words) or "tsv" (tab-separated
    /// tokens; a token may be a literal space, preserving original spacing)
    #[arg(long, default_value = "space", value_parser = ["space", "tsv"])]
    format: String,

    /// Extract two-stage training features (issue #147) from a POS-tagged
    /// corpus instead: writes {features_file}.stage1 (boundary features),
    /// .stage2 (word-level features), and .lexicon. Cannot be combined
    /// with --pos or --format tsv.
    #[arg(long)]
    two_stage: bool,

    /// Stage-2 word-feature set for --two-stage: "full" (best quality),
    /// "balanced", or "fast" (best throughput; default)
    #[arg(long, default_value = "fast", value_parser = TwoStageFeatureSet::from_str)]
    stage2_features: TwoStageFeatureSet,

    /// Path to the input corpus file (one pre-segmented sentence per line)
    corpus_file: PathBuf,
    /// Path to the output features file (with --two-stage, the prefix for
    /// the three output files)
    features_file: PathBuf,
}

/// Arguments for the train command.
#[derive(Debug, Args)]
#[command(about = "Train a segmenter")]
struct TrainArgs {
    /// Early-stopping threshold for AdaBoost training. Ignored with `--pos`
    /// or `--two-stage`
    #[arg(short, long, default_value = "0.01")]
    threshold: f64,

    /// Maximum number of AdaBoost boosting iterations. Ignored with `--pos`
    /// or `--two-stage`
    #[arg(short = 'i', long, default_value = "100")]
    num_iterations: usize,

    /// URI of an existing model to load before training (incremental training)
    #[arg(short = 'm', long)]
    load_model_uri: Option<String>,

    /// Train a POS tagging model with the Averaged Perceptron
    #[arg(long)]
    pos: bool,

    /// Number of training epochs for the POS model (applies to both
    /// `--pos` and `--two-stage` training; for `--two-stage`, both stage 1
    /// and stage 2 train for this many epochs)
    #[arg(long, default_value = "10")]
    num_epochs: usize,

    /// Train a two-stage model (issue #147) instead: reads
    /// {features_file}.stage1/.stage2/.lexicon (from extract --two-stage)
    /// and writes a litsea-two-stage model. Cannot be combined with --pos
    /// or -m/--load-model-uri (incremental training is not supported)
    #[arg(long)]
    two_stage: bool,

    /// Classifier-skip dominance threshold for --two-stage: a known word
    /// whose most frequent tag covers at least this fraction of its
    /// training occurrences is tagged without invoking the classifier.
    /// Must be in (0.5, 1.0]
    #[arg(long, default_value = "0.99")]
    dominance: f64,

    /// Path to the features file produced by the extract command (with
    /// --two-stage, the prefix passed to extract --two-stage)
    features_file: PathBuf,
    /// Path to write the trained model to
    model_file: PathBuf,
}

/// Arguments for the segment command.
#[derive(Debug, Args)]
#[command(about = "Segment a sentence")]
struct SegmentArgs {
    /// Language of the input text (japanese, chinese, or korean)
    #[arg(short, long, default_value = "japanese", value_parser = Language::from_str)]
    language: Language,

    /// Segment with POS tagging (auto-detects joint or two-stage model)
    #[arg(long)]
    pos: bool,

    /// Model URI: a plain path, file:// path, or http(s):// URL
    model_uri: String,
}

/// Arguments for the evaluate command.
#[derive(Debug, Args)]
#[command(about = "Evaluate a model against a held-out gold corpus")]
struct EvaluateArgs {
    /// Language of the model and gold corpus (japanese, chinese, or korean)
    #[arg(short, long, default_value = "japanese", value_parser = Language::from_str)]
    language: Language,

    /// Evaluate segmentation + POS tagging (gold format: "word/POS word/POS
    /// ..."); the model file is auto-detected as either a joint or a
    /// two-stage model and evaluated the same way
    #[arg(long)]
    pos: bool,

    /// Gold corpus format: "space" (space-separated tokens) or "tsv"
    /// (tab-separated tokens; a token may be a literal space). Ignored with --pos
    #[arg(long, default_value = "space", value_parser = ["space", "tsv"])]
    format: String,

    /// URI of the model to evaluate (path, file://, or http(s):// with remote_model)
    model_uri: String,
    /// Path to the gold corpus file (one sentence per line)
    gold_file: PathBuf,
}

/// Subcommands for litsea CLI.
#[derive(Debug, Subcommand)]
enum Commands {
    /// Extract features from a corpus
    Extract(ExtractArgs),
    /// Train a segmenter
    Train(TrainArgs),
    /// Segment a sentence
    Segment(SegmentArgs),
    /// Evaluate a model against a held-out gold corpus
    Evaluate(EvaluateArgs),
}

/// Arguments for the litsea command.
#[derive(Debug, Parser)]
#[command(
    name = "litsea",
    author,
    about = "A morphological analysis command line interface",
    version = version(),
    propagate_version = true,
)]
struct CommandArgs {
    #[command(subcommand)]
    command: Commands,
}

/// Extract features from a corpus file and write them to a specified output file.
/// This function reads pre-segmented sentences from the corpus file (the
/// word boundaries come from the corpus itself) and writes the extracted
/// features to the output file. With `--two-stage` the corpus is
/// POS-tagged and three files (boundary, word-level, and lexicon) are
/// written via `extract_two_stage` (issue #147); with `--pos` (and no
/// `--two-stage`) it is POS-tagged and features are extracted via
/// `extract_with_pos`; otherwise each line is space-separated words, or
/// tab-separated tokens with `--format tsv` (a token may be a literal
/// space, preserving the original spacing; not supported with `--pos` or
/// `--two-stage`).
///
/// # Arguments
/// * `args` - The arguments for the extract command [`ExtractArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
fn extract(args: ExtractArgs) -> Result<(), Box<dyn Error>> {
    let extractor = Extractor::new(args.language);

    if args.two_stage {
        if args.pos {
            return Err("--two-stage cannot be combined with --pos".into());
        }
        if args.format == "tsv" {
            return Err("--two-stage cannot be combined with --format tsv".into());
        }
        extractor.extract_two_stage(
            args.corpus_file.as_path(),
            args.features_file.as_path(),
            args.stage2_features,
        )?;
    } else if args.pos {
        // The POS pipeline has no TSV variant (issue #152 covers the
        // word-segmentation path only).
        if args.format == "tsv" {
            return Err("--format tsv is not supported together with --pos".into());
        }
        extractor.extract_with_pos(args.corpus_file.as_path(), args.features_file.as_path())?;
    } else if args.format == "tsv" {
        extractor.extract_tsv(args.corpus_file.as_path(), args.features_file.as_path())?;
    } else {
        extractor.extract(args.corpus_file.as_path(), args.features_file.as_path())?;
    }

    eprintln!("Feature extraction completed successfully.");
    Ok(())
}

/// Train a segmenter using the provided arguments.
/// This function initializes a Trainer with the specified parameters,
/// loads a model if specified, and trains the model using the features file.
///
/// # Arguments
/// * `args` - The arguments for the train command [`TrainArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
async fn train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        if r.load(Ordering::SeqCst) {
            r.store(false, Ordering::SeqCst);
        } else {
            std::process::exit(0);
        }
    })?;

    if args.two_stage {
        if args.pos {
            return Err("--two-stage cannot be combined with --pos".into());
        }
        if args.load_model_uri.is_some() {
            return Err(
                "--two-stage does not support -m/--load-model-uri (incremental training)".into()
            );
        }
        // Train the two-stage model (issue #147): a binary boundary
        // classifier plus a word-level tagger, assembled with the lexicon.
        let trainer =
            TwoStageTrainer::new(args.num_epochs, args.dominance, args.features_file.as_path())?;
        let metrics = trainer.train(&running, args.model_file.as_path())?;

        eprintln!("Result Metrics (Two-Stage):");
        eprintln!(
            "  Stage 1 (boundary) Accuracy: {:.2}% ( {} )",
            metrics.stage1.accuracy, metrics.stage1.num_instances
        );
        eprintln!("  Stage 1 Macro Precision: {:.2}%", metrics.stage1.macro_precision);
        eprintln!("  Stage 1 Macro Recall: {:.2}%", metrics.stage1.macro_recall);
        eprintln!(
            "  Stage 2 (tagging) Accuracy: {:.2}% ( {} )",
            metrics.stage2.accuracy, metrics.stage2.num_instances
        );
        eprintln!("  Stage 2 Macro Precision: {:.2}%", metrics.stage2.macro_precision);
        eprintln!("  Stage 2 Macro Recall: {:.2}%", metrics.stage2.macro_recall);
    } else if args.pos {
        // Train the POS tagging model with the Averaged Perceptron
        let mut trainer = PosTrainer::new(args.num_epochs, args.features_file.as_path())?;

        if let Some(model_uri) = &args.load_model_uri {
            trainer.load_model(model_uri).await?;
        }

        let metrics = trainer.train(&running, args.model_file.as_path())?;

        eprintln!("Result Metrics (POS):");
        eprintln!("  Accuracy: {:.2}% ( {} )", metrics.accuracy, metrics.num_instances);
        eprintln!("  Macro Precision: {:.2}%", metrics.macro_precision);
        eprintln!("  Macro Recall: {:.2}%", metrics.macro_recall);
    } else {
        // Train the word segmentation model with AdaBoost
        let mut trainer =
            Trainer::new(args.threshold, args.num_iterations, args.features_file.as_path())?;

        if let Some(model_uri) = &args.load_model_uri {
            trainer.load_model(model_uri).await?;
        }

        let metrics = trainer.train(&running, args.model_file.as_path())?;

        eprintln!("Result Metrics:");
        eprintln!(
            "  Accuracy: {:.2}% ( {} / {} )",
            metrics.accuracy,
            metrics.true_positives + metrics.true_negatives,
            metrics.num_instances
        );
        eprintln!(
            "  Precision: {:.2}% ( {} / {} )",
            metrics.precision,
            metrics.true_positives,
            metrics.true_positives + metrics.false_positives
        );
        eprintln!(
            "  Recall: {:.2}% ( {} / {} )",
            metrics.recall,
            metrics.true_positives,
            metrics.true_positives + metrics.false_negatives
        );
        eprintln!(
            "  Confusion Matrix:\n    True Positives: {}\n    False Positives: {}\n    False Negatives: {}\n    True Negatives: {}",
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
            metrics.true_negatives
        );
    }

    Ok(())
}

/// Writes one output line, treating a closed downstream pipe as normal
/// termination.
///
/// # Arguments
/// * `writer` - The output writer.
/// * `line` - The line to write (a newline is appended).
///
/// # Returns
/// `Ok(true)` to continue writing, `Ok(false)` when the downstream consumer
/// closed the pipe (e.g. `litsea segment model | head -1`), or the original
/// error for any other I/O failure.
fn write_output_line<W: Write>(writer: &mut W, line: &str) -> io::Result<bool> {
    match writeln!(writer, "{}", line) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(e),
    }
}

/// Flushes the output writer, treating a closed downstream pipe as normal
/// termination.
///
/// # Arguments
/// * `writer` - The output writer to flush.
///
/// # Returns
/// `Ok(())` on success or broken pipe; any other I/O error is returned so it
/// is surfaced instead of being lost in the writer's drop.
fn flush_output<W: Write>(writer: &mut W) -> io::Result<()> {
    match writer.flush() {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e),
    }
}

/// Segment sentences read from standard input using a trained model.
/// This function loads the model from the given model URI (a plain path, a
/// `file://` path, or an `http(s)://` URL with the `remote_model` feature):
/// with `--pos`, a joint (Averaged Perceptron) or two-stage POS model, with
/// the model kind auto-detected from the file (issue #147); otherwise an
/// AdaBoost model (word segmentation only). The segmented sentences are
/// written to standard output.
///
/// A downstream consumer closing stdout early (broken pipe) terminates the
/// command successfully instead of reporting an error.
///
/// # Arguments
/// * `args` - The arguments for the segment command [`SegmentArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
async fn segment(args: SegmentArgs) -> Result<(), Box<dyn Error>> {
    let language = args.language;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    if args.pos {
        // Joint or two-stage segmentation + POS tagging; the model kind is
        // auto-detected from the file (issue #147).
        let model = AnyPosModel::load(args.model_uri.as_str()).await?;
        let segmenter = model.into_segmenter(language);

        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let tokens = segmenter.segment_with_pos(line)?;
            let formatted: Vec<String> =
                tokens.iter().map(|(word, pos)| format!("{}/{}", word, pos)).collect();
            if !write_output_line(&mut writer, &formatted.join(" "))? {
                return Ok(());
            }
        }
    } else {
        // Word segmentation only, with an AdaBoost model
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model(args.model_uri.as_str()).await?;

        let segmenter = Segmenter::with_learner(language, learner);

        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let tokens = segmenter.segment(line);
            if !write_output_line(&mut writer, &tokens.join(" "))? {
                return Ok(());
            }
        }
    }

    flush_output(&mut writer)?;
    Ok(())
}

/// Evaluate a model against a held-out gold corpus and print quality
/// metrics.
///
/// Loads an AdaBoost model, or, with `--pos`, a joint (Averaged Perceptron)
/// or two-stage model auto-detected from the file (#147), from the model
/// URI, parses the gold corpus in the selected format (space-separated
/// tokens, tab-separated `tsv` tokens, or `word/POS` with `--pos`), and
/// prints held-out precision/recall/F1 one metric per line.
///
/// # Arguments
/// * `args` - The arguments for the evaluate command [`EvaluateArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
async fn evaluate(args: EvaluateArgs) -> Result<(), Box<dyn Error>> {
    let gold_file = File::open(args.gold_file.as_path())?;
    let reader = io::BufReader::new(gold_file);

    if args.pos {
        // Joint or two-stage model, auto-detected from the file (#147).
        let model = AnyPosModel::load(args.model_uri.as_str()).await?;
        let segmenter = model.into_segmenter(args.language);

        let gold = reader
            .lines()
            .collect::<Result<Vec<String>, _>>()?
            .into_iter()
            .map(|line| evaluation::parse_gold_pos_line(&line));
        let metrics = evaluation::evaluate_pos(&segmenter, gold)?;

        let seg = &metrics.segmentation;
        eprintln!("Evaluation Metrics (POS):");
        eprintln!("  Sentences: {}", seg.sentences);
        eprintln!("  Word Precision: {:.2}%", seg.word_precision);
        eprintln!("  Word Recall: {:.2}%", seg.word_recall);
        eprintln!("  Word F1: {:.2}%", seg.word_f1);
        eprintln!("  Tagged Word Precision: {:.2}%", metrics.tagged_precision);
        eprintln!("  Tagged Word Recall: {:.2}%", metrics.tagged_recall);
        eprintln!("  Tagged Word F1: {:.2}%", metrics.tagged_f1);
    } else {
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model(args.model_uri.as_str()).await?;
        let segmenter = Segmenter::with_learner(args.language, learner);

        let tsv = args.format == "tsv";
        let gold = reader
            .lines()
            .collect::<Result<Vec<String>, _>>()?
            .into_iter()
            .map(|line| evaluation::parse_gold_line(&line, tsv));
        let metrics = evaluation::evaluate_segmentation(&segmenter, gold);

        eprintln!("Evaluation Metrics:");
        eprintln!("  Sentences: {}", metrics.sentences);
        eprintln!("  Word Precision: {:.2}%", metrics.word_precision);
        eprintln!("  Word Recall: {:.2}%", metrics.word_recall);
        eprintln!("  Word F1: {:.2}%", metrics.word_f1);
        eprintln!("  Boundary Precision: {:.2}%", metrics.boundary_precision);
        eprintln!("  Boundary Recall: {:.2}%", metrics.boundary_recall);
        eprintln!("  Boundary F1: {:.2}%", metrics.boundary_f1);
    }

    Ok(())
}

/// Parses the command-line arguments and dispatches to the selected
/// subcommand.
///
/// # Returns
/// Returns a Result carrying the outcome of the executed subcommand.
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = CommandArgs::parse();

    match args.command {
        Commands::Extract(args) => extract(args),
        Commands::Train(args) => train(args).await,
        Commands::Segment(args) => segment(args).await,
        Commands::Evaluate(args) => evaluate(args).await,
    }
}

/// Entry point: runs the CLI and exits with status 1 after printing the
/// error message if a subcommand fails.
#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
