//! Command-line interface for litsea.
//!
//! Provides four subcommands: `extract` (turn a corpus into training
//! features, or, with `--pos`, into the three feature files consumed
//! by two-stage POS training), `train` (train an AdaBoost segmentation
//! model, or, with `--pos`, a two-stage boundary+lexicon POS model, or,
//! with `--perceptron`, a generic Averaged Perceptron over opaque labels —
//! the training step of the bundled segmentation models' collapse recipe),
//! `segment` (segment sentences from standard input with a trained model),
//! and `evaluate` (measure held-out quality against a gold corpus).

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
    AdaBoost, Extractor, Language, LitseaError, PerceptronTrainer, SegmentBuffer, Segmenter,
    Trainer, TwoStageFeatureSet, TwoStageLearner, TwoStageTrainer, evaluation,
};

/// Arguments for the extract command.
#[derive(Debug, Args)]
#[command(about = "Extract features from a corpus")]
struct ExtractArgs {
    /// Language of the corpus (japanese, chinese, or korean)
    #[arg(short, long, default_value = "japanese", value_parser = Language::from_str)]
    language: Language,

    /// Corpus format: "space" (space-separated words) or "tsv" (tab-separated
    /// tokens; a token may be a literal space, preserving original spacing)
    #[arg(long, default_value = "space", value_parser = ["space", "tsv"])]
    format: String,

    /// Extract two-stage POS training features (issue #147) from a
    /// POS-tagged corpus (format: "word/POS word/POS ...") instead: writes
    /// {features_file}.stage1 (boundary features), .stage2 (word-level
    /// features), and .lexicon. Cannot be combined with --format tsv.
    #[arg(long)]
    pos: bool,

    /// Stage-2 word-feature set for --pos: "full" (best quality),
    /// "balanced", or "fast" (best throughput; default)
    #[arg(long, default_value = "fast", value_parser = TwoStageFeatureSet::from_str)]
    stage2_features: TwoStageFeatureSet,

    /// Exclude the 16 tag-dependent feature templates (UP*/BP*/UQ*/BQ*/TQ*,
    /// which read the previous boundary decisions) so the trained model is
    /// pointwise and segment() skips its sequential scoring pass entirely
    /// (issue #183). Cannot be combined with --pos
    #[arg(long)]
    tag_free: bool,

    /// Path to the input corpus file (one pre-segmented sentence per line)
    corpus_file: PathBuf,
    /// Path to the output features file (with --pos, the prefix for
    /// the three output files)
    features_file: PathBuf,
}

/// Arguments for the train command.
#[derive(Debug, Args)]
#[command(about = "Train a segmenter")]
struct TrainArgs {
    /// Early-stopping threshold for AdaBoost training. Ignored with
    /// `--perceptron` or `--pos`
    #[arg(short, long, default_value = "0.01")]
    threshold: f64,

    /// Maximum number of AdaBoost boosting iterations. Ignored with
    /// `--perceptron` or `--pos`
    #[arg(short = 'i', long, default_value = "100")]
    num_iterations: usize,

    /// URI of an existing model to load before training (incremental training)
    #[arg(short = 'm', long)]
    load_model_uri: Option<String>,

    /// Train a generic Averaged Perceptron model (labels are opaque
    /// strings). This is the training step of the bundled segmentation
    /// models' collapse recipe (see scripts/collapse_binary_perceptron.py)
    #[arg(long)]
    perceptron: bool,

    /// Number of training epochs (applies to both `--perceptron` and
    /// `--pos` training; for `--pos`, both stage 1 and stage 2
    /// train for this many epochs)
    #[arg(long, default_value = "10")]
    num_epochs: usize,

    /// Train a two-stage POS model (issue #147) instead: reads
    /// {features_file}.stage1/.stage2/.lexicon (from extract --pos)
    /// and writes a litsea-two-stage model. Cannot be combined with
    /// --perceptron or -m/--load-model-uri (incremental training is not
    /// supported)
    #[arg(long)]
    pos: bool,

    /// Classifier-skip dominance threshold for --pos: a known word
    /// whose most frequent tag covers at least this fraction of its
    /// training occurrences is tagged without invoking the classifier.
    /// Must be in (0.5, 1.0]
    #[arg(long, default_value = "0.99")]
    dominance: f64,

    /// Path to the features file produced by the extract command (with
    /// --pos, the prefix passed to extract --pos)
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

    /// Segment with POS tagging (requires a two-stage model, from
    /// `train --pos`)
    #[arg(long)]
    pos: bool,

    /// Number of worker threads for batch segmentation (issue #185). The
    /// default (1) keeps the current single-threaded behavior; with N > 1,
    /// lines are processed in parallel and written in input order, so the
    /// output is byte-identical either way
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    threads: u16,

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
    /// ..."); requires a two-stage model (from `train --pos`)
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
/// features to the output file. With `--pos` the corpus is
/// POS-tagged and three files (boundary, word-level, and lexicon) are
/// written via `extract_two_stage` (issue #147); otherwise each line is
/// space-separated words, or tab-separated tokens with `--format tsv` (a
/// token may be a literal space, preserving the original spacing; not
/// supported with `--pos`). `--tag-free` (boundary pipeline only,
/// composable with `--format tsv`) drops the 16 tag-dependent templates so
/// the trained model is pointwise (issue #183).
///
/// # Arguments
/// * `args` - The arguments for the extract command [`ExtractArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
fn extract(args: ExtractArgs) -> Result<(), Box<dyn Error>> {
    let extractor = Extractor::new(args.language);

    if args.tag_free && args.pos {
        // Tag-free extraction is a boundary-pipeline concept (#183); the
        // two-stage POS pipeline uses its own label/feature scheme.
        return Err("--tag-free cannot be combined with --pos".into());
    }
    if args.pos {
        if args.format == "tsv" {
            return Err("--pos cannot be combined with --format tsv".into());
        }
        extractor.extract_two_stage(
            args.corpus_file.as_path(),
            args.features_file.as_path(),
            args.stage2_features,
        )?;
    } else if args.format == "tsv" && args.tag_free {
        extractor.extract_tsv_tag_free(args.corpus_file.as_path(), args.features_file.as_path())?;
    } else if args.format == "tsv" {
        extractor.extract_tsv(args.corpus_file.as_path(), args.features_file.as_path())?;
    } else if args.tag_free {
        extractor.extract_tag_free(args.corpus_file.as_path(), args.features_file.as_path())?;
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

    if args.pos {
        if args.perceptron {
            return Err("--pos cannot be combined with --perceptron".into());
        }
        if args.load_model_uri.is_some() {
            return Err("--pos does not support -m/--load-model-uri (incremental training)".into());
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
    } else if args.perceptron {
        // Train a generic Averaged Perceptron model (opaque string labels)
        let mut trainer = PerceptronTrainer::new(args.num_epochs, args.features_file.as_path())?;

        if let Some(model_uri) = &args.load_model_uri {
            trainer.load_model(model_uri).await?;
        }

        let metrics = trainer.train(&running, args.model_file.as_path())?;

        eprintln!("Result Metrics (Perceptron):");
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

/// Lines read per parallel chunk (issue #185): large enough to amortize the
/// per-chunk thread spawns and per-worker state reuse, small enough to
/// bound memory on unbounded streams.
const PARALLEL_CHUNK_LINES: usize = 4096;

/// Processes stdin lines through `process` across `threads` worker threads,
/// writing outputs to `writer` in input order (issue #185).
///
/// Lines are read in chunks of [`PARALLEL_CHUNK_LINES`]; each chunk is
/// split into `threads` contiguous slices, and each worker appends one
/// newline-terminated output line per non-empty trimmed input line to a
/// per-worker output buffer, using its own entry of `states` (worker-local
/// scratch, reused across chunks). The main thread then writes the worker
/// buffers in slice order, so the overall output is byte-identical to the
/// sequential loop: same trim and empty-line-skip behavior, same order. A
/// downstream broken pipe terminates successfully, matching
/// [`write_output_line`].
///
/// # Arguments
/// * `threads` - Number of worker threads (`states.len()` must match).
/// * `states` - One reusable worker-local state per thread.
/// * `process` - Renders one trimmed, non-empty input line into the output
///   buffer (no trailing newline; the driver adds it).
/// * `reader` - The line source (stdin).
/// * `writer` - The output sink (stdout).
///
/// # Returns
/// `Ok(())` when the input is exhausted or the downstream pipe closes.
///
/// # Errors
/// Propagates read/write I/O errors and any error returned by `process`.
fn process_lines_parallel<S, F, R, W>(
    threads: usize,
    states: &mut [S],
    process: F,
    reader: R,
    writer: &mut W,
) -> Result<(), Box<dyn Error>>
where
    S: Send,
    F: Fn(&str, &mut S, &mut String) -> Result<(), LitseaError> + Sync,
    R: BufRead,
    W: Write,
{
    debug_assert_eq!(states.len(), threads);
    let mut lines = reader.lines();
    let mut chunk: Vec<String> = Vec::with_capacity(PARALLEL_CHUNK_LINES);
    loop {
        chunk.clear();
        for line in lines.by_ref().take(PARALLEL_CHUNK_LINES) {
            chunk.push(line?);
        }
        if chunk.is_empty() {
            return Ok(());
        }
        let slice_len = chunk.len().div_ceil(threads);
        let outputs: Result<Vec<String>, LitseaError> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .chunks(slice_len)
                .zip(states.iter_mut())
                .map(|(slice, state)| {
                    let process = &process;
                    scope.spawn(move || {
                        let mut out = String::new();
                        for raw in slice {
                            let line = raw.trim();
                            if line.is_empty() {
                                continue;
                            }
                            process(line, state, &mut out)?;
                            out.push('\n');
                        }
                        Ok(out)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("segmentation worker panicked"))
                .collect()
        });
        for out in outputs? {
            match writer.write_all(out.as_bytes()) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }
}

/// Segment sentences read from standard input using a trained model.
/// This function loads the model from the given model URI (a plain path, a
/// `file://` path, or an `http(s)://` URL with the `remote_model` feature):
/// with `--pos`, a two-stage POS model (issue #147); otherwise an AdaBoost
/// model (word segmentation only). The segmented sentences are written to
/// standard output.
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

    let threads = usize::from(args.threads);

    if args.pos {
        // Two-stage segmentation + POS tagging (issue #147). The loader
        // rejects non-two-stage files with a precise error message.
        let mut learner = TwoStageLearner::new();
        learner.load_model(args.model_uri.as_str()).await?;
        let segmenter = Segmenter::with_two_stage_learner(language, learner);

        if threads > 1 {
            // Parallel path (#185): workers need no reusable scratch for
            // the POS pipeline, so the per-worker state is empty.
            let mut states = vec![(); threads];
            process_lines_parallel(
                threads,
                &mut states,
                |line, (), out| {
                    for (k, (word, pos)) in segmenter.segment_with_pos(line)?.iter().enumerate() {
                        if k > 0 {
                            out.push(' ');
                        }
                        out.push_str(word);
                        out.push('/');
                        out.push_str(&pos.to_string());
                    }
                    Ok(())
                },
                stdin.lock(),
                &mut writer,
            )?;
            return flush_output(&mut writer).map_err(Into::into);
        }

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

        if threads > 1 {
            // Parallel path (#185): one reusable SegmentBuffer per worker
            // (issue #184), so the steady state allocates nothing per line
            // in any worker.
            let mut states: Vec<SegmentBuffer> =
                (0..threads).map(|_| SegmentBuffer::new()).collect();
            process_lines_parallel(
                threads,
                &mut states,
                |line, buf, out| {
                    for (k, &(start, end)) in segmenter.segment_into(line, buf).iter().enumerate() {
                        if k > 0 {
                            out.push(' ');
                        }
                        out.push_str(&line[start..end]);
                    }
                    Ok(())
                },
                stdin.lock(),
                &mut writer,
            )?;
            return flush_output(&mut writer).map_err(Into::into);
        }

        // Reuse one segmentation buffer and one output line across the
        // whole stream (issue #184): after the first few lines the loop
        // allocates nothing per line. Output is identical to the previous
        // `segment(line).join(" ")` — the ranges are the same tokens.
        let mut buf = SegmentBuffer::new();
        let mut out = String::new();
        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            out.clear();
            for (k, &(start, end)) in segmenter.segment_into(line, &mut buf).iter().enumerate() {
                if k > 0 {
                    out.push(' ');
                }
                out.push_str(&line[start..end]);
            }
            if !write_output_line(&mut writer, &out)? {
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
/// Loads an AdaBoost model, or, with `--pos`, a two-stage model (#147),
/// from the model URI, parses the gold corpus in the selected format
/// (space-separated tokens, tab-separated `tsv` tokens, or `word/POS` with
/// `--pos`), and prints held-out precision/recall/F1 one metric per line.
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
        // Two-stage model (#147). The loader rejects non-two-stage files
        // with a precise error message.
        let mut learner = TwoStageLearner::new();
        learner.load_model(args.model_uri.as_str()).await?;
        let segmenter = Segmenter::with_two_stage_learner(args.language, learner);

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
