use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};

use litsea::adaboost::AdaBoost;
use litsea::extractor::Extractor;
use litsea::get_version;
use litsea::segmenter::Segmenter;
use litsea::trainer::Trainer;

/// Arguments for the extract command.
#[derive(Debug, Args)]
#[clap(
    author,
    about = "Extract features from a corpus",
    version = get_version(),
)]
struct ExtractArgs {
    corpus_file: PathBuf,
    features_file: PathBuf,
}

/// Arguments for the train command.
#[derive(Debug, Args)]
#[clap(author,
    about = "Train a segmenter",
    version = get_version(),
)]
struct TrainArgs {
    #[arg(short, long, default_value = "0.01")]
    threshold: f64,

    #[arg(short = 'i', long, default_value = "100")]
    num_iterations: usize,

    #[arg(short = 'n', long, default_value = "1")]
    num_threads: usize,

    #[arg(short = 'm', long)]
    load_model_uri: Option<String>,

    features_file: PathBuf,
    model_file: PathBuf,
}

/// Arguments for the segment command.
#[derive(Debug, Args)]
#[clap(author,
    about = "Segment a sentence",
    version = get_version(),
)]
struct SegmentArgs {
    model_uri: String,
}

/// Subcommands for lietsea CLI.
#[derive(Debug, Subcommand)]
enum Commands {
    Extract(ExtractArgs),
    Train(TrainArgs),
    Segment(SegmentArgs),
}

/// Arguments for the litsea command.
#[derive(Debug, Parser)]
#[clap(
    name = "litsea",
    author,
    about = "A morphological analysis command line interface",
    version = get_version(),
)]
struct CommandArgs {
    #[clap(subcommand)]
    command: Commands,
}

/// Extract features from a corpus file and write them to a specified output file.
/// This function reads sentences from the corpus file, segments them into words,
/// and writes the extracted features to the output file.
///
/// # Arguments
/// * `args` - The arguments for the extract command [`ExtractArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
fn extract(args: ExtractArgs) -> Result<(), Box<dyn Error>> {
    let mut extractor = Extractor::new();

    extractor.extract(args.corpus_file.as_path(), args.features_file.as_path())?;

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
    })
    .expect("Error setting Ctrl-C handler");

    let mut trainer = Trainer::new(
        args.threshold,
        args.num_iterations,
        args.num_threads,
        args.features_file.as_path(),
    );

    if let Some(model_uri) = &args.load_model_uri {
        trainer.load_model(model_uri).await?;
    }

    let metrics = trainer.train(running, args.model_file.as_path())?;

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

    Ok(())
}

/// Segment a sentence using the trained model.
/// This function loads the AdaBoost model from the specified file,
/// reads sentences from standard input, segments them into words,
/// and writes the segmented sentences to standard output.
///
/// # Arguments
/// * `args` - The arguments for the segment command [`SegmentArgs`].
///
/// # Returns
/// Returns a Result indicating success or failure.
async fn segment(args: SegmentArgs) -> Result<(), Box<dyn Error>> {
    let mut leaner = AdaBoost::new(0.01, 100, 1);
    leaner.load_model(args.model_uri.as_str()).await?;

    let segmenter = Segmenter::new(Some(leaner));
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = segmenter.segment(line);
        writeln!(writer, "{}", tokens.join(" "))?;
    }

    Ok(())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = CommandArgs::parse();

    match args.command {
        Commands::Extract(args) => extract(args),
        Commands::Train(args) => train(args).await,
        Commands::Segment(args) => segment(args).await,
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
