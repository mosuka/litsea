# Changelog

## Unreleased

### Added

- A pointwise fast path for `segment()` (#183): `PackedModel` records at
  compile time whether any of the 16 tag-dependent templates
  (`UP*`/`BP*`/`UQ*`/`BQ*`/`TQ*`, which read the previous boundary
  decisions) carries a non-zero weight, and when none does, the sequential
  scoring pass and its tag bookkeeping are skipped entirely. Output is
  unchanged in both directions -- the skipped loads would each add `0.0` --
  pinned by a tag-free differential test against the string-keyed
  reference; models with tag features are unaffected. The serial
  dependency between positions was measured at ~10-25% of `segment()`
  runtime (patch experiment, paired runs), so a pointwise model recovers
  that in full.
- `litsea extract --tag-free` and the backing
  `Extractor::extract_tag_free` / `Extractor::extract_tsv_tag_free`
  methods: extract boundary features with the 16 tag-dependent templates
  dropped, producing training data for pointwise models. Composable with
  `--format tsv`; rejected with `--pos` / `--two-stage`.

### Changed

- `models/korean.model` is retrained tag-free (#183): with the
  space-preserving corpus the tag features measured as contributing
  nothing (held-out Word F1 99.91% vs. 99.90%, Boundary F1 99.96% vs.
  99.95%), and the model shrinks from 3,994 features / ~110 KB to 3,132
  features / ~86 KB while becoming eligible for the pointwise fast path.
  Segmentation output on the golden-test sentences is unchanged. The
  bundled Japanese and Chinese models keep their tag features -- the
  measured trade-off (ja: -0.37pt Word F1 for ~+45-50% throughput, zh:
  -0.51pt for ~+12%) is documented in Pre-trained Models for users who
  prefer to retrain for speed.

## 0.11.0

### Added

- The `litsea-two-stage v1` model file format and its container type
  `TwoStageLearner` (#166, part of the two-stage segmentation + per-word
  POS tagging architecture, #147): a single headered text file bundling a
  stage-1 boundary classifier (embedded AdaBoost format), a candidate-tag
  lexicon with occurrence counts, a stage-2 word-level tagger (embedded
  Averaged Perceptron format), and an optional `dominance` classifier-skip
  threshold. A `ModelKind::detect` helper identifies the three model
  formats for loader dispatch. The format is purely additive: existing
  model files load unchanged and the existing loaders reject two-stage
  files with an explicit error.
- `AdaBoost::save_model_to_writer` and
  `AveragedPerceptron::save_model_to_writer` (the format-producing cores of
  the existing `save_model` methods, enabling section embedding), and an
  `AveragedPerceptron::classes()` accessor.
- The two-stage runtime behind the unchanged `segment_with_pos()` signature
  (#167): `Segmenter::with_two_stage_learner` installs a two-stage model's
  stage-1 boundary classifier as the segmenter's AdaBoost-path learner and
  tags each segmented word through the candidate-tag lexicon
  (single-candidate and dominance-dominant surfaces skip the classifier
  entirely) and a packed word-level stage-2 tagger (candidate-masked argmax
  for ambiguous surfaces, full argmax fallback for unknown surfaces). On
  the held-out UD GSD test the runtime reproduces the #147 prototype:
  Japanese word F1 96.48 (joint: 96.56) with tagged-word F1 92.71-93.41
  depending on the stage-2 feature set (joint: 92.51) at 1.5-2.5x the
  joint throughput. Word-level feature templates live in a single
  crate-private module shared with the training extractor below.
- The two-stage training pipeline and CLI support (#168):
  `Extractor::extract_two_stage` derives stage-1 boundary features, stage-2
  word-level features, and the candidate-tag lexicon from a single
  POS-tagged corpus pass, with a `TwoStageFeatureSet` (`Full` / `Balanced`
  / `Fast`, default `Fast`) selecting which stage-2 templates to write;
  `TwoStageTrainer` trains both stages, collapses the stage-1 boundary
  perceptron to scalar AdaBoost-format weights (a lossless transformation —
  see the `litsea::trainer` module docs), and assembles + saves a
  `litsea-two-stage v1` model, reporting `TwoStageMetrics` for both stages.
  `AnyPosModel::load` fetches a POS-capable model once and auto-detects
  whether it is a joint or two-stage model. The `litsea` CLI gains
  `extract --two-stage [--stage2-features full|balanced|fast]` and
  `train --two-stage [--dominance 0.99]`; `segment --pos` and
  `evaluate --pos` now auto-detect joint vs. two-stage models via
  `AnyPosModel`, with no new flags. Verified end-to-end on UD
  Japanese-GSD: `extract --two-stage` + `train --two-stage` (fast set)
  reproduces the #167 runtime numbers exactly (held-out word F1 96.48,
  tagged F1 92.71) in a 5.1 MB model file (vs. 11 MB for the joint model).
- Bundled two-stage models, cross-language validation, benches, and docs
  (#169): `models/{japanese,chinese,korean}_two_stage.model`, trained at
  50 epochs (chosen from an epoch sweep -- see below) with `fast`
  (Japanese) or `balanced` (Chinese, Korean) stage-2 features. As bundled,
  every language beats the currently published joint model on both Word
  F1 and Tagged F1: Japanese 96.78/92.95 (joint 96.56/92.51), Chinese
  90.82/82.29 (joint 90.52/81.18), Korean 83.24/78.86 (joint 80.51/71.03),
  at 1.8-2.8x the joint throughput on `cargo bench -- external_corpus`
  (three-run ranges: Japanese 2.65-3.05x, Chinese 2.13-2.44x, Korean
  1.75-1.90x). An epoch sweep (10-150 epochs) run during bundling found
  that stage-1 segmentation quality continues improving well past the
  joint models' original 10-epoch convention and plateaus around 50; at
  matched epoch counts, joint retains a small (~0.5-0.9pt), reproducible
  Chinese segmentation edge, documented rather than papered over. Korean
  two-stage uses the same unspaced `word/POS` protocol as `korean_pos.model`
  (not the space-preserving protocol `korean.model` uses), also documented
  explicitly. New docs: `docs/src/algorithm/two-stage-tagging.md` (+ JA)
  compares the two architectures and their tradeoffs; `pre-trained-models.md`
  and `training-guide/training-models.md` (+ JA mirrors) gain two-stage
  sections; `litsea/benches/bench.rs`'s `external_corpus` group gains
  `{japanese,chinese,korean}-two-stage` cases alongside the existing
  AdaBoost/joint ones.

### Changed

- `japanese.model`, `chinese.model`, and `korean.model` are retrained as a
  2-class (boundary/non-boundary) Averaged Perceptron and collapsed to
  scalar per-feature weights in the existing AdaBoost model format (#165)
  -- a lossless transform (see `scripts/collapse_binary_perceptron.py`'s
  docstring for the derivation), not an approximation; the file format and
  `Segmenter`/`AdaBoost` loading are unchanged. Held-out Word F1: Japanese
  91.48 -> 96.70 (+5.22), Chinese 77.56 -> 90.69 (+13.13), Korean 99.91 ->
  99.90 (space-preserving protocol, unchanged within measurement noise --
  Korean's space signal already made the task nearly deterministic for the
  prior AdaBoost model). An epoch sweep (10-200) and a magnitude-pruning
  sweep were run per language before settling on the shipped
  hyperparameters (Japanese: 50 epochs, pruned to the top 40,000 features;
  Chinese: 100 epochs, pruned to the top 70,000; Korean: 30 epochs,
  unpruned at 3,994 features) -- see `docs/src/pre-trained-models.md`'s
  new "Training Procedure" section for the reproducible recipe (two new
  scripts, `scripts/collapse_binary_perceptron.py` and
  `scripts/prune_adaboost_model.py`, no crate changes). Model files grow
  substantially (Japanese ~20 KB -> ~1.1 MB, Chinese ~18 KB -> ~2.0 MB,
  Korean ~9.4 KB -> ~110 KB), but a paired `cargo bench --
  external_corpus` comparison against the previous bundled files showed no
  measurable throughput regression (within run-to-run noise on this
  project's development machine) after pruning -- substantially better
  than the ~20% regression anticipated when this issue was filed. Two
  `litsea/tests/golden.rs` segmentation snapshots changed accordingly
  (`"こんにちは"` and `"我喜欢吃中国菜。"`) and were updated to the new,
  measurably better output.

## 0.10.0

### Added

- A `litsea evaluate` subcommand and a public `litsea::evaluation` module
  for held-out quality measurement (#161): word/boundary precision, recall,
  and F1 over exact character-offset spans (whitespace tokens excluded —
  the Korean space-preserving protocol), plus tagged-word metrics with
  `--pos`. The UD GSD test splits ship as gold data in `resources/eval/`
  (CC BY-SA 4.0, attributed in its README), so every documented held-out
  figure is reproducible with one command. First held-out POS measurements:
  word / tagged-word F1 96.56 / 92.51 (japanese_pos), 80.51 / 71.03
  (korean_pos), 90.52 / 81.18 (chinese_pos).
- An `external_corpus` Criterion benchmark group reproducing the seven
  litsea benches of the external tokenizer-speed-bench harness in-repo
  (#159): the same corpora (wagahaiwa_nekodearu / mujeong / rulin_waishi,
  vendored byte-identical into `resources/`), the same per-line workload,
  and `Throughput::Elements` so Criterion reports chars/sec directly.
  Throughput regressions can now be caught with `cargo bench` alone.

### Performance

- `segment()` scores the `WC*` (char + type) templates with one merged row
  probe per character instead of four keyed hash probes per decision
  position (#157). Weights and outputs are unchanged (pinned by the
  packed-vs-reference differential tests, including a new full-corpus
  sweep over every bundled Japanese model). On the external benchmark
  corpus this recovers most of the WC cost the retrained `japanese.model`
  pays for its 96 `WC` features: ~10.6M → ~11.8M chars/s (interleaved
  A/B, medians); models without `WC` features (e.g. `RWCP.model`) are
  unaffected. The POS path keeps the keyed gather: its per-class rows
  would make per-character merged rows memory-heavy, and its throughput
  is bounded by 17-class scoring (#147).

## 0.9.0

### Added

- Space-preserving TSV corpus format for word-segmentation training (#152):
  `Segmenter::add_corpus_tsv` / `add_corpus_tsv_with_writer`,
  `Extractor::extract_tsv`, the `litsea extract --format tsv` CLI flag, and
  a `-s` flag on `scripts/corpus_udtreebank.sh` that reconstructs the
  original spacing from `SpaceAfter` annotations. Tokens are tab-separated
  and a token may be a literal space, so the training text keeps the
  sentence's original spaces.

### Changed

- `korean.model` is retrained on the space-preserving TSV corpus (#152).
  The previous pipeline concatenated the training words without spaces, so
  the model never saw the inter-eojeol space — Korean's strongest boundary
  signal — during training. Held-out word F1 on real spaced text improves
  from 92.91 to 99.91 (boundary F1 97.30 → 99.96) and the model shrinks
  from ~20 KB to ~9.4 KB. The previously documented word F1 of 65.37
  measured a space-stripped protocol that does not reflect real input;
  metrics are now reported on the original spaced text (space tokens
  excluded from scoring). Golden test expectations were updated
  (이것은 / 고양이를 are now single words).

- The bundled AdaBoost segmentation models were retrained with more boosting
  iterations (#151). The previous models were trained with the CLI defaults
  (`-t 0.01 -i 100`); AdaBoost selects one feature per boosting iteration, so
  they contained only 40-50 features and performed far below their potential
  on held-out text. All three models are retrained on their existing UD GSD
  training corpora with `-t 0.0001 -i 20000`. Held-out test-split word F1:
  `japanese.model` 75.44 → 91.48, `korean.model` 40.96 → 65.37, and
  `chinese.model` 64.39 → 77.56. The model files grow from ~1.1-1.4 KB to
  ~18-20 KB each; the file format is unchanged. Golden test expectations were
  updated for the retrained models.

### Documentation

- The documentation (English and Japanese) now reports held-out word and
  boundary F1 for the bundled models together with the training options used,
  and clarifies that the `train` command prints in-sample metrics (#151).
- The mdbook documentation and source doc comments are re-synchronized with
  the current implementation (#153). The architecture "Workspace Structure"
  page is removed: it duplicated `Cargo.toml` and the repository layout and
  went stale on every release.

## 0.8.0

### Performance

`segment_with_pos()` on the packed two-pass pipeline (#144):
`segment_with_pos()` still ran the pre-#137 string-keyed pipeline — 42
feature strings built and double-copied per character, 42 string-keyed hash
probes, and a per-character class-name parse. The Averaged Perceptron is now
compiled once at model load into a `PackedPosModel`, the multiclass
counterpart of the `segment()` scoring tables: char-bearing templates become
one packed-u64-key hash map with sparse per-class weight rows (perceptron
updates touch only the gold/predicted pair, so features average 3.3 non-zero
classes), tag/type-only templates become direct-indexed dense tables with a
presence bitset so the sequential pass skips rows of absent features
entirely, and class labels are parsed once at build time. Each sentence is
scored in two passes: a static pass scatter-adds every tag-free feature into
an n × n_classes score matrix, and a sequential pass adds the 16
tag-dependent dense rows and takes the argmax. Long Japanese text:
198.1 ms → 54.6 ms (3.6x raw, ~3.2x normalized by the untouched AdaBoost
benchmark). The string-keyed path is kept test-only, and differential tests
over all three bundled POS models, bocchan lines, and stress sentences
measure zero output divergence.

An f32 variant of the scoring tables was evaluated and rejected (#145): it
also measured zero output divergence, but an A/B benchmark in matched
machine states showed no speedup beyond noise, so the f64 tables stay.

### Documentation

- The prediction-pipeline page describes the packed two-pass POS scorer, and
  the `packed_model` / `packed_pos_model` private modules were added to the
  module-design pages and the `lib.rs` listing (English and Japanese) (#144).

## 0.7.0

This release completes the `segment()` scoring rework begun with the packed
u64 feature keys of #137. There are no API changes, and the on-disk model
files are unchanged.

### Performance

Packed u64 feature keys (#137; merged as the final commit of v0.6.0 and not
covered by the 0.6.0 entry below, so described here with the rest of the
campaign): the AdaBoost model's string feature keys are compiled into packed
u64 keys (template id in the top byte, tag/type ids in 8 bits, char codes in
24 bits with sentinels just above the Unicode scalar range) once at model
load, and `segment()`'s hot loop scores by integer map probes instead of
formatting 42 `String`s per character and hashing them. The feature strings
are derived from a declarative 42-entry template table (`packed_model.rs`),
and `Language::char_type` became a table lookup over numeric char-type ids.
The compiled table is invalidated by the two learner-mutation gateways
(`learner_mut`, `add_corpus`) and lazily rebuilt. A miss adds 0.0, so the
f64 accumulation sequence matches the string-keyed reference bit for bit;
differential tests pin equality across all bundled models. Throughput on
the local tinysegmenter comparison harness rose from 698K to ~3.6M chars/s
(~5x). The on-disk model format is unchanged.

Direct-indexed dense tables for tag/type-only templates (#140): 29 of the 42
feature templates contain only tag/type slots, whose mixed-radix key spaces
are tiny (about 9.3K entries / 74 KB total for Japanese). Their weights are
compiled into per-template dense f64 arrays at model load and scored with a
single array load instead of a hashed map probe; the 13 char-bearing
templates (UW/BW/WC) keep the packed-u64 map. Unset dense entries hold 0.0
and the template iteration order is unchanged, so the segmentation output
stays bit-for-bit identical.

Two-pass scatter-add scoring (#141): only 16 of the 38-42 feature templates
depend on earlier boundary decisions. The model is compiled into
merged-vector tables (char → [UW1..6], char pair → [BW1..3],
(char, type) → WC, plus scatter-vector views of the UC/BC/TC dense tables),
and each sentence is scored in two passes: a static pass scatter-adds every
tag-free feature into a per-position buffer in one sweep, and a sequential
pass adds the 16 tag-dependent dense loads and decides boundaries.
Per-position hash probes drop from 13 to about 2 plus WC. The f64
accumulation order now differs from the string-keyed reference, so
bit-for-bit output identity is no longer structural; the exact-equality
differential suite (all bundled models, sentinel stress strings, 100 lines
of bocchan.txt) passes unchanged and remains the detection net. In-process
A/B against the previous stage: japanese.model 3.00x, RWCP.model 4.11x
(11-12M chars/s under load); instruction count drops 3.1x and cycles 3.25x.

### Documentation

- The prediction-pipeline, adaboost, feature-extraction, model-file-format,
  and adding-a-new-language pages (English and Japanese) were rewritten for
  the compiled scoring tables and the two-pass scorer (#137, #140, #141).

## 0.6.0

Note: crates.io already had 0.5.0 published, so the accumulated changes
below (originally drafted under a "0.5.0 (unreleased)" heading) shipped
as 0.6.0. This release contains the breaking API changes from
Phase 3 of the refactoring plan (`REFACTORING_PLAN.md`) plus the #97
quality campaign. Model files remain fully compatible: all pre-trained
models in `models/` load unchanged.

### Added

- `Language::char_type(char) -> &'static str`: allocation-free, match-based
  character classification (replaces the regex-based `CharTypePatterns`).
- `litsea::error::LitseaError` and `litsea::error::Result<T>`: a proper
  error enum (`Io`, `InvalidData`, `InvalidInput`, `Unsupported`, and
  `Download` with the `remote_model` feature) replacing the previous mix of
  `std::io::Error` and `Box<dyn Error>`.
- Synchronous model loading: `AdaBoost::load_model_from_path` /
  `load_model_from_reader` and `AveragedPerceptron::load_model_from_path` /
  `load_model_from_reader`. The async `load_model(uri)` remains for URI-based
  loading (`file://`, plain paths, and `http(s)://` with the `remote_model`
  feature). Local workflows no longer need an async runtime.
- Top-level re-exports: `litsea::{AdaBoost, AveragedPerceptron, Extractor,
  Language, LitseaError, PosTrainer, Result, Segmenter, SegmentLabel,
  Trainer, Upos, BinaryMetrics, MulticlassMetrics}`.
- `Segmenter` accessors: `language()`, `learner()`, `learner_mut()`,
  `pos_learner()`, `pos_learner_mut()`.
- `Debug` implementations for `Segmenter`, `Extractor`, `Trainer`, and
  `PosTrainer`; `Extractor::extract` / `extract_with_pos` take `&self`
  (they never mutate); CLI help texts are in English and subcommands
  inherit the version via `propagate_version` (#129).

### Fixed

- Incremental training (`litsea train -m <model>`): loading a model after
  training data had been initialized rebuilt the feature index from the
  model file, leaving the already-built instance data pointing at stale
  indices. `AdaBoost::load_model_from_reader` now merges weights by feature
  name into the existing index, and `AveragedPerceptron::load_model_from_reader`
  merges model classes with classes already registered from training data.
- `AdaBoost` assumed the bias bucket (the empty-string feature `""`) always
  occupied feature index 0, but the `add_instance` path (used by
  `Segmenter::add_corpus`) let an arbitrary real feature claim that slot:
  training could then silently corrupt or never select that feature
  (nondeterministically, depending on `HashSet` iteration order), and
  `save_model` dropped it from the saved file. The bias feature is now
  registered at index 0 on every construction path, `save_model` identifies
  the bias bucket by name instead of by index, and `train()` no longer
  panics on a learner with no instances (#98).
- Feature and model files are now parsed strictly as tab-separated.
  `AdaBoost::initialize_features` / `initialize_instances` and
  `AdaBoost::load_model_from_reader` previously split lines on any Unicode
  whitespace, silently mangling features that embed characters such as the
  ideographic space (U+3000) — common inside Japanese corpus tokens — so
  their trained weights were unreachable at inference time.
  `PosTrainer::new` similarly trimmed Unicode whitespace off line edges,
  destroying a trailing U+3000 in the last feature. Blank lines are now
  skipped consistently by both feature-file passes. Note: legacy
  hand-written space-separated features files are no longer accepted; the
  `extract` command has always produced tab-separated output (#99).
- `litsea segment` now treats a downstream consumer closing stdout early
  (e.g. `litsea segment model | head -1`) as normal termination instead of
  reporting `Error: Broken pipe` and exiting 1, and explicitly flushes its
  output buffer so real I/O errors are surfaced instead of being lost in
  the writer's drop (#102).
- Model loading now fails loudly on corrupt data instead of silently
  producing a broken model: both loaders reject non-finite weights
  (`NaN`/`inf`, which poisoned every score comparison), and the AdaBoost
  loader validates the file format — empty files, files without a bias
  line (the typical symptom of a truncated download), duplicate bias
  lines, and duplicate feature lines are rejected. Weight lines after the
  bias line remain accepted for legacy models (e.g. `RWCP.model`).
  `AveragedPerceptron::load_model_from_reader` also resets the averaging
  accumulators so an incremental train after a load no longer mixes stale
  state into the averaged weights. Remote downloads gained a 10 s connect /
  60 s request timeout, a 256 MiB size cap, and Content-Length
  verification (#101).
- The POS training pipeline never emitted the first character position of a
  sentence, while `segment_with_pos` predicts at that position to derive
  the first word's POS (since the Phase 2 fix). Sentence-initial sentinel
  features therefore had zero weight and first-word POS relied on right
  context only. The POS pipeline now emits the first position (the
  AdaBoost boundary pipeline is unchanged), and the shipped POS models
  were retrained: first-word POS accuracy on the UD-GSD dev sets improved
  from 85.6% to 91.8% (Japanese), 85.2% to 90.9% (Chinese), and 73.2% to
  85.2% (Korean), with first-word boundary accuracy unchanged (#100).
- `japanese_char_type` classified kanji as `'\u{4E00}'..='\u{9FA0}'`, 95 code
  points short of the full CJK Unified Ideographs block
  (`'\u{4E00}'..='\u{9FFF}'`) already used by `chinese_char_type` and
  `korean_char_type`. Characters in U+9FA1..U+9FFF were classified as `"O"`
  (Other) for Japanese only. The range is now unified across all three
  languages. The UD Japanese-GSD corpus that trained `japanese.model` and
  `japanese_pos.model` contains no characters in the affected range, so this
  introduces no train/inference asymmetry; the shipped models are unchanged
  (#130).

### Changed (breaking)

- `AdaBoost::threshold` / `num_iterations` are private; read them via the
  new `threshold()` / `num_iterations()` accessors (set them via `new`) (#129).
- All `train` methods take `running: &AtomicBool` instead of
  `Arc<AtomicBool>`. Migration: pass `&running` (keep the `Arc` only if you
  share the flag with e.g. a signal handler) (#129).
- `Segmenter::char_type` takes `char` and returns `&'static str`, aligned
  with `Language::char_type`; the empty-string case no longer exists (#129).
- The `remote_model` feature is no longer a default feature: the library
  default is local model loading only. Enable it explicitly for
  `http(s)://` model URIs (`features = ["remote_model"]`); the CLI enables
  it and is unchanged (#129).
- `FromStr` for `Language`, `Upos`, and `SegmentLabel` now uses dedicated
  thiserror-derived error types (`ParseLanguageError`, `ParseUposError`,
  `ParseSegmentLabelError`, re-exported at the crate root) instead of
  `type Err = String`. Display messages are unchanged (including the
  clap-rendered CLI message), so only code that relied on the error being a
  `String` needs updating (#128).
- `Language` and `LitseaError` are now `#[non_exhaustive]`: external
  exhaustive matches need a wildcard arm. `Upos` (fixed 17-tag UD standard)
  and `SegmentLabel` (structurally complete) deliberately stay exhaustive
  (#128).
- `Segmenter::new(language, Option<AdaBoost>)` is now
  `Segmenter::new(language)` (default learner) plus
  `Segmenter::with_learner(language, learner)`; the 0.01/100 default lives
  in `impl Default for AdaBoost`. Migration:
  `Segmenter::new(lang, None)` → `Segmenter::new(lang)`;
  `Segmenter::new(lang, Some(l))` → `Segmenter::with_learner(lang, l)` (#127).
- `Segmenter::segment_with_pos` returns
  `litsea::Result<Vec<(String, Upos)>>` instead of panicking when no POS
  learner is set (`LitseaError::PosLearnerNotSet`). Migration: append `?`
  or `unwrap()` at call sites that previously relied on the panic (#127).
- All fallible APIs now return `litsea::Result<T>` instead of
  `std::io::Result<T>` / `Result<T, Box<dyn Error>>`.
- `AdaBoost::predict` takes `&HashSet<String>` instead of consuming the set.
- Renames: `AdaBoost::get_bias` → `bias`, `AdaBoost::get_metrics` → `metrics`,
  `AveragedPerceptron::get_metrics` → `metrics`,
  `Segmenter::get_type` → `char_type`.
- `adaboost::Metrics` → `litsea::metrics::BinaryMetrics`,
  `perceptron::Metrics` → `litsea::metrics::MulticlassMetrics`.
- `Segmenter` fields (`language`, `learner`, `pos_learner`) are private;
  use the new accessors.
- `Segmenter::get_attributes` is no longer public.
- The `litsea::util` module was removed (`ModelScheme` is now internal).
- `parse_model_content` was renamed to the public `load_model_from_reader`.
- `CharTypePatterns` and `Language::char_type_patterns()` were removed in
  favor of `Language::char_type(char)`; the `regex` dependency is gone.
  `Segmenter::char_type` is unchanged.

### Performance

Release profile tuning (#105): the workspace now sets `lto = "thin"` and
`codegen-units = 1` for release builds, enabling cross-codegen-unit
inlining of the per-character feature-scoring call chain. Long-text
segmentation improved a further 10-13% and short-sentence segmentation up
to 14% on the bundled criterion suite; a leaf release rebuild grows from
~2.5 s to ~49 s as the compile-time cost. One nanobenchmark off the
segmentation hot path (`AdaBoost::predict` on a `HashSet`, ~190 ns →
~230 ns) regressed from changed inlining decisions and is accepted as a
trade-off. `panic = "abort"` was considered and rejected: it cannot be
scoped to the CLI binary alone and would complicate release-profile tests
and benches that rely on unwinding.

Training and model I/O (#104): the perceptron's three parallel training
maps (`weights`/`accumulated`/`timestamps`) are merged into one slot map,
so a weight update costs a single hashed lookup with no allocation on the
hit path, and the averaging pass no longer clones every feature key —
`litsea train --pos` on the UD-Japanese-GSD features (277k instances,
10 epochs) went from 8.7 s to 6.2 s (−29%). Both `save_model`
implementations now write through a `BufWriter` (previously one syscall
per line — ~540k for chinese_pos.model), and the perceptron writes weight
lines in sorted feature order, making saved models byte-reproducible and
diffable (loading is order-independent; existing models stay compatible).
Model loading parses lines without per-line vector allocations. The slot
layout keeps averaging state lazily materialized so inference-only loads
do not pay for it (loaded-model RSS +18 MB for the 19 MB Chinese POS
model from the slot struct itself; load time unchanged). AdaBoost's
training loop also reuses its error buffer across boosting iterations.

Inference hot path, second pass (#103): internal feature maps switched from
the default SipHash to `rustc_hash::FxHashMap` (keys are internally
generated, so HashDoS resistance is unnecessary); the AdaBoost bias is
cached instead of recomputed as an O(model-size) sum per sentence;
`sentence_context` borrows characters from the input instead of allocating
a `String` per character; and the perceptron's per-prediction scores vector
is reused across positions. Short-sentence segmentation improved a further
21-39% and long-text segmentation ~16% across all three languages on the
bundled criterion suite, with golden-test outputs unchanged. Adds the
`rustc-hash` dependency (MIT OR Apache-2.0, no transitive dependencies).

Measured on the bundled models (criterion, medians, vs v0.4.0):

- `segment()`: 65–70% faster (long Japanese text: 611 ms → 215 ms).
  The bias term is computed once per sentence instead of once per character,
  and attribute scoring sums weights directly without building a `HashSet`.
- `segment_with_pos()`: 88–91% faster (long Japanese text: 4.48 s → 0.40 s).
  The perceptron weight layout is transposed to feature → per-class vector,
  reducing hash lookups per position from features × classes to features,
  and attribute buffers are reused across positions.
- Character classification: 61 ns → 9 ns per call (regex scan → `match` on
  `char` ranges).
- `AveragedPerceptron::train`: no longer clones all instances per call and
  no longer rebuilds a `HashSet` per instance per epoch.

### Documentation

- All source doc comments are now in English (Japanese documentation lives
  in the `docs/ja` mdbook).
- The mdbook documentation (English and Japanese) and README are updated to
  the v0.5.0 API; README model paths now point at `models/`.
- Re-synced the Japanese documentation mirror (`docs/ja/src/`, 34 of 41
  pages) with `docs/src/`, which had drifted since 2026-06-13 through the
  entire #97 quality campaign (#98-#130): stale API signatures (the old
  `Segmenter::new(language, Option<AdaBoost>)` constructor, `char_type(&str)`,
  `Arc<AtomicBool>`), stale benchmark/accuracy numbers, and missing sections
  (POS-mode API, `PosTrainer`, `extract_with_pos`, model validation, feature
  hygiene) are now current (#124).

### Migration notes

- Replace `learner.load_model(path).await?` with
  `learner.load_model_from_path(Path::new(path))?` for local files; keep
  `load_model` for URLs.
- Replace `segmenter.learner` / `segmenter.pos_learner` field access with
  `learner()` / `learner_mut()` / `pos_learner()` / `pos_learner_mut()`.
- Error matching: handle `LitseaError` variants instead of
  `std::io::ErrorKind`.
