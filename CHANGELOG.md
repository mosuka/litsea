# Changelog

## 0.6.0 (unreleased)

Note: crates.io already has 0.5.0 published, so the accumulated unreleased
changes below (originally drafted under a "0.5.0 (unreleased)" heading)
will ship as 0.6.0. This release contains the breaking API changes from
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

### Changed (breaking)

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

### Migration notes

- Replace `learner.load_model(path).await?` with
  `learner.load_model_from_path(Path::new(path))?` for local files; keep
  `load_model` for URLs.
- Replace `segmenter.learner` / `segmenter.pos_learner` field access with
  `learner()` / `learner_mut()` / `pos_learner()` / `pos_learner_mut()`.
- Error matching: handle `LitseaError` variants instead of
  `std::io::ErrorKind`.
