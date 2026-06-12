# Changelog

## 0.5.0 (unreleased)

This release contains the breaking API changes from Phase 3 of the
refactoring plan (`REFACTORING_PLAN.md`). Model files remain fully
compatible: all pre-trained models in `models/` load unchanged.

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

### Changed (breaking)

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

### Migration notes

- Replace `learner.load_model(path).await?` with
  `learner.load_model_from_path(Path::new(path))?` for local files; keep
  `load_model` for URLs.
- Replace `segmenter.learner` / `segmenter.pos_learner` field access with
  `learner()` / `learner_mut()` / `pos_learner()` / `pos_learner_mut()`.
- Error matching: handle `LitseaError` variants instead of
  `std::io::ErrorKind`.
