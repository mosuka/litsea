# Two-Stage Model

The `two_stage` module defines the `litsea-two-stage v1` model container
(`TwoStageLearner`), the stage-2 feature-set selector (`TwoStageFeatureSet`),
and the auto-detecting POS-model loader (`AnyPosModel` / `ModelKind`). See
[Two-Stage vs. Joint Tagging](../algorithm/two-stage-tagging.md) for the
architecture and measured quality/speed comparison, and
[`TwoStageTrainer`](trainer.md#twostagetrainer) for training a model from
scratch.

## `TwoStageLearner`

```rust
pub struct TwoStageLearner {
    // private: stage1: AdaBoost,
    // private: stage2: AveragedPerceptron,
    // private: lexicon: HashMap<String, Vec<(Upos, u32)>>,
    // private: dominance: f64,
}
```

Owns the three parts of a two-stage model: a stage-1 boundary classifier
(scalar weights, AdaBoost format), a candidate-tag lexicon, and a stage-2
word-level tagger (`AveragedPerceptron`). Follows the same construction and
(de)serialization conventions as `AdaBoost` and `AveragedPerceptron`.

### Constructors

```rust
pub fn new() -> Self
pub fn from_parts(
    stage1: AdaBoost,
    stage2: AveragedPerceptron,
    lexicon: impl IntoIterator<Item = (String, Vec<(Upos, u32)>)>,
    dominance: f64,
) -> Result<Self>
```

`new` creates an empty learner (fill it with a `load_model*` call before
use). `from_parts` builds a learner from its pieces, validating the
combination: `dominance` must be in `(0.5, 1.0]`, every stage-2 class name
must be a valid `Upos` tag, and every lexicon entry must have a non-empty
surface (no tab/newline), a non-empty tag list with positive counts, and no
duplicate tag or surface. Lexicon entries are normalized to the canonical
order (count descending, ties by tag name ascending) regardless of input
order.

### Model I/O

```rust
pub fn save_model(&self, path: &Path) -> Result<()>
pub fn save_model_to_writer<W: Write>(&self, writer: &mut W) -> Result<()>
pub async fn load_model(&mut self, uri: &str) -> Result<()>
pub fn load_model_from_path(&mut self, path: &Path) -> Result<()>
pub fn load_model_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()>
```

Same conventions as `AdaBoost`/`AveragedPerceptron`: `save_model`/`load_model`
work with file paths or (for `load_model`) `file://`/`http(s)://` URIs (the
latter requires the `remote_model` feature); the `*_to_writer`/`*_from_reader`
variants work with any writer/reader. Saving an empty learner (no lexicon
entries, or either embedded learner untrained) returns
`LitseaError::InvalidInput`. Loading validates the full file structure — see
[Model File Format](../advanced/model-file-format.md) for the on-disk layout
— and rejects malformed content with `LitseaError::InvalidData`; the learner
is left unmodified on a load error.

```rust
use std::path::Path;

use litsea::two_stage::TwoStageLearner;

let mut learner = TwoStageLearner::new();
learner.load_model_from_path(Path::new("./models/japanese_two_stage.model"))?;
```

### Accessors

```rust
pub fn stage1(&self) -> &AdaBoost
pub fn stage2(&self) -> &AveragedPerceptron
pub fn dominance(&self) -> f64
pub fn lexicon_len(&self) -> usize
pub fn lexicon_entry(&self, surface: &str) -> Option<&[(Upos, u32)]>
```

`dominance` is the classifier-skip threshold: at inference, a known surface
whose most frequent tag covers at least this fraction of its training
occurrences is tagged without invoking the stage-2 classifier at all.
`lexicon_entry` returns the candidate tags observed for a surface during
training, most-frequent-first, or `None` if the surface was never seen.

To actually run inference, install the learner on a `Segmenter` via
[`Segmenter::with_two_stage_learner`](segmenter.md#with_two_stage_learner)
rather than calling into `TwoStageLearner` directly — the segmenter compiles
it into packed scoring tables for fast lookup.

## `TwoStageFeatureSet`

```rust
#[non_exhaustive]
pub enum TwoStageFeatureSet {
    Full,
    Balanced,
    #[default]
    Fast,
}
```

Selects which of the 23 word-level stage-2 templates
(see [Feature Extraction](../algorithm/feature-extraction.md)) get written
by [`Extractor::extract_two_stage`](extractor.md#extract_two_stage). `Fast`
(the default) is the minimal measured set — surface, word length, first/last
char type, adjacent context char + type, 2-char prefix/suffix. `Balanced`
adds first/last char identity and the word type-code string. `Full` includes
every template. Segmentation quality is identical across all three sets (it
is decided entirely by stage 1); only tagging quality and throughput vary.
The relative ordering of the three sets (not their exact figures, which were
measured on an early prototype at a different epoch count than the bundled
models) is documented on this type's own rustdoc; see
[Pre-trained Models](../pre-trained-models.md) for the bundled models'
current, measured numbers.

Implements `FromStr` (case-insensitive: `"full"`, `"balanced"`, `"fast"`) and
`Display` (lowercase). Marked `#[non_exhaustive]` — external `match`
expressions need a wildcard arm.

## `AnyPosModel` and `ModelKind`

```rust
pub enum ModelKind {
    AdaBoost,
    AveragedPerceptron,
    TwoStage,
}

pub enum AnyPosModel {
    Joint(Box<AveragedPerceptron>),
    TwoStage(Box<TwoStageLearner>),
}
```

`ModelKind::detect(content: &str) -> ModelKind` inspects a model file's
first line to identify its format — a dispatch heuristic, not full
validation; the matching loader still validates the content.

`AnyPosModel::load(uri: &str) -> Result<Self>` is the single-fetch entry
point for code that accepts either a joint or a two-stage POS model (this is
what `litsea segment --pos` and `litsea evaluate --pos` use): it fetches the
model bytes once, detects the kind, and parses with the matching loader.
Returns `LitseaError::InvalidData` if the file is a plain AdaBoost
segmentation model (which cannot tag), if the bytes aren't valid UTF-8, or if
the detected format fails to parse.

`AnyPosModel::into_segmenter(self, language: Language) -> Segmenter` builds
the matching `Segmenter` — `with_pos_learner` for `Joint`,
`with_two_stage_learner` for `TwoStage` — so callers don't need to branch on
the model kind themselves:

```rust
use litsea::language::Language;
use litsea::two_stage::AnyPosModel;

let model = AnyPosModel::load("./models/japanese_two_stage.model").await?;
let segmenter = model.into_segmenter(Language::Japanese);
let tagged = segmenter.segment_with_pos("これはテストです。")?;
// Works identically whether the file was a joint or two-stage model.
```
