//! Text segmentation engine.
//!
//! Defines [`Segmenter`], which performs word segmentation (AdaBoost),
//! joint segmentation + POS tagging (Averaged Perceptron), and two-stage
//! segmentation + POS tagging (boundary classifier + lexicon + word-level
//! tagger, issue #147), scoring through the packed tables of
//! `crate::packed_model` / `crate::packed_pos_model` / `crate::packed_two_stage`.
//! Also hosts the corpus-processing pipeline used to build training features.

use std::collections::HashSet;
use std::sync::{PoisonError, RwLock};

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::language::{Language, OTHER_TYPE_ID};
use crate::packed_model::{
    PackedModel, SENTINEL_BASE, Slot, TAG_B, TAG_O, TAG_U, TEMPLATES, templates_for, wc_key,
};
use crate::packed_pos_model::PackedPosModel;
use crate::packed_two_stage::PackedTwoStageModel;
use crate::perceptron::AveragedPerceptron;
use crate::two_stage::TwoStageLearner;
use crate::upos::{SegmentLabel, Upos};

/// The stage-2 half of a decomposed [`TwoStageLearner`], retained alongside
/// the packed tagging tables built from it once, at construction time
/// (mirroring how the segmenter retains its other learners).
#[derive(Debug)]
struct TwoStageState {
    /// Stage-2 word-level tagger.
    stage2: AveragedPerceptron,
    /// Surface -> observed `(tag, count)` candidates, most frequent first.
    lexicon: rustc_hash::FxHashMap<String, Vec<(Upos, u32)>>,
    /// Classifier-skip dominance threshold.
    dominance: f64,
}

/// Text segmenter supporting three modes: word segmentation via AdaBoost
/// binary classification; joint word segmentation + POS tagging via an
/// Averaged Perceptron; and two-stage word segmentation + POS tagging via a
/// boundary classifier, lexicon, and word-level tagger (see
/// [`with_two_stage_learner`](Self::with_two_stage_learner)). Word
/// segmentation and POS tagging are dispatched through
/// [`segment`](Self::segment) and
/// [`segment_with_pos`](Self::segment_with_pos) respectively. Characters are
/// classified into language-specific type codes with direct `match`-based
/// rules ([`Language::char_type`]).
#[derive(Debug)]
pub struct Segmenter {
    language: Language,
    /// The AdaBoost learner (for a two-stage segmenter, this is the stage-1
    /// boundary classifier — a collapsed perceptron stored in AdaBoost
    /// format, not a boosted model). All mutation must flow through
    /// [`learner_mut`](Self::learner_mut) (as [`add_corpus`](Self::add_corpus)
    /// does) so that `packed` is invalidated alongside.
    learner: AdaBoost,
    /// Optional Averaged Perceptron for POS tagging
    pos_learner: Option<AveragedPerceptron>,
    /// The learner's weights compiled to packed integer keys for
    /// [`segment`](Self::segment)'s hot loop. `None` after a learner
    /// mutation; lazily rebuilt on the next `segment` call.
    packed: RwLock<Option<PackedModel>>,
    /// The POS learner's weights compiled to packed integer keys for
    /// [`segment_with_pos`](Self::segment_with_pos)'s hot loop. `None` when
    /// no POS learner is set or after a POS-learner mutation (all mutation
    /// must flow through [`pos_learner_mut`](Self::pos_learner_mut) or
    /// [`add_corpus_with_pos`](Self::add_corpus_with_pos), which invalidate
    /// it); lazily rebuilt on the next `segment_with_pos` call.
    packed_pos: RwLock<Option<PackedPosModel>>,
    /// The stage-2 state of a two-stage model, set by
    /// [`with_two_stage_learner`](Self::with_two_stage_learner) (the
    /// stage-1 half lives in `learner`). When present,
    /// [`segment_with_pos`](Self::segment_with_pos) takes the two-stage
    /// path instead of the joint POS path.
    two_stage: Option<TwoStageState>,
    /// The two-stage state compiled into packed tagging tables. `None`
    /// when no two-stage learner is set; built eagerly by
    /// [`with_two_stage_learner`](Self::with_two_stage_learner).
    packed_two_stage: RwLock<Option<PackedTwoStageModel>>,
}

impl Segmenter {
    /// Creates a new instance of [`Segmenter`] with a default (untrained)
    /// AdaBoost learner.
    ///
    /// The default learner has no trained weights, so [`segment`](Self::segment)
    /// returns one word per character until a model is loaded (via
    /// [`learner_mut`](Self::learner_mut)) or training data is added with
    /// [`add_corpus`](Self::add_corpus). To start from a trained learner, use
    /// [`with_learner`](Self::with_learner).
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    ///
    /// # Returns
    /// A new Segmenter instance with the specified language.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese);
    /// ```
    pub fn new(language: Language) -> Self {
        Self::with_learner(language, AdaBoost::default())
    }

    /// Creates a new instance of [`Segmenter`] with the given AdaBoost
    /// learner (typically one that has loaded a trained model).
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    /// * `learner` - The AdaBoost learner to segment with.
    ///
    /// # Returns
    /// A new Segmenter instance with the specified language and learner.
    pub fn with_learner(language: Language, learner: AdaBoost) -> Self {
        // Compile the packed scoring table eagerly so the common
        // load-then-segment path never rebuilds mid-stream.
        let packed = RwLock::new(Some(PackedModel::build(language, &learner)));
        Segmenter {
            language,
            learner,
            pos_learner: None,
            packed,
            packed_pos: RwLock::new(None),
            two_stage: None,
            packed_two_stage: RwLock::new(None),
        }
    }

    /// Creates a new instance of [`Segmenter`] with a POS learner.
    ///
    /// The AdaBoost learner is the untrained default: use
    /// [`segment_with_pos`](Self::segment_with_pos) with this constructor;
    /// [`segment`](Self::segment) would return one word per character.
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    /// * `pos_learner` - An AveragedPerceptron instance for POS tagging.
    ///
    /// # Returns
    /// A new Segmenter instance configured for joint segmentation + POS tagging.
    pub fn with_pos_learner(language: Language, pos_learner: AveragedPerceptron) -> Self {
        let learner = AdaBoost::default();
        // The default learner has no features; compiling it yields all-zero
        // tables (correctly sized for the language), so segment() stays
        // well-defined even though this constructor targets the POS path.
        let packed = RwLock::new(Some(PackedModel::build(language, &learner)));
        // Compile the packed POS scoring table eagerly so the common
        // load-then-segment path never rebuilds mid-stream.
        let packed_pos = RwLock::new(Some(PackedPosModel::build(language, &pos_learner)));
        Segmenter {
            language,
            learner,
            pos_learner: Some(pos_learner),
            packed,
            packed_pos,
            two_stage: None,
            packed_two_stage: RwLock::new(None),
        }
    }

    /// Creates a new instance of [`Segmenter`] with a two-stage model
    /// (issue #147): the learner's stage-1 boundary classifier becomes the
    /// segmenter's AdaBoost-path learner (so [`segment`](Self::segment)
    /// works naturally), and [`segment_with_pos`](Self::segment_with_pos)
    /// tags each segmented word through the lexicon (candidate
    /// restriction, dominance skip) and the stage-2 word-level tagger.
    ///
    /// # Arguments
    /// * `language` - The language to use for character type classification.
    /// * `learner` - The two-stage learner (typically one that has loaded
    ///   a `litsea-two-stage v1` model).
    ///
    /// # Returns
    /// A new Segmenter instance configured for two-stage segmentation +
    /// POS tagging.
    pub fn with_two_stage_learner(language: Language, learner: TwoStageLearner) -> Self {
        let (stage1, stage2, lexicon, dominance) = learner.into_parts();
        // Compile both packed tables eagerly so the common
        // load-then-segment path never rebuilds mid-stream.
        let packed = RwLock::new(Some(PackedModel::build(language, &stage1)));
        let packed_two_stage =
            RwLock::new(Some(PackedTwoStageModel::build(language, &stage2, &lexicon, dominance)));
        Segmenter {
            language,
            learner: stage1,
            pos_learner: None,
            packed,
            packed_pos: RwLock::new(None),
            two_stage: Some(TwoStageState {
                stage2,
                lexicon,
                dominance,
            }),
            packed_two_stage,
        }
    }

    /// Returns the language this segmenter was created for.
    #[must_use]
    pub fn language(&self) -> Language {
        self.language
    }

    /// Returns a reference to the AdaBoost learner used for segmentation
    /// (for a two-stage segmenter, this is the stage-1 boundary classifier).
    #[must_use]
    pub fn learner(&self) -> &AdaBoost {
        &self.learner
    }

    /// Returns a mutable reference to the AdaBoost learner used for
    /// segmentation (for a two-stage segmenter, this is the stage-1
    /// boundary classifier).
    ///
    /// The caller may mutate the learner (load a model, add instances,
    /// train), so the compiled packed scoring table is dropped here; the
    /// next [`segment`](Self::segment) call rebuilds it from the learner's
    /// then-current weights.
    pub fn learner_mut(&mut self) -> &mut AdaBoost {
        *self.packed.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        &mut self.learner
    }

    /// Returns a reference to the POS learner, if one is set (always `None`
    /// for a two-stage segmenter, even though it does perform POS tagging —
    /// see [`with_two_stage_learner`](Self::with_two_stage_learner)).
    #[must_use]
    pub fn pos_learner(&self) -> Option<&AveragedPerceptron> {
        self.pos_learner.as_ref()
    }

    /// Returns a mutable reference to the POS learner, if one is set.
    ///
    /// The caller may mutate the POS learner (load a model, add instances,
    /// train), so the compiled packed POS scoring table is dropped here; the
    /// next [`segment_with_pos`](Self::segment_with_pos) call rebuilds it
    /// from the learner's then-current weights.
    pub fn pos_learner_mut(&mut self) -> Option<&mut AveragedPerceptron> {
        *self.packed_pos.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        self.pos_learner.as_mut()
    }

    /// Gets the type of a character based on language-specific rules
    /// (delegates to [`Language::char_type`]).
    ///
    /// # Arguments
    /// * `c` - The character to classify.
    ///
    /// # Returns
    /// The language-specific type code of the character. Returns "O"
    /// (Other) if no rule matches.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese);
    /// let char_type = segmenter.char_type('あ');
    /// assert_eq!(char_type, "I"); // Hiragana
    /// ```
    #[must_use]
    pub fn char_type(&self, c: char) -> &'static str {
        self.language.char_type(c)
    }

    /// Builds the padded character and character-type arrays for a text.
    ///
    /// Returns `(chars, types)` where the first three entries are the B3/B2/B1
    /// head sentinels and the last three are the E1/E2/E3 tail sentinels, so
    /// real characters occupy indices `3..chars.len() - 3`. Characters borrow
    /// directly from `text` (no per-character allocation); the byte length is
    /// used as a capacity upper bound.
    fn sentence_context<'a>(&self, text: &'a str) -> (Vec<&'a str>, Vec<&'static str>) {
        let mut chars: Vec<&str> = Vec::with_capacity(text.len() + 6);
        let mut types: Vec<&'static str> = Vec::with_capacity(text.len() + 6);
        chars.extend_from_slice(&["B3", "B2", "B1"]);
        types.extend_from_slice(&["O"; 3]);
        for (i, ch) in text.char_indices() {
            types.push(self.language.char_type(ch));
            chars.push(&text[i..i + ch.len_utf8()]);
        }
        chars.extend_from_slice(&["E1", "E2", "E3"]);
        types.extend_from_slice(&["O", "O", "O"]);
        (chars, types)
    }

    /// Packed variant of [`sentence_context`](Self::sentence_context) for
    /// [`segment`](Self::segment)'s hot loop.
    ///
    /// Returns `(chars, char_codes, type_ids)` with the same layout (three
    /// head sentinels, real characters, three tail sentinels): `chars` holds
    /// string slices for word assembly, `char_codes` holds the numeric char
    /// codes used in packed keys (code points; sentinels map to
    /// `SENTINEL_BASE + k` in B3/B2/B1/E1/E2/E3 order), and `type_ids` holds
    /// [`Language::char_type_id`] values (padding uses the "O" id, conflated
    /// with a real Other-class character exactly as in the string
    /// representation).
    fn packed_context<'a>(&self, text: &'a str) -> (Vec<&'a str>, Vec<u32>, Vec<u8>) {
        let mut chars: Vec<&str> = Vec::with_capacity(text.len() + 6);
        let mut char_codes: Vec<u32> = Vec::with_capacity(text.len() + 6);
        let mut type_ids: Vec<u8> = Vec::with_capacity(text.len() + 6);
        chars.extend_from_slice(&["B3", "B2", "B1"]);
        char_codes.extend_from_slice(&[SENTINEL_BASE, SENTINEL_BASE + 1, SENTINEL_BASE + 2]);
        type_ids.extend_from_slice(&[OTHER_TYPE_ID; 3]);
        for (i, ch) in text.char_indices() {
            chars.push(&text[i..i + ch.len_utf8()]);
            char_codes.push(u32::from(ch));
            type_ids.push(self.language.char_type_id(ch));
        }
        chars.extend_from_slice(&["E1", "E2", "E3"]);
        char_codes.extend_from_slice(&[SENTINEL_BASE + 3, SENTINEL_BASE + 4, SENTINEL_BASE + 5]);
        type_ids.extend_from_slice(&[OTHER_TYPE_ID; 3]);
        (chars, char_codes, type_ids)
    }

    /// Runs `f` with the packed scoring table, rebuilding it first if a
    /// learner mutation invalidated it. The fast path takes only an
    /// uncontended read lock (one per sentence).
    fn with_packed<R>(&self, f: impl FnOnce(&PackedModel) -> R) -> R {
        {
            let guard = self.packed.read().unwrap_or_else(PoisonError::into_inner);
            if let Some(packed) = guard.as_ref() {
                return f(packed);
            }
        }
        let mut guard = self.packed.write().unwrap_or_else(PoisonError::into_inner);
        // get_or_insert_with covers the race where another thread rebuilt
        // the table between the two lock acquisitions.
        let packed = guard.get_or_insert_with(|| PackedModel::build(self.language, &self.learner));
        f(packed)
    }

    /// Runs `f` with the packed POS scoring table, rebuilding it first if a
    /// POS-learner mutation invalidated it. The fast path takes only an
    /// uncontended read lock (one per sentence). The caller passes the POS
    /// learner it already borrowed (presence is checked by
    /// [`segment_with_pos`](Self::segment_with_pos)).
    fn with_packed_pos<R>(
        &self,
        pos_learner: &AveragedPerceptron,
        f: impl FnOnce(&PackedPosModel) -> R,
    ) -> R {
        {
            let guard = self.packed_pos.read().unwrap_or_else(PoisonError::into_inner);
            if let Some(packed) = guard.as_ref() {
                return f(packed);
            }
        }
        let mut guard = self.packed_pos.write().unwrap_or_else(PoisonError::into_inner);
        // get_or_insert_with covers the race where another thread rebuilt
        // the table between the two lock acquisitions.
        let packed = guard.get_or_insert_with(|| PackedPosModel::build(self.language, pos_learner));
        f(packed)
    }

    /// Runs `f` with the packed two-stage tagging tables, building them on
    /// first use and caching them for the lifetime of the segmenter (there
    /// is no mutable accessor for the two-stage state, so unlike
    /// [`with_packed`](Self::with_packed) and
    /// [`with_packed_pos`](Self::with_packed_pos) this cache is never
    /// invalidated after it is first populated). The fast path takes only
    /// an uncontended read lock (one per sentence). The caller passes the
    /// two-stage state it already borrowed (presence is checked by
    /// [`segment_with_pos`](Self::segment_with_pos)).
    fn with_packed_two_stage<R>(
        &self,
        state: &TwoStageState,
        f: impl FnOnce(&PackedTwoStageModel) -> R,
    ) -> R {
        {
            let guard = self.packed_two_stage.read().unwrap_or_else(PoisonError::into_inner);
            if let Some(packed) = guard.as_ref() {
                return f(packed);
            }
        }
        let mut guard = self.packed_two_stage.write().unwrap_or_else(PoisonError::into_inner);
        // get_or_insert_with covers the race where another thread rebuilt
        // the table between the two lock acquisitions.
        let packed = guard.get_or_insert_with(|| {
            PackedTwoStageModel::build(
                self.language,
                &state.stage2,
                &state.lexicon,
                state.dominance,
            )
        });
        f(packed)
    }

    /// Shared corpus-processing pipeline behind `process_corpus` and
    /// `process_corpus_with_pos`.
    ///
    /// `tokens` yields `(word, label)` pairs where `label` is assigned to the
    /// first character of the word; continuation characters receive
    /// `cont_label`. Builds the padded context arrays and invokes `callback`
    /// with the feature set and label for each character position.
    ///
    /// When `include_first` is false the first character position is skipped:
    /// for the boundary (AdaBoost) pipeline its label is degenerate (always a
    /// word start). The POS pipeline passes true, because
    /// [`segment_with_pos`](Self::segment_with_pos) predicts at the first
    /// position to derive the first word's POS, so training must cover it
    /// (issue #100).
    fn process_tokens<'a, L, I, F>(
        &self,
        tokens: I,
        cont_label: L,
        include_first: bool,
        mut callback: F,
    ) where
        L: Clone,
        I: Iterator<Item = (&'a str, L)>,
        F: FnMut(HashSet<String>, L),
    {
        // Padding for lookback: tags[i-3], tags[i-2], tags[i-1] are referenced by
        // the attribute builder. The real characters' tags follow the padding.
        let mut tags: Vec<&'static str> = vec!["U"; 3];
        let mut labels: Vec<L> = Vec::new();
        let mut text = String::new();

        for (word, label) in tokens {
            let char_count = word.chars().count();
            if char_count == 0 {
                continue;
            }
            tags.push("B");
            labels.push(label);
            for _ in 1..char_count {
                tags.push("O");
                labels.push(cont_label.clone());
            }
            text.push_str(word);
        }

        if tags.len() < 4 {
            return;
        }
        // Override the first real character's tag to "U" (Unknown) instead of "B",
        // because there is no preceding word boundary decision to reference at position 0.
        tags[3] = "U";

        let (chars, types) = self.sentence_context(&text);

        let first = if include_first { 3 } else { 4 };
        for i in first..(chars.len() - 3) {
            let attrs = self.get_attributes(i, &tags, &chars, &types);
            callback(attrs, labels[i - 3].clone());
        }
    }

    /// Processes a corpus string ("word word ..."), calling the callback for
    /// each character position except the first with its attributes and
    /// boundary label (1 = word start, -1 = continuation).
    fn process_corpus<F>(&self, corpus: &str, callback: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_tokens(corpus.split(' ').map(|word| (word, 1i8)), -1i8, false, callback);
    }

    /// Processes a tab-separated corpus line ("word\tword\t..."), calling the
    /// callback for each character position except the first with its
    /// attributes and boundary label (1 = word start, -1 = continuation).
    ///
    /// Unlike [`process_corpus`](Self::process_corpus), a token may be a
    /// literal space `" "` (e.g. the inter-eojeol space in Korean), so the
    /// training text preserves the original spacing of the sentence. Empty
    /// tokens are ignored by the shared pipeline.
    fn process_corpus_tsv<F>(&self, corpus: &str, callback: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_tokens(corpus.split('\t').map(|word| (word, 1i8)), -1i8, false, callback);
    }

    /// Processes a POS-tagged corpus, yielding the attributes and
    /// SegmentLabel for every character position, including the first one
    /// (whose label carries the first word's POS; see
    /// [`segment_with_pos`](Self::segment_with_pos)).
    ///
    /// Corpus format: "word/POS word/POS ..."
    /// Example: "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT"
    fn process_corpus_with_pos<F>(&self, corpus: &str, callback: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        let tokens = corpus.split(' ').map(|token| {
            // Parse "word/POS" (no slash means the POS defaults to X)
            let (word, pos) = match token.rfind('/') {
                Some(idx) => (&token[..idx], token[idx + 1..].parse().unwrap_or(Upos::X)),
                None => (token, Upos::X),
            };
            (word, SegmentLabel::B(pos))
        });
        self.process_tokens(tokens, SegmentLabel::O, true, callback);
    }

    /// Adds a corpus to the segmenter with a custom writer function.
    ///
    /// # Arguments
    /// * `corpus` - A string slice representing the corpus to be added.
    /// * `writer` - A closure that takes a HashSet of attributes and a label (i8) and writes them.
    ///
    /// # Note
    /// The writer function is called once for each character position in the
    /// corpus except the first (whose boundary label is degenerate: it always
    /// starts a word), receiving the position's attribute set and boundary
    /// label (1 = word start, -1 = continuation). For example, "テスト です"
    /// has five characters, so the writer is invoked four times.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Japanese);
    /// segmenter.add_corpus_with_writer("テスト です", |attrs, label| {
    ///    println!("Attributes: {:?}, Label: {}", attrs, label);
    /// });
    /// ```
    ///
    /// This will process the corpus and call the writer function for each
    /// character position except the first, passing the attributes and label.
    pub fn add_corpus_with_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_corpus(corpus, writer);
    }

    /// Adds a corpus to the segmenter.
    ///
    /// # Arguments
    /// * `corpus` - A string slice representing the corpus to be added.
    ///
    /// This method processes the corpus, extracts features, and adds instances to the AdaBoost learner.
    /// If the corpus is empty, it does nothing.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let mut segmenter = Segmenter::new(Language::Japanese);
    /// segmenter.add_corpus("テスト です");
    /// ```
    /// This will process the corpus and add instances to the segmenter.
    pub fn add_corpus(&mut self, corpus: &str) {
        let mut instances = Vec::new();
        self.process_corpus(corpus, |attrs, label| {
            instances.push((attrs, label));
        });
        // Mutate through learner_mut() so the packed scoring table is
        // invalidated alongside the learner change.
        let learner = self.learner_mut();
        for (attrs, label) in instances {
            learner.add_instance(attrs, label);
        }
    }

    /// Adds a tab-separated corpus line to the segmenter with a custom
    /// writer function.
    ///
    /// Corpus format: tokens separated by tab characters. A token may be a
    /// literal space `" "` (e.g. the inter-eojeol space in Korean), which
    /// lets the training text preserve the original spacing so the model can
    /// learn from space characters as boundary context (issue #152).
    ///
    /// # Arguments
    /// * `corpus` - A tab-separated corpus line ("word\tword\t...").
    /// * `writer` - A closure that receives each character position's
    ///   attribute set and boundary label (1 = word start, -1 = continuation).
    ///
    /// # Note
    /// Like [`add_corpus_with_writer`](Self::add_corpus_with_writer), the
    /// writer is called once for each character position except the first.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let segmenter = Segmenter::new(Language::Korean);
    /// segmenter.add_corpus_tsv_with_writer("나는\t \t고양이", |attrs, label| {
    ///    println!("Attributes: {:?}, Label: {}", attrs, label);
    /// });
    /// ```
    pub fn add_corpus_tsv_with_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, i8),
    {
        self.process_corpus_tsv(corpus, writer);
    }

    /// Adds a tab-separated corpus line to the segmenter.
    ///
    /// Corpus format: tokens separated by tab characters; a token may be a
    /// literal space `" "` (see
    /// [`add_corpus_tsv_with_writer`](Self::add_corpus_tsv_with_writer)).
    ///
    /// # Arguments
    /// * `corpus` - A tab-separated corpus line ("word\tword\t...").
    ///
    /// This method processes the corpus, extracts features, and adds
    /// instances to the AdaBoost learner. If the corpus is empty, it does
    /// nothing.
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let mut segmenter = Segmenter::new(Language::Korean);
    /// segmenter.add_corpus_tsv("나는\t \t고양이");
    /// ```
    pub fn add_corpus_tsv(&mut self, corpus: &str) {
        let mut instances = Vec::new();
        self.process_corpus_tsv(corpus, |attrs, label| {
            instances.push((attrs, label));
        });
        // Mutate through learner_mut() so the packed scoring table is
        // invalidated alongside the learner change.
        let learner = self.learner_mut();
        for (attrs, label) in instances {
            learner.add_instance(attrs, label);
        }
    }

    /// Adds a POS-tagged corpus as Averaged Perceptron training data.
    ///
    /// # Arguments
    /// * `corpus` - A POS-tagged corpus ("word/POS word/POS ..." format)
    ///
    /// # Example
    /// ```
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let mut segmenter = Segmenter::new(Language::Japanese);
    /// segmenter.add_corpus_with_pos("これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT");
    /// ```
    pub fn add_corpus_with_pos(&mut self, corpus: &str) {
        let mut instances = Vec::new();
        self.process_corpus_with_pos(corpus, |attrs, label| {
            instances.push((attrs, label));
        });
        // Invalidate the packed POS scoring table alongside the learner
        // mutation (mirrors learner_mut() on the AdaBoost side).
        *self.packed_pos.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        let pos_learner = self.pos_learner.get_or_insert_with(AveragedPerceptron::new);
        for (attrs, label) in instances {
            pos_learner.add_instance(attrs, label.to_string());
        }
    }

    /// Processes a POS-tagged corpus's features with a custom writer.
    ///
    /// # Arguments
    /// * `corpus` - A POS-tagged corpus ("word/POS word/POS ..." format)
    /// * `writer` - A closure receiving the attribute set and SegmentLabel for each character position
    pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        self.process_corpus_with_pos(corpus, writer);
    }

    /// Segments a sentence into words.
    ///
    /// # Arguments
    /// * `sentence` - A string slice representing the sentence to be parsed.
    ///
    /// # Returns
    /// A vector of strings, where each string is a segmented word from the sentence.
    ///
    /// # Note
    /// The method scores each character position through the compiled
    /// `crate::packed_model::PackedModel` in two passes — a scatter-add
    /// static pass over the tag-independent features and a sequential pass
    /// over the 16 tag-dependent dense templates — deciding at each position
    /// whether it starts a new word. No attribute strings are constructed:
    /// the AdaBoost learner (for a two-stage segmenter, a collapsed
    /// perceptron stored in AdaBoost format) is only consulted for its bias
    /// term and to (re)build the packed table after a learner mutation.
    /// If the sentence is empty, it returns an empty vector.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    ///
    /// use litsea::adaboost::AdaBoost;
    /// use litsea::language::Language;
    /// use litsea::segmenter::Segmenter;
    ///
    /// let model_file =
    ///     PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join("RWCP.model");
    /// let mut learner = AdaBoost::new(0.01, 100);
    /// learner.load_model_from_path(&model_file).unwrap();
    ///
    /// let segmenter = Segmenter::with_learner(Language::Japanese, learner);
    /// let result = segmenter.segment("これはテストです。");
    /// assert_eq!(result, vec!["これ", "は", "テスト", "です", "。"]);
    /// ```
    /// This will segment the sentence into words and return them as a vector of strings.
    #[must_use]
    pub fn segment(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let (chars, char_codes, type_ids) = self.packed_context(sentence);

        // The bias is a sum over all model weights; compute it once per
        // sentence instead of once per character.
        let bias = self.learner.bias();

        self.with_packed(|packed| {
            let type_radix = self.language.type_codes().len();
            let n = chars.len();
            // Decision positions: lo..=hi (position 3 is the first real
            // character and always starts the first word).
            let lo = 4usize;
            let hi = n - 4;

            // ---- Static pass: everything that does not depend on boundary
            // tags is accumulated into a per-position buffer in one sweep.
            // The f64 accumulation order differs from the string-keyed
            // reference here (see the module docs of packed_model); output
            // equality is pinned empirically by the differential tests.
            let mut static_scores = vec![0.0f64; n];
            // Unigram families: the char/type at context position q feeds
            // template UW(k+1)/UC(k+1) at decision position i = q + 3 - k
            // (their slot delta k reads context index i - 3 + k). UW is one
            // merged probe; UC is a direct index into its scatter vector.
            for (q, code) in char_codes.iter().enumerate() {
                if let Some(v) = packed.uw.get(code) {
                    for (k, w) in v.iter().enumerate() {
                        let i = (q + 3).wrapping_sub(k);
                        if (lo..=hi).contains(&i) {
                            static_scores[i] += w;
                        }
                    }
                }
                for (k, w) in packed.uc[type_ids[q] as usize].iter().enumerate() {
                    let i = (q + 3).wrapping_sub(k);
                    if (lo..=hi).contains(&i) {
                        static_scores[i] += w;
                    }
                }
            }
            // Bigram families: the adjacent pair (q, q+1) feeds BW(k+1)/
            // BC(k+1) at i = q + 2 - k; the triple (q, q+1, q+2) feeds
            // TC(k+1) at i = q + 3 - k.
            for q in 0..n - 1 {
                let key = (u64::from(char_codes[q]) << 24) | u64::from(char_codes[q + 1]);
                if let Some(v) = packed.bw.get(&key) {
                    for (k, w) in v.iter().enumerate() {
                        let i = q + 2 - k;
                        if (lo..=hi).contains(&i) {
                            static_scores[i] += w;
                        }
                    }
                }
                let pair = type_ids[q] as usize * type_radix + type_ids[q + 1] as usize;
                for (k, w) in packed.bc[pair].iter().enumerate() {
                    let i = q + 2 - k;
                    if (lo..=hi).contains(&i) {
                        static_scores[i] += w;
                    }
                }
                if q + 2 < n {
                    let triple = pair * type_radix + type_ids[q + 2] as usize;
                    for (k, w) in packed.tc[triple].iter().enumerate() {
                        let i = (q + 3).wrapping_sub(k);
                        if (lo..=hi).contains(&i) {
                            static_scores[i] += w;
                        }
                    }
                }
            }
            // WC templates (Japanese/Chinese only): one merged-row probe per
            // character instead of four keyed probes per position (#157).
            // Char q feeds position q+1 as c[i-1] (WC1 with t[i], WC3 with
            // t[i-1] = t[q]) and position q as c[i] (WC2 with t[i-1], WC4
            // with t[i] = t[q]); rows are laid out [slot][type_id], with the
            // slot order pinned against TEMPLATES by a unit test.
            if templates_for(self.language).len() == TEMPLATES.len() && !packed.wc.is_empty() {
                let t = type_radix;
                for (q, code) in char_codes.iter().enumerate() {
                    let Some(row) = packed.wc.get(code) else { continue };
                    let i = q + 1;
                    if (lo..=hi).contains(&i) {
                        static_scores[i] +=
                            row[type_ids[i] as usize] + row[2 * t + type_ids[i - 1] as usize];
                    }
                    if (lo..=hi).contains(&q) {
                        static_scores[q] +=
                            row[t + type_ids[q - 1] as usize] + row[3 * t + type_ids[q] as usize];
                    }
                }
            }

            // ---- Sequential pass: only the 16 tag-dependent templates
            // (all dense loads, indexed directly with the mixed-radix
            // layout of Template::dense_index — pinned by a unit test)
            // plus the boundary decision remain.
            // Padding for lookback: tags[0..3] are fixed U (unknown), and
            // tags[3] is also U since there is no boundary decision before
            // the first character.
            let t = type_radix;
            let d = &packed.dense;
            let mut tags: Vec<u8> = Vec::with_capacity(n);
            tags.extend_from_slice(&[TAG_U; 4]);
            let mut result = Vec::new();
            let mut word = chars[3].to_string();
            for (i, ch) in chars.iter().enumerate().take(n - 3).skip(4) {
                let (p1, p2, p3) =
                    (tags[i - 3] as usize, tags[i - 2] as usize, tags[i - 1] as usize);
                let (c1, c2, c3, c4) = (
                    type_ids[i - 3] as usize,
                    type_ids[i - 2] as usize,
                    type_ids[i - 1] as usize,
                    type_ids[i] as usize,
                );
                let score = bias
                    + static_scores[i]
                    + d[0][p1]
                    + d[1][p2]
                    + d[2][p3]
                    + d[3][p1 * 3 + p2]
                    + d[4][p2 * 3 + p3]
                    + d[27][p1 * t + c1]
                    + d[28][p2 * t + c2]
                    + d[29][p3 * t + c3]
                    + d[30][(p2 * t + c2) * t + c3]
                    + d[31][(p2 * t + c3) * t + c4]
                    + d[32][(p3 * t + c2) * t + c3]
                    + d[33][(p3 * t + c3) * t + c4]
                    + d[34][((p2 * t + c1) * t + c2) * t + c3]
                    + d[35][((p2 * t + c2) * t + c3) * t + c4]
                    + d[36][((p3 * t + c1) * t + c2) * t + c3]
                    + d[37][((p3 * t + c2) * t + c3) * t + c4];
                if score >= 0.0 {
                    result.push(std::mem::take(&mut word));
                    tags.push(TAG_B);
                } else {
                    tags.push(TAG_O);
                }
                word.push_str(ch);
            }
            result.push(word);
            result
        })
    }

    /// Reference implementation of [`segment`](Self::segment) using the
    /// string-keyed lookup path (the pre-#136 hot loop). Kept test-only as
    /// the oracle for differential tests: `segment` must produce identical
    /// output for any model and input.
    #[cfg(test)]
    fn segment_reference(&self, sentence: &str) -> Vec<String> {
        if sentence.is_empty() {
            return Vec::new();
        }
        let learner = &self.learner;
        let (chars, types) = self.sentence_context(sentence);
        let mut tags: Vec<&'static str> = Vec::with_capacity(chars.len());
        tags.extend_from_slice(&["U"; 4]);
        let bias = learner.bias();

        let mut result = Vec::new();
        let mut word = chars[3].to_string();
        for (i, ch) in chars.iter().enumerate().take(chars.len() - 3).skip(4) {
            let mut score = bias;
            self.write_attributes(i, &tags, &chars, &types, &mut |attr| {
                score += learner.weight(attr);
            });
            if score >= 0.0 {
                result.push(std::mem::take(&mut word));
                tags.push("B");
            } else {
                tags.push("O");
            }
            word.push_str(ch);
        }
        result.push(word);
        result
    }

    /// Segments the sentence and tags each word with its POS, using either
    /// the joint per-character Averaged Perceptron path or, if a two-stage
    /// learner is set, the two-stage boundary+lexicon path — see the
    /// `# Two-stage mode` section below.
    ///
    /// In the joint path, the Averaged Perceptron (`pos_learner`) predicts
    /// the label (`B-<POS>` / `O`) at each character position and the
    /// method returns word/POS pairs. The first word's POS is derived from
    /// the predicted label at the first character position.
    ///
    /// Scoring runs against the compiled packed POS tables
    /// (`crate::packed_pos_model::PackedPosModel`, issue #143) in the same
    /// two passes as [`segment`](Self::segment): a static pass scatter-adds
    /// every tag-free feature into an `n x n_classes` score matrix, and a
    /// sequential pass adds the 16 tag-dependent dense rows and takes the
    /// argmax. As in `segment()`, the accumulation order differs from the
    /// string-keyed reference path (kept test-only as
    /// [`segment_with_pos_reference`](Self::segment_with_pos_reference));
    /// output equality is pinned empirically by the differential tests.
    ///
    /// # Arguments
    /// * `sentence` - The sentence to segment
    ///
    /// # Returns
    /// `Result<Vec<(String, Upos)>>` - Pairs of words and their POS tags.
    /// An empty sentence yields `Ok` with an empty vector.
    ///
    /// # Two-stage mode
    /// When the segmenter was built with
    /// [`with_two_stage_learner`](Self::with_two_stage_learner), this
    /// method instead runs the two-stage pipeline (issue #147): the
    /// sentence is segmented by the stage-1 boundary classifier (exactly
    /// as [`segment`](Self::segment)), then each word is tagged through
    /// the candidate-tag lexicon — single-candidate and dominant surfaces
    /// skip the classifier entirely — with the stage-2 word-level tagger
    /// deciding ambiguous surfaces (candidate-masked argmax) and unknown
    /// surfaces (full argmax).
    ///
    /// # Errors
    /// Returns [`LitseaError::PosLearnerNotSet`] if neither a POS learner
    /// nor a two-stage learner is set. Set one beforehand with
    /// [`with_pos_learner`](Self::with_pos_learner),
    /// [`with_two_stage_learner`](Self::with_two_stage_learner), or
    /// [`add_corpus_with_pos`](Self::add_corpus_with_pos).
    pub fn segment_with_pos(&self, sentence: &str) -> Result<Vec<(String, Upos)>> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(state) = self.two_stage.as_ref() {
            let words = self.segment(sentence);
            let tags =
                self.with_packed_two_stage(state, |packed| packed.tag_words(self.language, &words));
            return Ok(words.into_iter().zip(tags).collect());
        }
        let pos_learner = self.pos_learner.as_ref().ok_or(LitseaError::PosLearnerNotSet)?;
        let (chars, char_codes, type_ids) = self.packed_context(sentence);

        let result = self.with_packed_pos(pos_learner, |packed| {
            let cn = packed.n_classes;
            // A perceptron without classes yields no prediction: every
            // position falls back to O, so the whole sentence is one word
            // tagged X, matching the reference path.
            if cn == 0 {
                return vec![(sentence.to_string(), Upos::X)];
            }
            let type_radix = self.language.type_codes().len();
            let n = chars.len();
            // Decision positions: lo..=hi. Unlike segment() (lo = 4), the
            // POS path also predicts at the first real character (i = 3) to
            // derive the first word's POS.
            let lo = 3usize;
            let hi = n - 4;

            // ---- Static pass: everything that does not depend on boundary
            // tags is accumulated into a per-position score row in one
            // sweep, exactly as in segment() but with n_classes-wide rows.
            // The f64 accumulation order differs from the string-keyed
            // reference (see the module docs of packed_pos_model); output
            // equality is pinned empirically by the differential tests.
            let mut static_scores = vec![0.0f64; n * cn];
            // Unigram families: one merged probe (UW) and one scatter-twin
            // block (UC) per context position q feed decision positions
            // i = q + 3 - k.
            for (q, code) in char_codes.iter().enumerate() {
                if let Some(row) = packed.uw.get(code) {
                    for &(k, c, w) in row.iter() {
                        let i = (q + 3).wrapping_sub(k as usize);
                        if (lo..=hi).contains(&i) {
                            static_scores[i * cn + c as usize] += w;
                        }
                    }
                }
                let block = &packed.uc[type_ids[q] as usize * 6 * cn..][..6 * cn];
                for k in 0..6 {
                    let i = (q + 3).wrapping_sub(k);
                    if (lo..=hi).contains(&i) {
                        let dst = &mut static_scores[i * cn..][..cn];
                        for (s, w) in dst.iter_mut().zip(&block[k * cn..][..cn]) {
                            *s += *w;
                        }
                    }
                }
            }
            // Bigram families: the adjacent pair (q, q+1) feeds BW/BC at
            // i = q + 2 - k; the triple (q, q+1, q+2) feeds TC at
            // i = q + 3 - k.
            for q in 0..n - 1 {
                let key = (u64::from(char_codes[q]) << 24) | u64::from(char_codes[q + 1]);
                if let Some(row) = packed.bw.get(&key) {
                    for &(k, c, w) in row.iter() {
                        let i = q + 2 - k as usize;
                        if (lo..=hi).contains(&i) {
                            static_scores[i * cn + c as usize] += w;
                        }
                    }
                }
                let pair = type_ids[q] as usize * type_radix + type_ids[q + 1] as usize;
                let block = &packed.bc[pair * 3 * cn..][..3 * cn];
                for k in 0..3 {
                    let i = q + 2 - k;
                    if (lo..=hi).contains(&i) {
                        let dst = &mut static_scores[i * cn..][..cn];
                        for (s, w) in dst.iter_mut().zip(&block[k * cn..][..cn]) {
                            *s += *w;
                        }
                    }
                }
                if q + 2 < n {
                    let triple = pair * type_radix + type_ids[q + 2] as usize;
                    let block = &packed.tc[triple * 4 * cn..][..4 * cn];
                    for k in 0..4 {
                        let i = (q + 3).wrapping_sub(k);
                        if (lo..=hi).contains(&i) {
                            let dst = &mut static_scores[i * cn..][..cn];
                            for (s, w) in dst.iter_mut().zip(&block[k * cn..][..cn]) {
                                *s += *w;
                            }
                        }
                    }
                }
            }
            // WC probes (Japanese/Chinese only), gathered per position with
            // the slot layout of WC1..WC4: (w3,c4), (c3,w4), (w3,c3),
            // (w4,c4) — pinned against TEMPLATES by a unit test.
            if templates_for(self.language).len() == TEMPLATES.len() && !packed.wc.is_empty() {
                for i in lo..=hi {
                    let mut probe = |idx: usize, chr: u32, typ: u8| {
                        if let Some(row) = packed.wc.get(&wc_key(idx, chr, typ)) {
                            let dst = &mut static_scores[i * cn..][..cn];
                            for &(c, w) in row.iter() {
                                dst[c as usize] += w;
                            }
                        }
                    };
                    probe(0, char_codes[i - 1], type_ids[i]);
                    probe(1, char_codes[i], type_ids[i - 1]);
                    probe(2, char_codes[i - 1], type_ids[i - 1]);
                    probe(3, char_codes[i], type_ids[i]);
                }
            }

            // ---- Sequential pass: the 16 tag-dependent dense rows plus
            // the argmax remain per position (predict_seq).
            // Padding for lookback: tags[0..3] are fixed U (unknown), and
            // tags[3] is also U since there is no boundary decision before
            // the first character.
            let mut tags: Vec<u8> = Vec::with_capacity(n);
            tags.extend_from_slice(&[TAG_U; 4]);
            // Score row reused across positions to amortize allocations.
            let mut scores = vec![0.0f64; cn];
            let predict = |i: usize, tags: &[u8], scores: &mut [f64]| -> usize {
                packed.predict_seq(
                    &static_scores[i * cn..][..cn],
                    (tags[i - 3] as usize, tags[i - 2] as usize, tags[i - 1] as usize),
                    (
                        type_ids[i - 3] as usize,
                        type_ids[i - 2] as usize,
                        type_ids[i - 1] as usize,
                        type_ids[i] as usize,
                    ),
                    type_radix,
                    scores,
                )
            };

            // The first character always starts the first word; its
            // predicted label is used only to determine the first word's
            // POS.
            let first_idx = predict(3, &tags, &mut scores);
            let mut current_pos = packed.label(first_idx).pos().unwrap_or(Upos::X);

            // Words are contiguous runs of the sentence, so they are
            // materialized from byte offsets in one exact-size allocation
            // each instead of being grown character by character.
            let mut result: Vec<(String, Upos)> = Vec::new();
            let mut word_start = 0usize;
            let mut byte_pos = chars[3].len();

            for (i, ch) in chars.iter().enumerate().take(n - 3).skip(4) {
                let label = packed.label(predict(i, &tags, &mut scores));
                if label.is_boundary() {
                    // Finalize the current word and push it to the result
                    result.push((sentence[word_start..byte_pos].to_string(), current_pos));
                    word_start = byte_pos;
                    current_pos = label.pos().unwrap_or(Upos::X);
                    tags.push(TAG_B);
                } else {
                    tags.push(TAG_O);
                }
                byte_pos += ch.len();
            }

            result.push((sentence[word_start..].to_string(), current_pos));
            result
        });
        Ok(result)
    }

    /// Reference implementation of
    /// [`segment_with_pos`](Self::segment_with_pos) using the string-keyed
    /// lookup path (the pre-#143 hot loop). Kept test-only as the oracle for
    /// differential tests: `segment_with_pos` must produce identical output
    /// for any model and input.
    #[cfg(test)]
    fn segment_with_pos_reference(&self, sentence: &str) -> Result<Vec<(String, Upos)>> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let pos_learner = self.pos_learner.as_ref().ok_or(LitseaError::PosLearnerNotSet)?;

        let (chars, types) = self.sentence_context(sentence);
        let mut tags: Vec<&'static str> = Vec::with_capacity(chars.len());
        tags.extend_from_slice(&["U"; 4]);
        // Attribute and score buffers reused across positions to amortize
        // allocations.
        let mut attrs_buf: Vec<String> = Vec::new();
        let mut scores_buf: Vec<f64> = Vec::new();

        // The first character always starts the first word; its predicted
        // label is used only to determine the first word's POS.
        self.collect_attributes(3, &tags, &chars, &types, &mut attrs_buf);
        let first_label: SegmentLabel = pos_learner
            .predict_slice(&attrs_buf, &mut scores_buf)
            .parse()
            .unwrap_or(SegmentLabel::O);
        let mut current_pos = first_label.pos().unwrap_or(Upos::X);

        let mut result: Vec<(String, Upos)> = Vec::new();
        let mut word = chars[3].to_string();

        for (i, ch) in chars.iter().enumerate().take(chars.len() - 3).skip(4) {
            self.collect_attributes(i, &tags, &chars, &types, &mut attrs_buf);
            let label: SegmentLabel = pos_learner
                .predict_slice(&attrs_buf, &mut scores_buf)
                .parse()
                .unwrap_or(SegmentLabel::O);
            if label.is_boundary() {
                // Finalize the current word and push it to the result
                result.push((std::mem::take(&mut word), current_pos));
                current_pos = label.pos().unwrap_or(Upos::X);
                tags.push("B");
            } else {
                tags.push("O");
            }
            word.push_str(ch);
        }

        result.push((word, current_pos));
        Ok(result)
    }

    /// Builds the attribute set for a specific index (used by the corpus
    /// processing pipeline, where the public callbacks expect a `HashSet`).
    fn get_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[&str],
        types: &[&'static str],
    ) -> HashSet<String> {
        let mut attrs = HashSet::with_capacity(48);
        self.write_attributes(i, tags, chars, types, &mut |attr| {
            attrs.insert(attr.to_string());
        });
        attrs
    }

    /// Collects the attributes for a position into a reusable `Vec<String>`,
    /// reusing the existing string allocations where possible.
    ///
    /// Test-only since the packed POS scorer (issue #143): kept as part of
    /// the string-keyed reference path
    /// ([`segment_with_pos_reference`](Self::segment_with_pos_reference)).
    #[cfg(test)]
    fn collect_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[&str],
        types: &[&'static str],
        out: &mut Vec<String>,
    ) {
        let mut idx = 0;
        self.write_attributes(i, tags, chars, types, &mut |attr| {
            if idx < out.len() {
                out[idx].clear();
                out[idx].push_str(attr);
            } else {
                out.push(attr.to_string());
            }
            idx += 1;
        });
        out.truncate(idx);
    }

    /// Writes each attribute for position `i` into a reusable buffer and
    /// passes it to `sink`. The feature template itself is defined once in
    /// [`crate::packed_model::TEMPLATES`]; this renders each template as
    /// `{prefix}:{slot values}` via plain `push_str` (no `core::fmt`), in
    /// table order. The language-specific `WC*` templates are included for
    /// Japanese and Chinese only (Korean's uniform syllable types would make
    /// them noise), via [`crate::packed_model::templates_for`].
    ///
    /// # Panics
    /// Panics if `i` is less than 3 or if `i + 2` exceeds the length of
    /// `chars` or `types`. Callers must ensure that `i` is within the valid
    /// range `[3, chars.len() - 3)`.
    fn write_attributes(
        &self,
        i: usize,
        tags: &[&'static str],
        chars: &[&str],
        types: &[&'static str],
        sink: &mut dyn FnMut(&str),
    ) {
        let mut buf = String::with_capacity(32);
        for template in templates_for(self.language) {
            buf.clear();
            buf.push_str(template.prefix);
            buf.push(':');
            for slot in template.slots {
                buf.push_str(match *slot {
                    Slot::Tag(d) => tags[i - 3 + d as usize],
                    Slot::Chr(d) => chars[i - 3 + d as usize],
                    Slot::Typ(d) => types[i - 3 + d as usize],
                });
            }
            sink(&buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_char_type_japanese() {
        let segmenter = Segmenter::new(Language::Japanese);

        assert_eq!(segmenter.char_type('あ'), "I"); // Hiragana
        assert_eq!(segmenter.char_type('漢'), "H"); // Kanji
        assert_eq!(segmenter.char_type('。'), "P"); // Punctuation
        assert_eq!(segmenter.char_type('A'), "A"); // Latin
        assert_eq!(segmenter.char_type('1'), "N"); // Digit
        assert_eq!(segmenter.char_type('@'), "O"); // Not matching any pattern
    }

    #[test]
    fn test_char_type_chinese() {
        let segmenter = Segmenter::new(Language::Chinese);

        assert_eq!(segmenter.char_type('的'), "F"); // Function word
        assert_eq!(segmenter.char_type('中'), "C"); // CJK Unified
        assert_eq!(segmenter.char_type('国'), "C"); // CJK Unified
        assert_eq!(segmenter.char_type('。'), "P"); // Punctuation
        assert_eq!(segmenter.char_type('A'), "A"); // Latin
        assert_eq!(segmenter.char_type('5'), "N"); // Digit
        assert_eq!(segmenter.char_type('@'), "O"); // Other
    }

    #[test]
    fn test_char_type_korean() {
        let segmenter = Segmenter::new(Language::Korean);

        assert_eq!(segmenter.char_type('는'), "E"); // Particle (topic marker)
        assert_eq!(segmenter.char_type('가'), "SN"); // Hangul Syllable without 받침
        assert_eq!(segmenter.char_type('한'), "SF"); // Hangul Syllable with 받침
        assert_eq!(segmenter.char_type('ㄱ'), "G"); // Compatibility Jamo
        assert_eq!(segmenter.char_type('漢'), "H"); // Hanja
        assert_eq!(segmenter.char_type('A'), "A"); // Latin
        assert_eq!(segmenter.char_type('5'), "N"); // Digit
        assert_eq!(segmenter.char_type('@'), "O"); // Other
    }

    #[test]
    fn test_add_corpus_with_writer() {
        let segmenter = Segmenter::new(Language::Japanese);
        let sentence = "テスト です";
        let mut collected = Vec::new();

        segmenter.add_corpus_with_writer(sentence, |attrs, label| {
            collected.push((attrs, label));
        });

        // "テスト です" has 5 characters; the callback loop runs for indices 4..8
        // (skipping the first character at index 3), producing 4 instances.
        assert_eq!(collected.len(), 4);

        // Exactly one word boundary (at "で", start of second word "です")
        let positive_count = collected.iter().filter(|(_, label)| *label == 1).count();
        let negative_count = collected.iter().filter(|(_, label)| *label == -1).count();
        assert_eq!(positive_count, 1);
        assert_eq!(negative_count, 3);

        // Check that attributes contain expected keys
        let (attrs, _) = &collected[0];
        assert!(attrs.iter().any(|a| a.starts_with("UW")));
        assert!(attrs.iter().any(|a| a.starts_with("UC")));
    }

    #[test]
    fn test_add_corpus_tsv_with_writer() {
        let segmenter = Segmenter::new(Language::Korean);
        // Space-preserving TSV corpus: 나는 + inter-eojeol space + 봄 + period.
        // Training text is "나는 봄." (5 chars); the callback skips the first
        // character, producing 4 instances: 는, ' ', 봄, '.'.
        let sentence = "나는\t \t봄\t.";
        let mut collected = Vec::new();

        segmenter.add_corpus_tsv_with_writer(sentence, |attrs, label| {
            collected.push((attrs, label));
        });

        assert_eq!(collected.len(), 4);

        // Per-position boundary labels: 는 is a continuation of 나는; the
        // space, 봄, and '.' each start a token.
        let labels: Vec<i8> = collected.iter().map(|(_, label)| *label).collect();
        assert_eq!(labels, vec![-1, 1, 1, 1]);

        // The space character must appear inside UW context features, proving
        // the training text preserved it.
        let has_space_feature = collected
            .iter()
            .any(|(attrs, _)| attrs.iter().any(|a| a.starts_with("UW") && a.ends_with(' ')));
        assert!(has_space_feature, "expected a UW feature whose value is the space character");
    }

    #[test]
    fn test_add_corpus_tsv_ignores_empty_tokens() {
        let segmenter = Segmenter::new(Language::Korean);
        let mut with_empty = Vec::new();
        segmenter.add_corpus_tsv_with_writer("나는\t\t봄", |attrs, label| {
            with_empty.push((attrs, label));
        });
        let mut without_empty = Vec::new();
        segmenter.add_corpus_tsv_with_writer("나는\t봄", |attrs, label| {
            without_empty.push((attrs, label));
        });
        // The empty token contributes no characters, so both corpora are
        // identical to the pipeline.
        assert_eq!(with_empty, without_empty);
    }

    #[test]
    fn test_add_corpus() {
        let mut segmenter = Segmenter::new(Language::Japanese);
        let sentence = "テスト です";
        segmenter.add_corpus(sentence);
        // Should not panic or add anything, just a smoke test
    }

    #[test]
    fn test_segment() {
        let sentence = "これはテストです。";

        let model_file =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join("RWCP.model");
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_path(&model_file).unwrap();

        let segmenter = Segmenter::with_learner(Language::Japanese, learner);

        let result = segmenter.segment(sentence);

        assert!(!result.is_empty());
        // "これはテストです。" segments into: "これ", "は", "テスト", "です", "。"
        // The RWCP model predicts word boundaries after these positions.
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], "これ");
        assert_eq!(result[1], "は");
        assert_eq!(result[2], "テスト");
        assert_eq!(result[3], "です");
        assert_eq!(result[4], "。");
    }

    #[test]
    fn test_add_sentence_empty() {
        let mut segmenter = Segmenter::new(Language::Japanese);
        segmenter.add_corpus("");
        // Should not panic or add anything
    }

    #[test]
    fn test_segment_empty_sentence() {
        let segmenter = Segmenter::new(Language::Japanese);
        let result = segmenter.segment("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_attributes() {
        let segmenter = Segmenter::new(Language::Japanese);

        let tags: Vec<&'static str> = vec!["U"; 7];

        let chars = vec![
            "B3", // index 0
            "B2", // index 1
            "B1", // index 2
            "あ", // index 3
            "い", // index 4
            "う", // index 5
            "E1", // index 6
        ];

        let types: Vec<&'static str> = vec!["O", "O", "O", "O", "I", "I", "O"];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:い"));
        assert!(attrs.contains("UC4:I"));
        assert!(attrs.contains("UP3:U"));
        // Language-specific WC features (Japanese includes them)
        assert!(attrs.contains("WC1:あI")); // w3 + c4
        assert!(attrs.contains("WC2:Oい")); // c3 + w4
        assert!(attrs.contains("WC3:あO")); // w3 + c3
        assert!(attrs.contains("WC4:いI")); // w4 + c4
        // 38 base features (UP/BP/UW/BW/UC/BC/TC/UQ/BQ/TQ) + 4 WC features (Japanese-specific)
        assert_eq!(attrs.len(), 42);
    }

    #[test]
    fn test_write_attributes_exact_strings_japanese() {
        // Pin test for issue #136: the exact attribute strings AND their
        // emission order are load-bearing. The strings are the model's feature
        // keys, and segment() sums f64 weights in emission order (float
        // addition is not associative), so neither may change. Every slot in
        // the fixture holds a distinct value so that any slot-mapping mistake
        // changes the output.
        let segmenter = Segmenter::new(Language::Japanese);
        let tags: Vec<&'static str> = vec!["U", "B", "O", "U", "B", "O", "U"];
        let chars = vec!["B3", "B2", "B1", "あ", "い", "う", "E1"];
        let types: Vec<&'static str> = vec!["O", "P", "A", "N", "I", "H", "K"];

        let mut attrs: Vec<String> = Vec::new();
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut attrs);

        let expected = [
            "UP1:B",
            "UP2:O",
            "UP3:U",
            "BP1:BO",
            "BP2:OU",
            "UW1:B2",
            "UW2:B1",
            "UW3:あ",
            "UW4:い",
            "UW5:う",
            "UW6:E1",
            "BW1:B1あ",
            "BW2:あい",
            "BW3:いう",
            "UC1:P",
            "UC2:A",
            "UC3:N",
            "UC4:I",
            "UC5:H",
            "UC6:K",
            "BC1:AN",
            "BC2:NI",
            "BC3:IH",
            "TC1:PAN",
            "TC2:ANI",
            "TC3:NIH",
            "TC4:IHK",
            "UQ1:BP",
            "UQ2:OA",
            "UQ3:UN",
            "BQ1:OAN",
            "BQ2:ONI",
            "BQ3:UAN",
            "BQ4:UNI",
            "TQ1:OPAN",
            "TQ2:OANI",
            "TQ3:UPAN",
            "TQ4:UANI",
            "WC1:あI",
            "WC2:Nい",
            "WC3:あN",
            "WC4:いI",
        ];
        assert_eq!(attrs, expected);
    }

    #[test]
    fn test_write_attributes_exact_strings_korean() {
        // Korean counterpart of the pin test above: 38 features (no WC*), and
        // the two-character type codes SN/SF concatenate without separators.
        let segmenter = Segmenter::new(Language::Korean);
        let tags: Vec<&'static str> = vec!["U", "B", "O", "U", "B", "O", "U"];
        let chars = vec!["B3", "B2", "B1", "한", "국", "어", "E1"];
        let types: Vec<&'static str> = vec!["O", "E", "SN", "SF", "J", "G", "H"];

        let mut attrs: Vec<String> = Vec::new();
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut attrs);

        let expected = [
            "UP1:B",
            "UP2:O",
            "UP3:U",
            "BP1:BO",
            "BP2:OU",
            "UW1:B2",
            "UW2:B1",
            "UW3:한",
            "UW4:국",
            "UW5:어",
            "UW6:E1",
            "BW1:B1한",
            "BW2:한국",
            "BW3:국어",
            "UC1:E",
            "UC2:SN",
            "UC3:SF",
            "UC4:J",
            "UC5:G",
            "UC6:H",
            "BC1:SNSF",
            "BC2:SFJ",
            "BC3:JG",
            "TC1:ESNSF",
            "TC2:SNSFJ",
            "TC3:SFJG",
            "TC4:JGH",
            "UQ1:BE",
            "UQ2:OSN",
            "UQ3:USF",
            "BQ1:OSNSF",
            "BQ2:OSFJ",
            "BQ3:USNSF",
            "BQ4:USFJ",
            "TQ1:OESNSF",
            "TQ2:OSNSFJ",
            "TQ3:UESNSF",
            "TQ4:USNSFJ",
        ];
        assert_eq!(attrs, expected);
    }

    #[test]
    fn test_collect_attributes_reuses_buffer() {
        let segmenter = Segmenter::new(Language::Japanese);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec!["B3", "B2", "B1", "あ", "い", "う", "E1"];
        let types: Vec<&'static str> = vec!["O", "O", "O", "O", "I", "I", "O"];

        let mut buf: Vec<String> = Vec::new();
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut buf);
        assert_eq!(buf.len(), 42);
        // The slice contents must match the HashSet variant.
        let set = segmenter.get_attributes(4, &tags, &chars, &types);
        for attr in &buf {
            assert!(set.contains(attr), "missing from set: {}", attr);
        }
        // A second collection reuses the buffer and yields the same result.
        segmenter.collect_attributes(4, &tags, &chars, &types, &mut buf);
        assert_eq!(buf.len(), 42);
    }

    #[test]
    #[should_panic]
    fn test_get_attributes_panics_index_too_low() {
        let segmenter = Segmenter::new(Language::Japanese);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec!["B3", "B2", "B1", "あ", "い", "う", "E1"];
        let types: Vec<&'static str> = vec!["O"; 7];
        // i=2 is out of valid range [3, chars.len()-3); should panic on chars[i-3]
        let _ = segmenter.get_attributes(2, &tags, &chars, &types);
    }

    #[test]
    #[should_panic]
    fn test_get_attributes_panics_index_too_high() {
        let segmenter = Segmenter::new(Language::Japanese);
        let tags: Vec<&'static str> = vec!["U"; 7];
        let chars = vec!["B3", "B2", "B1", "あ", "い", "う", "E1"];
        let types: Vec<&'static str> = vec!["O"; 7];
        // i=5 means i+2=7 which exceeds chars.len()=7; should panic on chars[i+2]
        let _ = segmenter.get_attributes(5, &tags, &chars, &types);
    }

    #[test]
    fn test_get_attributes_korean() {
        let segmenter = Segmenter::new(Language::Korean);

        let tags: Vec<&'static str> = vec!["U"; 7];

        let chars = vec![
            "B3", // index 0
            "B2", // index 1
            "B1", // index 2
            "한", // index 3
            "국", // index 4
            "어", // index 5
            "E1", // index 6
        ];

        let types: Vec<&'static str> = vec!["O", "O", "O", "SF", "SF", "SN", "O"];

        let attrs = segmenter.get_attributes(4, &tags, &chars, &types);
        assert!(attrs.contains("UW4:국"));
        assert!(attrs.contains("UC4:SF"));
        // Korean does NOT include WC features
        assert!(!attrs.contains("WC1:한SF"));
        assert!(!attrs.contains("WC2:SF국"));
        // 38 base features only (Korean does not include WC word-character features)
        assert_eq!(attrs.len(), 38);
    }

    // --- POS tagging tests ---

    #[test]
    fn test_add_corpus_with_pos() {
        let mut segmenter = Segmenter::new(Language::Japanese);
        segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
        // pos_learner is initialized
        assert!(segmenter.pos_learner.is_some());
    }

    #[test]
    fn test_add_corpus_with_pos_writer() {
        let segmenter = Segmenter::new(Language::Japanese);
        let corpus = "テスト/NOUN です/AUX";
        let mut collected = Vec::new();

        segmenter.add_corpus_with_pos_writer(corpus, |attrs, label| {
            collected.push((attrs, label));
        });

        // "テストです" has 5 characters; the POS pipeline emits every
        // position including the first one (i=3..8), producing 5 instances:
        // テ=B-NOUN, ス=O, ト=O, で=B-AUX, す=O.
        assert_eq!(collected.len(), 5);
        assert_eq!(collected[0].1, SegmentLabel::B(Upos::NOUN));

        let boundary_count = collected.iter().filter(|(_, l)| l.is_boundary()).count();
        assert_eq!(boundary_count, 2); // B-NOUN at "テ" and B-AUX at "で"

        // Check the B-AUX label
        let b_aux = collected.iter().find(|(_, l)| *l == SegmentLabel::B(Upos::AUX));
        assert!(b_aux.is_some());
    }

    #[test]
    fn test_pos_writer_emits_first_position() {
        // Regression test for #100: the POS pipeline must emit the first
        // character position (its label carries the first word's POS, which
        // segment_with_pos predicts at inference), while the AdaBoost
        // boundary pipeline keeps skipping it (the boundary label at the
        // first position is degenerate — always a word start).
        let segmenter = Segmenter::new(Language::Japanese);

        let mut pos_labels = Vec::new();
        segmenter.add_corpus_with_pos_writer("テスト/NOUN です/AUX", |_, label| {
            pos_labels.push(label);
        });
        assert_eq!(pos_labels.len(), 5);
        assert_eq!(pos_labels[0], SegmentLabel::B(Upos::NOUN));

        let mut boundary_instances = 0;
        segmenter.add_corpus_with_writer("テスト です", |_, _| {
            boundary_instances += 1;
        });
        assert_eq!(boundary_instances, 4); // AdaBoost path still starts at i=4
    }

    #[test]
    fn test_segment_with_pos() {
        let mut segmenter = Segmenter::new(Language::Japanese);

        // Add the training data multiple times and train
        for _ in 0..20 {
            segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
            segmenter.add_corpus_with_pos("私/PRON の/PART 猫/NOUN は/PART 可愛い/ADJ 。/PUNCT");
        }

        // Train the perceptron
        let running = AtomicBool::new(true);
        segmenter.pos_learner.as_mut().unwrap().train(10, &running);

        // Segmentation + POS tagging
        let result = segmenter.segment_with_pos("これはテストです。").unwrap();
        assert!(!result.is_empty());

        // Verify the result is (word, POS) pairs
        for (word, pos) in &result {
            assert!(!word.is_empty());
            // The POS is one of the Upos variants
            let _ = pos.to_string();
        }
    }

    // --- Packed scoring path tests (#136) ---

    /// Sentences whose packed keys must round-trip: golden-style inputs plus
    /// sentinel-lookalike stress strings (real "B1"/"E2" substrings, mixed
    /// scripts, digits, ASCII).
    const STRESS_SENTENCES: [&str; 8] = [
        "B1テスト",
        "テB1あE2ト",
        "E1E2E3",
        "B1B2B3E1E2E3",
        "abc B1 123",
        "UOBそのタグ文字",
        "SNSF가나한글",
        "漢字とtestと123。",
    ];

    fn load_adaboost(model: &str) -> AdaBoost {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model);
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_path(&path).unwrap();
        learner
    }

    fn assert_segment_matches_reference(segmenter: &Segmenter, sentences: &[&str]) {
        for sentence in sentences {
            assert_eq!(
                segmenter.segment(sentence),
                segmenter.segment_reference(sentence),
                "packed segment() diverged from the string-keyed reference for {sentence:?}"
            );
        }
    }

    #[test]
    fn test_segment_differential_japanese_models() {
        let sentences = [
            "これはテストです。",
            "私の猫は可愛い。",
            "東京都に住んでいます。",
            "字",
            "こんにちは",
            "価格は1000円です。",
            "RustでNLPを実装する。",
        ];
        for model in ["japanese.model", "RWCP.model", "JEITA_Genpaku_ChaSen_IPAdic.model"] {
            let segmenter = Segmenter::with_learner(Language::Japanese, load_adaboost(model));
            assert_segment_matches_reference(&segmenter, &sentences);
            assert_segment_matches_reference(&segmenter, &STRESS_SENTENCES);
        }
    }

    #[test]
    fn test_segment_differential_chinese_model() {
        let sentences =
            ["这是一个测试。", "我喜欢吃中国菜。", "他在北京工作。", "好", "2024年的春天。"];
        let segmenter = Segmenter::with_learner(Language::Chinese, load_adaboost("chinese.model"));
        assert_segment_matches_reference(&segmenter, &sentences);
        assert_segment_matches_reference(&segmenter, &STRESS_SENTENCES);
    }

    #[test]
    fn test_segment_differential_korean_model() {
        let sentences =
            ["이것은 테스트입니다.", "나는 고양이를 좋아한다.", "한국어 형태소 분석기.", "글"];
        let segmenter = Segmenter::with_learner(Language::Korean, load_adaboost("korean.model"));
        assert_segment_matches_reference(&segmenter, &sentences);
        assert_segment_matches_reference(&segmenter, &STRESS_SENTENCES);
    }

    #[test]
    fn test_segment_differential_bocchan_corpus() {
        // Broad-coverage differential run over real text: the first 100
        // non-empty lines of bocchan.txt against the RWCP model.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources/bocchan.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(100).collect();
        assert!(!lines.is_empty());
        let segmenter = Segmenter::with_learner(Language::Japanese, load_adaboost("RWCP.model"));
        assert_segment_matches_reference(&segmenter, &lines);
    }

    #[test]
    fn test_segment_differential_synthetic_ambiguity_model() {
        // Handpicked sentinel-adjacent features: the packed table must give
        // each of these strings its own key so scoring matches the
        // string-keyed reference on inputs that generate them.
        let model = "BW1:B1x\t0.5\nUW1:B1\t0.4\nBW2:B1\t0.3\nUW4:x\t-0.2\n0.0\n";
        let mut learner = AdaBoost::new(0.01, 100);
        learner.load_model_from_reader(model.as_bytes()).unwrap();
        let segmenter = Segmenter::with_learner(Language::Japanese, learner);
        assert_segment_matches_reference(&segmenter, &["xyz", "B1x", "xB1", "B1", "x"]);
    }

    #[test]
    fn test_segment_scatter_offsets_per_template() {
        // One synthetic model per UW/BW template, each with a single strong
        // feature, segmented over strings where that feature fires at the
        // head/tail sentinel boundaries. An off-by-one in the scatter-add
        // offsets flips a boundary decision and diverges from the
        // string-keyed reference, so each template's offset is verified in
        // isolation.
        let single_feature_models = [
            "UW1:あ\t2.0\n0.0\n",
            "UW2:あ\t2.0\n0.0\n",
            "UW3:あ\t2.0\n0.0\n",
            "UW4:あ\t2.0\n0.0\n",
            "UW5:あ\t2.0\n0.0\n",
            "UW6:あ\t2.0\n0.0\n",
            "UW1:B2\t2.0\n0.0\n",
            "UW6:E2\t2.0\n0.0\n",
            "BW1:あい\t2.0\n0.0\n",
            "BW2:あい\t2.0\n0.0\n",
            "BW3:あい\t2.0\n0.0\n",
            "BW1:B1あ\t2.0\n0.0\n",
            "BW3:いE1\t2.0\n0.0\n",
        ];
        let inputs =
            ["あ", "あい", "あいう", "xあいうy", "ああああ", "あいあい", "いいあ", "あ漢い"];
        for model in single_feature_models {
            let mut learner = AdaBoost::new(0.01, 100);
            learner.load_model_from_reader(model.as_bytes()).unwrap();
            let segmenter = Segmenter::with_learner(Language::Japanese, learner);
            for input in inputs {
                assert_eq!(
                    segmenter.segment(input),
                    segmenter.segment_reference(input),
                    "model {model:?} input {input:?}"
                );
            }
        }
    }

    #[test]
    fn test_segment_cache_invalidated_by_learner_mut() {
        // Loading another model through learner_mut() must drop the compiled
        // table. Note load_model_from_reader MERGES into an already-populated
        // learner, so the correct oracle is the string-keyed reference (which
        // always reads the learner's current weights), not a fresh segmenter.
        let sentence = "これはテストです。";
        let mut segmenter =
            Segmenter::with_learner(Language::Japanese, load_adaboost("japanese.model"));
        let before = segmenter.segment(sentence);

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/RWCP.model");
        segmenter.learner_mut().load_model_from_path(&path).unwrap();
        let after = segmenter.segment(sentence);

        assert_eq!(after, segmenter.segment_reference(sentence));
        assert_ne!(before, after, "the merged model must actually disagree on this sentence");
    }

    #[test]
    fn test_segment_cache_invalidated_by_add_corpus_and_train() {
        // The add_corpus + learner_mut().train() workflow must also
        // invalidate: afterwards segment() must equal the string-keyed
        // reference (which always reads current weights).
        let mut segmenter = Segmenter::new(Language::Japanese);
        let _ = segmenter.segment("テストです");
        for _ in 0..10 {
            segmenter.add_corpus("テスト です");
        }
        let running = AtomicBool::new(true);
        segmenter.learner_mut().train(&running);
        assert_segment_matches_reference(&segmenter, &["テストです", "これはテストです。"]);
    }

    #[test]
    fn test_debug_impls() {
        // #129: user-facing types are debuggable.
        let segmenter = Segmenter::new(Language::Japanese);
        assert!(!format!("{:?}", segmenter).is_empty());
        let extractor = crate::extractor::Extractor::new(Language::Japanese);
        assert!(!format!("{:?}", extractor).is_empty());
    }

    #[test]
    fn test_segment_with_pos_without_learner_errors() {
        // #127: calling segment_with_pos without a POS learner returns a
        // recoverable error instead of panicking.
        let segmenter = Segmenter::new(Language::Japanese);
        let result = segmenter.segment_with_pos("これはテストです。");
        assert!(matches!(result, Err(LitseaError::PosLearnerNotSet)));
        // The empty sentence stays Ok regardless of learner state.
        assert!(segmenter.segment_with_pos("").unwrap().is_empty());
    }

    #[test]
    fn test_segment_with_pos_empty() {
        let segmenter = Segmenter::with_pos_learner(Language::Japanese, AveragedPerceptron::new());
        let result = segmenter.segment_with_pos("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_process_corpus_with_pos_empty() {
        let segmenter = Segmenter::new(Language::Japanese);
        let mut called = false;
        segmenter.add_corpus_with_pos_writer("", |_, _| {
            called = true;
        });
        assert!(!called);
    }

    // --- Packed POS scoring path tests (#143) ---

    fn load_perceptron(model: &str) -> AveragedPerceptron {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model);
        let mut learner = AveragedPerceptron::new();
        learner.load_model_from_path(&path).unwrap();
        learner
    }

    fn assert_pos_matches_reference(segmenter: &Segmenter, sentences: &[&str]) {
        for sentence in sentences {
            assert_eq!(
                segmenter.segment_with_pos(sentence).unwrap(),
                segmenter.segment_with_pos_reference(sentence).unwrap(),
                "packed segment_with_pos diverged from the string-keyed reference for {sentence:?}"
            );
        }
    }

    #[test]
    fn test_segment_with_pos_differential_japanese_model() {
        let sentences = [
            "これはテストです。",
            "私の猫は可愛い。",
            "東京都に住んでいます。",
            "字",
            "こんにちは",
            "価格は1000円です。",
            "RustでNLPを実装する。",
        ];
        let segmenter =
            Segmenter::with_pos_learner(Language::Japanese, load_perceptron("japanese_pos.model"));
        assert_pos_matches_reference(&segmenter, &sentences);
        assert_pos_matches_reference(&segmenter, &STRESS_SENTENCES);
    }

    #[test]
    fn test_segment_with_pos_differential_chinese_model() {
        let sentences =
            ["这是一个测试。", "我喜欢吃中国菜。", "他在北京工作。", "好", "2024年的春天。"];
        let segmenter =
            Segmenter::with_pos_learner(Language::Chinese, load_perceptron("chinese_pos.model"));
        assert_pos_matches_reference(&segmenter, &sentences);
        assert_pos_matches_reference(&segmenter, &STRESS_SENTENCES);
    }

    #[test]
    fn test_segment_with_pos_differential_korean_model() {
        let sentences =
            ["이것은 테스트입니다.", "나는 고양이를 좋아한다.", "한국어 형태소 분석기.", "글"];
        let segmenter =
            Segmenter::with_pos_learner(Language::Korean, load_perceptron("korean_pos.model"));
        assert_pos_matches_reference(&segmenter, &sentences);
        assert_pos_matches_reference(&segmenter, &STRESS_SENTENCES);
    }

    #[test]
    fn test_segment_with_pos_differential_bocchan_corpus() {
        // Broad-coverage differential run over real text: the first 50
        // non-empty lines of bocchan.txt against the bundled Japanese POS
        // model.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources/bocchan.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(50).collect();
        assert!(!lines.is_empty());
        let segmenter =
            Segmenter::with_pos_learner(Language::Japanese, load_perceptron("japanese_pos.model"));
        assert_pos_matches_reference(&segmenter, &lines);
    }

    #[test]
    #[ignore = "full-corpus sweep (slow with the string-keyed reference); run explicitly with --ignored"]
    fn test_segment_differential_bocchan_full() {
        // Full-corpus differential net for the packed AdaBoost scorer:
        // every non-empty bocchan line must match the string-keyed
        // reference exactly, for every bundled Japanese model. Added with
        // the WC merged-row restructuring (#157); japanese.model exercises
        // the WC scatter path (6384 WC features, per the #165 retraining),
        // RWCP.model the empty-map skip, and JEITA the small-model path.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources/bocchan.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(lines.len() > 400, "bocchan corpus unexpectedly small: {}", lines.len());
        for model in ["japanese.model", "RWCP.model", "JEITA_Genpaku_ChaSen_IPAdic.model"] {
            let segmenter = Segmenter::with_learner(Language::Japanese, load_adaboost(model));
            let mut diverged = 0usize;
            for line in &lines {
                if segmenter.segment(line) != segmenter.segment_reference(line) {
                    diverged += 1;
                }
            }
            assert_eq!(diverged, 0, "{model}: {diverged} of {} lines diverged", lines.len());
        }
    }

    #[test]
    #[ignore = "full-corpus sweep (slow with the string-keyed reference); run explicitly with --ignored"]
    fn test_segment_with_pos_differential_bocchan_full() {
        // Full-corpus differential net for the packed POS scorer: every
        // non-empty bocchan line must match the string-keyed reference
        // exactly. The fast suite covers the first 50 lines; this sweep
        // covers the whole novel. (It also served as the adoption gate for
        // the f32-table experiment, #145: zero divergence but no measurable
        // speedup, so f64 stays.)
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources/bocchan.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(lines.len() > 400, "bocchan corpus unexpectedly small: {}", lines.len());
        let segmenter =
            Segmenter::with_pos_learner(Language::Japanese, load_perceptron("japanese_pos.model"));
        let mut diverged = 0usize;
        for line in &lines {
            if segmenter.segment_with_pos(line).unwrap()
                != segmenter.segment_with_pos_reference(line).unwrap()
            {
                diverged += 1;
            }
        }
        assert_eq!(diverged, 0, "{diverged} of {} lines diverged", lines.len());
    }

    #[test]
    fn test_segment_with_pos_differential_trained_in_memory() {
        // Weights straight out of train() (not a saved/loaded file) must
        // also match: exercises zero-weight columns in live FeatureSlots and
        // the packed-cache invalidation of add_corpus_with_pos.
        let mut segmenter = Segmenter::new(Language::Japanese);
        for _ in 0..20 {
            segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
            segmenter.add_corpus_with_pos("私/PRON の/PART 猫/NOUN は/PART 可愛い/ADJ 。/PUNCT");
        }
        let running = AtomicBool::new(true);
        segmenter.pos_learner_mut().unwrap().train(10, &running);
        assert_pos_matches_reference(
            &segmenter,
            &["これはテストです。", "私の猫は可愛い。", "未知の文も分割する。"],
        );
    }

    #[test]
    fn test_segment_with_pos_cache_invalidated_by_mutation() {
        // A prediction compiles the packed table; further training through
        // pos_learner_mut()/add_corpus_with_pos must drop it so the next
        // prediction reflects the new weights (the reference always reads
        // the learner's current weights).
        let mut segmenter = Segmenter::new(Language::Japanese);
        for _ in 0..5 {
            segmenter.add_corpus_with_pos("これ/PRON は/PART テスト/NOUN です/AUX 。/PUNCT");
        }
        let running = AtomicBool::new(true);
        segmenter.pos_learner_mut().unwrap().train(5, &running);
        let _ = segmenter.segment_with_pos("これはテストです。").unwrap();

        for _ in 0..20 {
            segmenter.add_corpus_with_pos("猫/NOUN が/PART 鳴く/VERB 。/PUNCT");
        }
        segmenter.pos_learner_mut().unwrap().train(10, &running);
        assert_pos_matches_reference(&segmenter, &["猫が鳴く。", "これはテストです。"]);
    }

    #[test]
    fn test_segment_with_pos_zero_class_perceptron() {
        // A perceptron without classes predicts nothing: every position maps
        // to O and the whole sentence becomes one word tagged X, exactly as
        // the reference path's empty-prediction fallback behaves.
        let segmenter = Segmenter::with_pos_learner(Language::Japanese, AveragedPerceptron::new());
        let result = segmenter.segment_with_pos("テスト").unwrap();
        assert_eq!(result, vec![("テスト".to_string(), Upos::X)]);
        assert_eq!(result, segmenter.segment_with_pos_reference("テスト").unwrap());
    }
}
