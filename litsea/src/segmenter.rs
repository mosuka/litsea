//! Text segmentation engine.
//!
//! Defines [`Segmenter`], which performs word segmentation (AdaBoost) and
//! two-stage segmentation + POS tagging (boundary classifier + lexicon +
//! word-level tagger, issue #147), scoring through the packed tables of
//! `crate::packed_model` / `crate::packed_two_stage`.
//! Also hosts the corpus-processing pipeline used to build training features.

use std::collections::HashSet;
use std::sync::{PoisonError, RwLock};

use crate::adaboost::AdaBoost;
use crate::error::{LitseaError, Result};
use crate::language::{Language, OTHER_TYPE_ID};
use crate::packed_model::{
    PackedModel, SENTINEL_BASE, Slot, TAG_B, TAG_O, TAG_U, TEMPLATES, templates_for,
};
use crate::packed_two_stage::PackedTwoStageModel;
use crate::two_stage::TwoStageLearner;
use crate::upos::{SegmentLabel, Upos};

/// Reusable scratch and output storage for
/// [`Segmenter::segment_into`], the allocation-free variant of
/// [`Segmenter::segment`] (issue #184).
///
/// A `SegmentBuffer` owns every per-call allocation the segmentation hot
/// path needs — the packed context arrays, the static score buffer, the
/// boundary-tag scratch, and the output token ranges. Each `segment_into`
/// call clears and refills it, so a buffer reused across a batch of
/// sentences reaches a steady state where segmentation allocates nothing:
/// every vector keeps the capacity of the longest sentence seen so far.
///
/// The buffer holds plain data (no borrows), so one buffer can be reused
/// across sentences, models, and languages; for parallel batch processing
/// use one buffer per thread. Construct with [`SegmentBuffer::new`] (or
/// `Default`); the fields are internal.
#[derive(Debug, Default)]
pub struct SegmentBuffer {
    /// Packed char codes with sentinels (see `Segmenter::packed_context`).
    char_codes: Vec<u32>,
    /// Char type ids with sentinels.
    type_ids: Vec<u8>,
    /// Byte offset of each real character, plus the sentence length as the
    /// final entry — the source for the output ranges.
    char_starts: Vec<usize>,
    /// Per-position static-pass score accumulator.
    static_scores: Vec<f64>,
    /// Boundary-tag scratch for the sequential pass (unused on the
    /// pointwise fast path, #183).
    tags: Vec<u8>,
    /// Output: byte ranges of the segmented tokens, in order.
    ranges: Vec<(usize, usize)>,
}

impl SegmentBuffer {
    /// Creates an empty buffer.
    ///
    /// # Returns
    /// A new [`SegmentBuffer`] with no allocated capacity; capacity grows
    /// on first use and is retained across [`Segmenter::segment_into`]
    /// calls.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Text segmenter supporting two modes: word segmentation via AdaBoost
/// binary classification, and two-stage word segmentation + POS tagging via
/// a boundary classifier, lexicon, and word-level tagger (see
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
    /// The learner's weights compiled to packed integer keys for
    /// [`segment`](Self::segment)'s hot loop. `None` after a learner
    /// mutation; lazily rebuilt on the next `segment` call.
    packed: RwLock<Option<PackedModel>>,
    /// The stage-2 half of a two-stage model, compiled into packed tagging
    /// tables by [`with_two_stage_learner`](Self::with_two_stage_learner)
    /// (the stage-1 half lives in `learner`). When present,
    /// [`segment_with_pos`](Self::segment_with_pos) runs the two-stage
    /// tagging path.
    ///
    /// Unlike `packed` this is not a lazily-rebuilt cache of some other
    /// field — it *is* the stage-2 model; the raw [`TwoStageLearner`] parts
    /// are dropped after compilation, and there is no mutable accessor for
    /// it. If in-place two-stage mutation is ever added, follow the
    /// [`learner_mut`](Self::learner_mut) shape (keep the learner, cache the
    /// compilation, invalidate on mutation) rather than mutating this field
    /// directly.
    two_stage: Option<PackedTwoStageModel>,
}

// Compile-time assertion: parallel batch callers (e.g. the CLI's
// `segment --threads`, issue #185) share one `Segmenter` across worker
// threads and move per-worker `SegmentBuffer`s into them. A future field
// whose type is not `Send + Sync` (an `Rc`, a `RefCell`, a raw pointer)
// must fail to compile here, at the definition, rather than at a caller.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Segmenter>();
    assert_send_sync::<SegmentBuffer>();
};

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
            packed,
            two_stage: None,
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
        // load-then-segment path never rebuilds mid-stream. The raw stage-2
        // parts are dropped after compilation: the packed model contains
        // everything the tagging path needs, and there is no mutation path
        // that would require rebuilding it (see the `two_stage` field doc).
        let packed = RwLock::new(Some(PackedModel::build(language, &stage1)));
        let two_stage = PackedTwoStageModel::build(language, &stage2, &lexicon, dominance);
        Segmenter {
            language,
            learner: stage1,
            packed,
            two_stage: Some(two_stage),
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

    /// Fills `buf`'s context arrays for [`segment_into`](Self::segment_into):
    /// the same sentinel layout as [`packed_context`](Self::packed_context)
    /// for `char_codes` / `type_ids`, plus `char_starts` holding the byte
    /// offset of every real character followed by `text.len()` (so the word
    /// covering real characters `r..s` spans bytes
    /// `char_starts[r]..char_starts[s]`). No string slices are stored, which
    /// is what lets the buffer be reused across sentences.
    fn packed_context_into(&self, text: &str, buf: &mut SegmentBuffer) {
        buf.char_codes.clear();
        buf.type_ids.clear();
        buf.char_starts.clear();
        buf.char_codes
            .extend_from_slice(&[SENTINEL_BASE, SENTINEL_BASE + 1, SENTINEL_BASE + 2]);
        buf.type_ids.extend_from_slice(&[OTHER_TYPE_ID; 3]);
        for (i, ch) in text.char_indices() {
            buf.char_starts.push(i);
            buf.char_codes.push(u32::from(ch));
            buf.type_ids.push(self.language.char_type_id(ch));
        }
        buf.char_starts.push(text.len());
        buf.char_codes.extend_from_slice(&[
            SENTINEL_BASE + 3,
            SENTINEL_BASE + 4,
            SENTINEL_BASE + 5,
        ]);
        buf.type_ids.extend_from_slice(&[OTHER_TYPE_ID; 3]);
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
    /// `sep` selects the token separator: `' '` for the classic
    /// "word/POS word/POS ..." corpus, `'\t'` for the space-preserving TSV
    /// corpus (issue #198), where a token may be a literal space `" "` and
    /// therefore carries no `/POS` suffix (it gets [`Upos::X`], which the
    /// two-stage extractor discards along with every other POS when it
    /// collapses stage-1 labels to `B`/`O`).
    ///
    /// Example (`sep = ' '`): "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT"
    fn process_corpus_with_pos<F>(&self, corpus: &str, sep: char, callback: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        let tokens = corpus.split(sep).map(|token| {
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

    /// Processes a POS-tagged corpus's features with a custom writer.
    ///
    /// # Arguments
    /// * `corpus` - A POS-tagged corpus ("word/POS word/POS ..." format)
    /// * `writer` - A closure receiving the attribute set and SegmentLabel for each character position
    pub fn add_corpus_with_pos_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        self.process_corpus_with_pos(corpus, ' ', writer);
    }

    /// Tab-separated variant of
    /// [`add_corpus_with_pos_writer`](Self::add_corpus_with_pos_writer)
    /// (issue #198).
    ///
    /// Tokens are separated by tabs and a token may be a literal space
    /// `" "`, so the training text preserves the original spacing of the
    /// sentence — the POS-pipeline counterpart of
    /// [`add_corpus_tsv_with_writer`](Self::add_corpus_tsv_with_writer).
    /// For space-delimited languages (Korean, English) this is what lets a
    /// two-stage model learn from the space characters its input actually
    /// contains, instead of from an unspaced concatenation it never sees at
    /// inference.
    ///
    /// A literal-space token has no `/POS` suffix and is therefore labeled
    /// `B(`[`Upos::X`]`)`; the two-stage extractor collapses every stage-1
    /// label to `B`/`O`, so the tag on a space is inert there.
    ///
    /// # Arguments
    /// * `corpus` - A tab-separated POS-tagged corpus ("word/POS\tword/POS\t..." format)
    /// * `writer` - A closure receiving the attribute set and SegmentLabel for each character position
    pub fn add_corpus_tsv_with_pos_writer<F>(&self, corpus: &str, writer: F)
    where
        F: FnMut(HashSet<String>, SegmentLabel),
    {
        self.process_corpus_with_pos(corpus, '\t', writer);
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
    /// whether it starts a new word. For a pointwise model (no tag-dependent
    /// features, e.g. one trained with those templates filtered out) the
    /// sequential pass is skipped entirely (issue #183); output is
    /// unaffected, since the skipped loads would all contribute `0.0`.
    /// No attribute strings are constructed:
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
        // Thin wrapper over segment_into (issue #184): a fresh buffer per
        // call, with each output range materialized as an owned String.
        // Keeping a single scoring implementation means every differential
        // and golden test of this method also pins segment_into's core.
        let mut buf = SegmentBuffer::new();
        self.segment_into(sentence, &mut buf)
            .iter()
            .map(|&(start, end)| sentence[start..end].to_string())
            .collect()
    }

    /// Segments a sentence into byte ranges of `sentence`, reusing `buf`'s
    /// allocations — the allocation-free variant of
    /// [`segment`](Self::segment) (issue #184).
    ///
    /// Each returned `(start, end)` pair is a byte range into `sentence`
    /// (`&sentence[start..end]` is the token), in order; the ranges tile
    /// the sentence exactly. Reusing the same [`SegmentBuffer`] across a
    /// batch of sentences amortizes every per-call allocation away: after
    /// the first few sentences the buffer's vectors have the capacity they
    /// need and segmentation allocates nothing.
    ///
    /// # Arguments
    /// * `sentence` - The sentence to segment.
    /// * `buf` - The scratch/output buffer to (re)use; cleared and refilled
    ///   by this call.
    ///
    /// # Returns
    /// The segmented tokens as byte ranges into `sentence`, borrowed from
    /// `buf` (valid until the next use of `buf`). Empty for an empty
    /// sentence.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    ///
    /// use litsea::adaboost::AdaBoost;
    /// use litsea::language::Language;
    /// use litsea::segmenter::{SegmentBuffer, Segmenter};
    ///
    /// let model_file =
    ///     PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join("RWCP.model");
    /// let mut learner = AdaBoost::new(0.01, 100);
    /// learner.load_model_from_path(&model_file).unwrap();
    /// let segmenter = Segmenter::with_learner(Language::Japanese, learner);
    ///
    /// let sentence = "これはテストです。";
    /// let mut buf = SegmentBuffer::new();
    /// let tokens: Vec<&str> = segmenter
    ///     .segment_into(sentence, &mut buf)
    ///     .iter()
    ///     .map(|&(start, end)| &sentence[start..end])
    ///     .collect();
    /// assert_eq!(tokens, vec!["これ", "は", "テスト", "です", "。"]);
    /// ```
    pub fn segment_into<'b>(
        &self,
        sentence: &str,
        buf: &'b mut SegmentBuffer,
    ) -> &'b [(usize, usize)] {
        buf.ranges.clear();
        if sentence.is_empty() {
            return &buf.ranges;
        }
        self.packed_context_into(sentence, buf);

        // The bias is a sum over all model weights; compute it once per
        // sentence instead of once per character.
        let bias = self.learner.bias();

        self.with_packed(|packed| {
            let char_codes = &buf.char_codes;
            let type_ids = &buf.type_ids;
            let char_starts = &buf.char_starts;
            let type_radix = self.language.type_codes().len();
            let n = char_codes.len();
            // Decision positions: lo..=hi (position 3 is the first real
            // character and always starts the first word).
            let lo = 4usize;
            let hi = n - 4;

            // ---- Static pass: everything that does not depend on boundary
            // tags is accumulated into a per-position buffer in one sweep.
            // The f64 accumulation order differs from the string-keyed
            // reference here (see the module docs of packed_model); output
            // equality is pinned empirically by the differential tests.
            buf.static_scores.clear();
            buf.static_scores.resize(n, 0.0);
            let static_scores = &mut buf.static_scores;
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
            // A boundary at decision position i means real character i - 3
            // starts a new word, closing the current word at byte offset
            // char_starts[i - 3]; the final word always ends at the last
            // char_starts entry (the sentence length).
            let static_scores = &buf.static_scores;
            let ranges = &mut buf.ranges;
            let mut word_start = 0usize; // real-character index
            if packed.has_tag_features {
                let t = type_radix;
                let d = &packed.dense;
                buf.tags.clear();
                let tags = &mut buf.tags;
                tags.extend_from_slice(&[TAG_U; 4]);
                for i in 4..=hi {
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
                        ranges.push((char_starts[word_start], char_starts[i - 3]));
                        word_start = i - 3;
                        tags.push(TAG_B);
                    } else {
                        tags.push(TAG_O);
                    }
                }
            } else {
                // Pointwise fast path (#183): every tag-dependent table is
                // all-zero, so the 16 loads above would each add 0.0 and
                // the tag bookkeeping would feed nothing — the decision
                // reduces exactly to the static score plus the bias. This
                // branch is equivalence-pinned against segment_reference
                // by the tag-free differential test.
                for i in 4..=hi {
                    if bias + static_scores[i] >= 0.0 {
                        ranges.push((char_starts[word_start], char_starts[i - 3]));
                        word_start = i - 3;
                    }
                }
            }
            ranges.push((char_starts[word_start], char_starts[n - 6]));
        });
        &buf.ranges
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

    /// Segments the sentence and tags each word with its POS through the
    /// two-stage pipeline (issue #147): the sentence is segmented by the
    /// stage-1 boundary classifier (exactly as [`segment`](Self::segment)),
    /// then each word is tagged through the candidate-tag lexicon —
    /// single-candidate and dominant surfaces skip the classifier entirely —
    /// with the stage-2 word-level tagger deciding ambiguous surfaces
    /// (candidate-masked argmax) and unknown surfaces (full argmax).
    ///
    /// # Arguments
    /// * `sentence` - The sentence to segment
    ///
    /// # Returns
    /// `Result<Vec<(String, Upos)>>` - Pairs of words and their POS tags.
    /// An empty sentence yields `Ok` with an empty vector.
    ///
    /// # Errors
    /// Returns [`LitseaError::PosLearnerNotSet`] if no two-stage learner is
    /// set. Build the segmenter with
    /// [`with_two_stage_learner`](Self::with_two_stage_learner) beforehand.
    pub fn segment_with_pos(&self, sentence: &str) -> Result<Vec<(String, Upos)>> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let packed = self.two_stage.as_ref().ok_or(LitseaError::PosLearnerNotSet)?;
        let words = self.segment(sentence);
        let tags = packed.tag_words(self.language, &words);
        Ok(words.into_iter().zip(tags).collect())
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
    fn test_char_type_english() {
        let segmenter = Segmenter::new(Language::English);

        assert_eq!(segmenter.char_type('T'), "U"); // Uppercase
        assert_eq!(segmenter.char_type('t'), "A"); // Lowercase
        assert_eq!(segmenter.char_type(' '), "W"); // Space
        assert_eq!(segmenter.char_type('\''), "Q"); // Apostrophe
        assert_eq!(segmenter.char_type('.'), "P"); // Punctuation
        assert_eq!(segmenter.char_type('5'), "N"); // Digit
        assert_eq!(segmenter.char_type('字'), "O"); // Other
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

    // --- POS corpus pipeline tests ---

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
    fn test_add_corpus_tsv_with_pos_writer() {
        // Space-preserving POS corpus (issue #198): the literal-space token
        // becomes one B-labeled position of its own, and the space character
        // reaches the training text so neighbouring positions see it.
        let segmenter = Segmenter::new(Language::English);
        let corpus = "do/AUX\t \tit/PRON";
        let mut collected = Vec::new();

        segmenter.add_corpus_tsv_with_pos_writer(corpus, |attrs, label| {
            collected.push((attrs, label));
        });

        // "do it" has 5 characters, the space included: d=B-AUX, o=O,
        // " "=B-X, i=B-PRON, t=O.
        assert_eq!(collected.len(), 5);
        assert_eq!(collected[0].1, SegmentLabel::B(Upos::AUX));
        // The space token has no /POS suffix, so it is labeled B-X; the
        // two-stage extractor collapses every stage-1 label to B/O, making
        // the tag itself inert there.
        assert_eq!(collected[2].1, SegmentLabel::B(Upos::X));
        assert_eq!(collected[3].1, SegmentLabel::B(Upos::PRON));

        let boundary_count = collected.iter().filter(|(_, l)| l.is_boundary()).count();
        assert_eq!(boundary_count, 3); // "do", the space, and "it"

        // The space character must be visible in the feature context -- this
        // is the whole reason the TSV variant exists.
        assert!(
            collected.iter().any(|(attrs, _)| attrs.contains("UW4: ")),
            "the space character must reach the training features"
        );
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
    fn test_segment_differential_english_model() {
        let sentences = [
            "This is a test.",
            "I don't know.",
            "Google's search engine.",
            "The price is $1,000.",
            "word",
        ];
        let segmenter = Segmenter::with_learner(Language::English, load_adaboost("english.model"));
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

    /// Reconstructs tokens from `segment_into`'s byte ranges and asserts
    /// they match `segment()`'s output, and that the ranges tile the input
    /// exactly: first start 0, each start equal to the previous end, last
    /// end equal to the sentence length.
    fn assert_segment_into_matches(segmenter: &Segmenter, sentences: &[&str]) {
        let mut buf = SegmentBuffer::new();
        for sentence in sentences {
            let ranges = segmenter.segment_into(sentence, &mut buf);
            let mut cursor = 0usize;
            for &(start, end) in ranges {
                assert_eq!(start, cursor, "ranges do not tile {sentence:?}: {ranges:?}");
                assert!(start < end, "empty or reversed range in {sentence:?}: {ranges:?}");
                cursor = end;
            }
            assert_eq!(cursor, sentence.len(), "ranges do not cover {sentence:?}: {ranges:?}");
            let tokens: Vec<String> =
                ranges.iter().map(|&(s, e)| sentence[s..e].to_string()).collect();
            assert_eq!(
                tokens,
                segmenter.segment(sentence),
                "segment_into diverged from segment on {sentence:?}"
            );
        }
    }

    #[test]
    fn test_segment_into_tiles_and_matches_segment() {
        let sentences = [
            "これはテストです。",
            "私の猫は可愛い。",
            "東京都に住んでいます。",
            "字",
            "こんにちは",
            "価格は1000円です。",
            "RustでNLPを実装する。",
        ];
        // Tagged path (bundled models with tag features) and legacy models.
        for model in ["japanese.model", "RWCP.model", "JEITA_Genpaku_ChaSen_IPAdic.model"] {
            let segmenter = Segmenter::with_learner(Language::Japanese, load_adaboost(model));
            assert_segment_into_matches(&segmenter, &sentences);
            assert_segment_into_matches(&segmenter, &STRESS_SENTENCES);
        }
        // Pointwise fast path (#183): korean.model is tag-free, and its
        // corpus contains multi-byte Hangul plus literal space tokens.
        let segmenter = Segmenter::with_learner(Language::Korean, load_adaboost("korean.model"));
        assert_segment_into_matches(
            &segmenter,
            &["이것은 테스트입니다.", "나는 고양이를 좋아한다.", "글", "2024년 봄."],
        );
        // Pointwise fast path: english.model is tag-free too, with literal
        // space tokens and ASCII text.
        let segmenter = Segmenter::with_learner(Language::English, load_adaboost("english.model"));
        assert_segment_into_matches(
            &segmenter,
            &["This is a test.", "I don't know.", "Google's search engine.", "word"],
        );
        // Empty input yields an empty range slice.
        let mut buf = SegmentBuffer::new();
        assert!(segmenter.segment_into("", &mut buf).is_empty());
    }

    #[test]
    fn test_segment_into_buffer_reuse_is_stateless() {
        // A buffer carrying capacity (and stale contents) from previous
        // sentences must produce exactly what a fresh buffer produces:
        // long -> short -> long again, across models with and without tag
        // features. This is the load-bearing test for the clear/refill
        // logic — stale range, tag, or score entries would surface here.
        let tagged = Segmenter::with_learner(Language::Japanese, load_adaboost("japanese.model"));
        let pointwise =
            Segmenter::with_learner(Language::Japanese, load_adaboost_tag_free("japanese.model"));
        let sequence = [
            "東京都に住んでいます。価格は1000円です。これはテストです。",
            "字",
            "こんにちは",
            "東京都に住んでいます。価格は1000円です。これはテストです。",
            "",
            "私の猫は可愛い。",
        ];
        for segmenter in [&tagged, &pointwise] {
            let mut reused = SegmentBuffer::new();
            for sentence in sequence {
                let got: Vec<(usize, usize)> =
                    segmenter.segment_into(sentence, &mut reused).to_vec();
                let mut fresh = SegmentBuffer::new();
                let want = segmenter.segment_into(sentence, &mut fresh);
                assert_eq!(got, want, "reused buffer diverged on {sentence:?}");
            }
        }
    }

    /// Loads a bundled model with its tag-dependent (`UP*`/`BP*`/`UQ*`/
    /// `BQ*`/`TQ*`) feature lines filtered out, producing a pointwise model
    /// that exercises `segment()`'s tag-free fast path (#183).
    fn load_adaboost_tag_free(model: &str) -> AdaBoost {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models").join(model);
        let text = std::fs::read_to_string(&path).unwrap();
        let filtered: String = text
            .lines()
            .filter(|l| !crate::packed_model::is_tag_dependent_feature(l))
            .collect::<Vec<_>>()
            .join("\n");
        let mut learner = AdaBoost::default();
        learner.load_model_from_reader(filtered.as_bytes()).unwrap();
        learner
    }

    #[test]
    fn test_segment_differential_tag_free_fast_path() {
        // The pointwise fast path (#183) must produce output identical to
        // the string-keyed reference. Filter the tag features out of a real
        // bundled model, confirm the gate actually fires, and run the same
        // differential coverage as the tagged path gets: fixed sentences,
        // stress sentences, and real bocchan text.
        let segmenter =
            Segmenter::with_learner(Language::Japanese, load_adaboost_tag_free("japanese.model"));
        assert!(
            segmenter.with_packed(|p| !p.has_tag_features),
            "tag-filtered model must take the pointwise fast path"
        );
        let sentences = [
            "これはテストです。",
            "私の猫は可愛い。",
            "東京都に住んでいます。",
            "字",
            "こんにちは",
            "価格は1000円です。",
            "RustでNLPを実装する。",
        ];
        assert_segment_matches_reference(&segmenter, &sentences);
        assert_segment_matches_reference(&segmenter, &STRESS_SENTENCES);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../resources/bocchan.txt");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(100).collect();
        assert!(!lines.is_empty());
        assert_segment_matches_reference(&segmenter, &lines);

        // Control: the unfiltered bundled model takes the tagged path.
        let tagged = Segmenter::with_learner(Language::Japanese, load_adaboost("japanese.model"));
        assert!(tagged.with_packed(|p| p.has_tag_features));
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
        // #127: calling segment_with_pos without a two-stage learner returns
        // a recoverable error instead of panicking.
        let segmenter = Segmenter::new(Language::Japanese);
        let result = segmenter.segment_with_pos("これはテストです。");
        assert!(matches!(result, Err(LitseaError::PosLearnerNotSet)));
        // The empty sentence stays Ok regardless of learner state.
        assert!(segmenter.segment_with_pos("").unwrap().is_empty());
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
}
