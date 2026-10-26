//! User dictionary: words the segmenter must emit as single tokens (issue
//! #263), optionally with a UPOS tag for POS tagging (issue #264).
//!
//! A [`UserDictionary`] stores registered words in a trie keyed by Unicode
//! scalar values, the same `u32` codes that `Segmenter::segment_into` packs
//! into its context buffer, so matching walks the packed codes directly
//! without decoding the sentence again. The matching semantics (both edges
//! of an occurrence forced to be boundaries, no boundary inside it,
//! leftmost-longest resolution of overlapping occurrences, and the optional
//! gate) and the tag semantics are documented on [`UserDictionary`] and
//! implemented in `crate::segmenter`.

use rustc_hash::FxHashMap;

use crate::error::{LitseaError, Result};
use crate::upos::Upos;

/// Node id of the trie root.
const ROOT: u32 = 0;

/// A set of words that [`Segmenter`](crate::segmenter::Segmenter) emits as
/// single tokens whenever they occur in the input.
///
/// # Semantics
///
/// - An occurrence of a registered word is emitted as exactly one token:
///   the segmenter forces a boundary at each edge of the occurrence and
///   suppresses every boundary inside it. A forced decision enters the
///   model's boundary-tag history like a model decision would, so the
///   positions after the word are scored with consistent features.
/// - Occurrences that overlap are resolved leftmost-longest: scanning the
///   sentence from the left, the longest registered word starting at the
///   current position wins, and the scan resumes after it.
/// - Words are matched as given — no case folding, no width or Unicode
///   normalization. A word may contain a space; for languages whose
///   models emit spaces as tokens (Korean, English), such a word becomes
///   one token that contains the space.
/// - Words of one character or less have no inside and are not stored
///   ([`add`](Self::add) and [`add_with_pos`](Self::add_with_pos) return
///   `false` for them, and their tag is dropped too).
///
/// # Tags
///
/// A word registered with [`add_with_pos`](Self::add_with_pos) carries a
/// UPOS tag (issue #264). In
/// [`Segmenter::segment_with_pos`](crate::segmenter::Segmenter::segment_with_pos),
/// every token whose surface is a tagged word gets that tag instead of
/// the tagger's — whether the lexicon would have fixed the tag or the
/// stage-2 tagger would have decided it — and every other token keeps the
/// tagger's tag. Without a gate the tagged tokens are exactly the
/// occurrences the dictionary forces; with a gate, a rejected occurrence
/// that the model segments exactly as registered anyway is tagged as
/// well. An untagged word is tagged by the two-stage path like any other
/// word, and [`Segmenter::segment`](crate::segmenter::Segmenter::segment)
/// ignores tags altogether.
///
/// # Gate
///
/// With a gate set ([`set_gate`](Self::set_gate)), an occurrence is applied
/// only if its *flip ratio* is at most the gate. The flip ratio is the sum
/// of the model's absolute scores at the positions whose unconstrained
/// decision the word would overturn, divided by the sum of the absolute
/// scores at every position the word constrains, so it lies in `0.0..=1.0`
/// and needs no per-model tuning: `0.0` applies a word only where the
/// model already agrees with it, `1.0` applies it everywhere. The gate
/// guards dictionaries that contain short or ambiguous entries (a
/// two-character fragment registered as a word breaks every other word it
/// occurs in); with no gate (the default) a registered word always wins.
///
/// # Example
///
/// ```
/// use litsea::upos::Upos;
/// use litsea::user_dictionary::UserDictionary;
///
/// let mut dictionary = UserDictionary::from_words(["雪だるま", "新幹線"]);
/// assert_eq!(dictionary.len(), 2);
/// assert!(dictionary.contains("新幹線"));
/// assert!(!dictionary.add("あ")); // one character: not stored
/// assert!(dictionary.add_with_pos("チャットGPT", Upos::PROPN));
/// assert_eq!(dictionary.pos("チャットGPT"), Some(Upos::PROPN));
/// assert_eq!(dictionary.pos("新幹線"), None); // registered without a tag
/// dictionary.set_gate(Some(0.5)).unwrap();
/// assert_eq!(dictionary.gate(), Some(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct UserDictionary {
    /// Trie edges: `(parent node, character code)` → child node.
    edges: FxHashMap<(u32, u32), u32>,
    /// Whether the node with the given id ends a registered word. Index 0
    /// is the root.
    terminal: Vec<bool>,
    /// UPOS tags of the words registered with a tag (#264), keyed by the
    /// word's terminal node. Untagged words have no entry, so a dictionary
    /// without tags keeps this map empty.
    tags: FxHashMap<u32, Upos>,
    /// Number of registered words.
    len: usize,
    /// Length in characters of the longest registered word (0 when empty).
    max_len: usize,
    /// Optional flip-ratio gate in `0.0..=1.0`; `None` applies every
    /// occurrence.
    gate: Option<f64>,
}

impl Default for UserDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl UserDictionary {
    /// Creates an empty dictionary with no gate.
    ///
    /// # Returns
    /// A dictionary that matches nothing until words are added.
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: FxHashMap::default(),
            terminal: vec![false],
            tags: FxHashMap::default(),
            len: 0,
            max_len: 0,
            gate: None,
        }
    }

    /// Creates a dictionary from an iterator of words.
    ///
    /// Each word goes through [`add`](Self::add), so words of one character
    /// or less and duplicates are skipped silently.
    ///
    /// # Arguments
    /// * `words` - The words to register.
    ///
    /// # Returns
    /// The dictionary holding every word of two or more characters from
    /// `words`, with no gate.
    pub fn from_words<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut dictionary = Self::new();
        for word in words {
            dictionary.add(word.as_ref());
        }
        dictionary
    }

    /// Registers a word.
    ///
    /// A word that is already registered keeps its tag, if it has one.
    ///
    /// # Arguments
    /// * `word` - The word to register, matched later exactly as given.
    ///
    /// # Returns
    /// `true` if the word was newly registered; `false` if it has fewer
    /// than two characters (there is no inside to keep together) or was
    /// already registered.
    pub fn add(&mut self, word: &str) -> bool {
        self.insert(word).is_some_and(|(_, added)| added)
    }

    /// Registers a word with a UPOS tag (issue #264).
    ///
    /// The word constrains segmentation exactly like one registered with
    /// [`add`](Self::add); the tag only affects
    /// [`Segmenter::segment_with_pos`](crate::segmenter::Segmenter::segment_with_pos),
    /// which gives every token whose surface is this word the tag (see the
    /// type documentation).
    ///
    /// # Arguments
    /// * `word` - The word to register, matched later exactly as given.
    /// * `pos` - The tag to give the word's tokens. If the word is already
    ///   registered, its tag is set to `pos`, replacing any earlier one.
    ///
    /// # Returns
    /// `true` if the word was newly registered; `false` if it has fewer
    /// than two characters (nothing is stored, the tag included) or was
    /// already registered (its tag is still set).
    pub fn add_with_pos(&mut self, word: &str, pos: Upos) -> bool {
        let Some((node, added)) = self.insert(word) else {
            return false;
        };
        self.tags.insert(node, pos);
        added
    }

    /// Inserts a word into the trie, shared by [`add`](Self::add) and
    /// [`add_with_pos`](Self::add_with_pos).
    ///
    /// # Arguments
    /// * `word` - The word to register.
    ///
    /// # Returns
    /// `None` if the word has fewer than two characters (nothing is
    /// stored); otherwise the word's terminal node and whether the word
    /// was newly registered.
    fn insert(&mut self, word: &str) -> Option<(u32, bool)> {
        let char_count = word.chars().count();
        if char_count < 2 {
            return None;
        }
        let mut node = ROOT;
        for ch in word.chars() {
            let code = u32::from(ch);
            let next_id = self.terminal.len() as u32;
            node = *self.edges.entry((node, code)).or_insert_with(|| {
                // Deferred push: the closure runs only when the edge is new,
                // and `next_id` is the id the pushed node will get.
                next_id
            });
            if node == next_id {
                self.terminal.push(false);
            }
        }
        let slot = &mut self.terminal[node as usize];
        if *slot {
            return Some((node, false));
        }
        *slot = true;
        self.len += 1;
        self.max_len = self.max_len.max(char_count);
        Some((node, true))
    }

    /// Returns the trie node a word leads to.
    ///
    /// # Arguments
    /// * `word` - The word to walk, compared exactly as given.
    ///
    /// # Returns
    /// The node reached after the word's last character, or `None` if the
    /// trie has no path for the word. The node ends a registered word only
    /// if it is marked terminal.
    fn node(&self, word: &str) -> Option<u32> {
        let mut node = ROOT;
        for ch in word.chars() {
            node = *self.edges.get(&(node, u32::from(ch)))?;
        }
        Some(node)
    }

    /// Returns whether a word is registered.
    ///
    /// # Arguments
    /// * `word` - The word to look up, compared exactly as given.
    ///
    /// # Returns
    /// `true` if `word` was registered through [`add`](Self::add),
    /// [`add_with_pos`](Self::add_with_pos) or
    /// [`from_words`](Self::from_words).
    #[must_use]
    pub fn contains(&self, word: &str) -> bool {
        self.node(word).is_some_and(|node| self.terminal[node as usize])
    }

    /// Returns the UPOS tag registered for a word (issue #264).
    ///
    /// # Arguments
    /// * `word` - The word to look up, compared exactly as given.
    ///
    /// # Returns
    /// The tag set by [`add_with_pos`](Self::add_with_pos), or `None` if
    /// the word is not registered or was registered without a tag.
    #[must_use]
    pub fn pos(&self, word: &str) -> Option<Upos> {
        if self.tags.is_empty() {
            return None;
        }
        // Only terminal nodes are ever keyed in `tags`.
        self.tags.get(&self.node(word)?).copied()
    }

    /// Returns whether any registered word carries a tag.
    ///
    /// # Returns
    /// `true` after a successful [`add_with_pos`](Self::add_with_pos);
    /// lets `segment_with_pos` skip the tag lookups for an untagged
    /// dictionary.
    pub(crate) fn has_tags(&self) -> bool {
        !self.tags.is_empty()
    }

    /// Returns the number of registered words.
    ///
    /// # Returns
    /// The count of distinct words of two or more characters added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no word is registered.
    ///
    /// # Returns
    /// `true` if [`len`](Self::len) is zero. An empty dictionary leaves
    /// segmentation untouched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the length in characters of the longest registered word.
    ///
    /// # Returns
    /// The maximum character count over the registered words, or 0 when
    /// the dictionary is empty. Bounds the trie walk per position.
    pub(crate) fn max_len(&self) -> usize {
        self.max_len
    }

    /// Sets or clears the flip-ratio gate (see the type documentation).
    ///
    /// # Arguments
    /// * `gate` - `Some(ratio)` with `ratio` in `0.0..=1.0` to apply an
    ///   occurrence only when its flip ratio is at most `ratio`, or `None`
    ///   to apply every occurrence.
    ///
    /// # Returns
    /// `Ok(())` when the gate was set.
    ///
    /// # Errors
    /// Returns [`LitseaError::InvalidInput`] if `ratio` is outside
    /// `0.0..=1.0` or is NaN; the previous gate is kept.
    pub fn set_gate(&mut self, gate: Option<f64>) -> Result<()> {
        if let Some(ratio) = gate
            && !(0.0..=1.0).contains(&ratio)
        {
            return Err(LitseaError::InvalidInput(format!(
                "user dictionary gate must be within 0.0..=1.0, got {ratio}"
            )));
        }
        self.gate = gate;
        Ok(())
    }

    /// Returns the flip-ratio gate.
    ///
    /// # Returns
    /// `Some(ratio)` if a gate is set, `None` if every occurrence is
    /// applied.
    #[must_use]
    pub fn gate(&self) -> Option<f64> {
        self.gate
    }

    /// Enumerates the registered words that start at `start` in `codes`.
    ///
    /// Walks the trie over `codes[start..]`, visiting at most
    /// [`max_len`](Self::max_len) codes, and calls `on_match` with the
    /// exclusive end index of every registered word found, in increasing
    /// length order.
    ///
    /// # Arguments
    /// * `codes` - The sentence as Unicode scalar values (no sentinels).
    /// * `start` - The index in `codes` where matching starts.
    /// * `on_match` - Called as `on_match(end)` for every match
    ///   `codes[start..end]`.
    pub(crate) fn matches_from(
        &self,
        codes: &[u32],
        start: usize,
        mut on_match: impl FnMut(usize),
    ) {
        let limit = codes.len().min(start + self.max_len());
        let mut node = ROOT;
        for (end, &code) in codes.iter().enumerate().take(limit).skip(start) {
            match self.edges.get(&(node, code)) {
                Some(&next) => node = next,
                None => return,
            }
            if self.terminal[node as usize] {
                on_match(end + 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(text: &str) -> Vec<u32> {
        text.chars().map(u32::from).collect()
    }

    fn matches(dictionary: &UserDictionary, text: &str, start: usize) -> Vec<usize> {
        let mut ends = Vec::new();
        dictionary.matches_from(&codes(text), start, |end| ends.push(end));
        ends
    }

    #[test]
    fn test_new_is_empty_without_gate() {
        let dictionary = UserDictionary::new();
        assert!(dictionary.is_empty());
        assert_eq!(dictionary.len(), 0);
        assert_eq!(dictionary.max_len(), 0);
        assert_eq!(dictionary.gate(), None);
        assert!(!dictionary.contains(""));
        assert!(!dictionary.contains("あ"));
        assert_eq!(matches(&dictionary, "あいう", 0), Vec::<usize>::new());
    }

    #[test]
    fn test_default_equals_new() {
        let dictionary = UserDictionary::default();
        assert!(dictionary.is_empty());
        assert_eq!(dictionary.gate(), None);
    }

    #[test]
    fn test_add_registers_words_of_two_or_more_characters() {
        let mut dictionary = UserDictionary::new();
        assert!(dictionary.add("雪だるま"));
        assert!(dictionary.add("新幹線"));
        assert_eq!(dictionary.len(), 2);
        assert_eq!(dictionary.max_len(), 4);
        assert!(dictionary.contains("雪だるま"));
        assert!(dictionary.contains("新幹線"));
        assert!(!dictionary.contains("雪だる")); // prefix only
        assert!(!dictionary.contains("雪だるまを")); // longer
    }

    #[test]
    fn test_add_rejects_short_words_and_duplicates() {
        let mut dictionary = UserDictionary::new();
        assert!(!dictionary.add(""));
        assert!(!dictionary.add("あ"));
        assert!(dictionary.is_empty());
        assert!(dictionary.add("あい"));
        assert!(!dictionary.add("あい"));
        assert_eq!(dictionary.len(), 1);
        assert_eq!(dictionary.max_len(), 2);
    }

    #[test]
    fn test_from_words_skips_short_words() {
        let dictionary = UserDictionary::from_words(["あ", "あい", "", "あいう", "あい"]);
        assert_eq!(dictionary.len(), 2);
        assert!(dictionary.contains("あい"));
        assert!(dictionary.contains("あいう"));
        assert_eq!(dictionary.max_len(), 3);
    }

    #[test]
    fn test_from_words_accepts_owned_strings() {
        let words = vec![String::from("New York"), String::from("ab")];
        let dictionary = UserDictionary::from_words(&words);
        assert!(dictionary.contains("New York"));
        assert!(dictionary.contains("ab"));
        assert_eq!(dictionary.max_len(), 8);
    }

    #[test]
    fn test_add_with_pos_registers_word_and_tag() {
        let mut dictionary = UserDictionary::new();
        assert!(!dictionary.has_tags());
        assert!(dictionary.add_with_pos("チャットGPT", Upos::PROPN));
        assert!(dictionary.has_tags());
        assert!(dictionary.contains("チャットGPT"));
        assert_eq!(dictionary.pos("チャットGPT"), Some(Upos::PROPN));
        assert_eq!(dictionary.len(), 1);
        assert_eq!(dictionary.max_len(), 7);
        // A prefix or an extension of a tagged word is not tagged.
        assert_eq!(dictionary.pos("チャット"), None);
        assert_eq!(dictionary.pos("チャットGPTに"), None);
    }

    #[test]
    fn test_add_with_pos_on_a_registered_word_sets_and_replaces_the_tag() {
        let mut dictionary = UserDictionary::from_words(["雪だるま"]);
        assert_eq!(dictionary.pos("雪だるま"), None);
        // Already registered: not new, but the tag is set...
        assert!(!dictionary.add_with_pos("雪だるま", Upos::NOUN));
        assert_eq!(dictionary.pos("雪だるま"), Some(Upos::NOUN));
        // ...and a later tag replaces it (last wins).
        assert!(!dictionary.add_with_pos("雪だるま", Upos::PROPN));
        assert_eq!(dictionary.pos("雪だるま"), Some(Upos::PROPN));
        assert_eq!(dictionary.len(), 1);
    }

    #[test]
    fn test_add_keeps_an_existing_tag() {
        let mut dictionary = UserDictionary::new();
        assert!(dictionary.add_with_pos("新幹線", Upos::NOUN));
        assert!(!dictionary.add("新幹線"));
        assert_eq!(dictionary.pos("新幹線"), Some(Upos::NOUN));
    }

    #[test]
    fn test_add_with_pos_drops_short_words_with_their_tag() {
        let mut dictionary = UserDictionary::new();
        assert!(!dictionary.add_with_pos("猫", Upos::NOUN));
        assert!(!dictionary.add_with_pos("", Upos::NOUN));
        assert!(dictionary.is_empty());
        assert!(!dictionary.has_tags());
        assert_eq!(dictionary.pos("猫"), None);
    }

    #[test]
    fn test_pos_of_untagged_and_unregistered_words() {
        let mut dictionary = UserDictionary::from_words(["あいう"]);
        dictionary.add_with_pos("あい", Upos::VERB);
        // "あいう" is untagged; "あ" is a trie node on the path but not a
        // word; "かき" is not in the trie at all.
        assert_eq!(dictionary.pos("あいう"), None);
        assert_eq!(dictionary.pos("あい"), Some(Upos::VERB));
        assert_eq!(dictionary.pos("あ"), None);
        assert_eq!(dictionary.pos("かき"), None);
        assert_eq!(dictionary.pos(""), None);
    }

    #[test]
    fn test_set_gate_accepts_the_closed_unit_interval() {
        let mut dictionary = UserDictionary::new();
        for ratio in [0.0, 0.5, 1.0] {
            dictionary.set_gate(Some(ratio)).unwrap();
            assert_eq!(dictionary.gate(), Some(ratio));
        }
        dictionary.set_gate(None).unwrap();
        assert_eq!(dictionary.gate(), None);
    }

    #[test]
    fn test_set_gate_rejects_out_of_range_and_nan() {
        let mut dictionary = UserDictionary::new();
        dictionary.set_gate(Some(0.5)).unwrap();
        for ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let error = dictionary.set_gate(Some(ratio)).unwrap_err();
            assert!(
                matches!(error, LitseaError::InvalidInput(_)),
                "unexpected error for {ratio}: {error:?}"
            );
            assert!(error.to_string().contains("0.0..=1.0"));
            // The previous gate is kept.
            assert_eq!(dictionary.gate(), Some(0.5));
        }
    }

    #[test]
    fn test_matches_from_reports_every_registered_prefix() {
        let dictionary = UserDictionary::from_words(["雪だ", "雪だるま", "だるま"]);
        assert_eq!(matches(&dictionary, "雪だるまを作る", 0), vec![2, 4]);
        assert_eq!(matches(&dictionary, "雪だるまを作る", 1), vec![4]);
        assert_eq!(matches(&dictionary, "雪だるまを作る", 2), Vec::<usize>::new());
        // A partial occurrence at the end of the input is not a match.
        assert_eq!(matches(&dictionary, "雪だる", 0), vec![2]);
        assert_eq!(matches(&dictionary, "雪だる", 1), Vec::<usize>::new());
    }

    #[test]
    fn test_matches_from_is_bounded_by_max_len() {
        let dictionary = UserDictionary::from_words(["ab"]);
        // "abab": the walk from 0 stops after 2 codes even though "aba..."
        // continues; only the 2-character word is reported.
        assert_eq!(matches(&dictionary, "abab", 0), vec![2]);
        assert_eq!(matches(&dictionary, "abab", 2), vec![4]);
        assert_eq!(matches(&dictionary, "abab", 4), Vec::<usize>::new());
    }

    #[test]
    fn test_matches_from_matches_spaces_literally() {
        let dictionary = UserDictionary::from_words(["New York"]);
        assert_eq!(matches(&dictionary, "in New York", 3), vec![11]);
        assert_eq!(matches(&dictionary, "in NewYork", 3), Vec::<usize>::new());
    }

    #[test]
    fn test_clone_and_debug() {
        let mut dictionary = UserDictionary::from_words(["あい"]);
        dictionary.add_with_pos("うえ", Upos::ADJ);
        dictionary.set_gate(Some(0.25)).unwrap();
        let cloned = dictionary.clone();
        assert!(cloned.contains("あい"));
        assert_eq!(cloned.pos("うえ"), Some(Upos::ADJ));
        assert_eq!(cloned.gate(), Some(0.25));
        let debug = format!("{dictionary:?}");
        assert!(debug.starts_with("UserDictionary"));
        assert!(debug.contains("gate: Some(0.25)"));
        assert!(debug.contains("ADJ"));
    }
}
