//! User dictionary: words the segmenter must emit as single tokens (issue
//! #263).
//!
//! A [`UserDictionary`] stores registered words in a trie keyed by Unicode
//! scalar values, the same `u32` codes that `Segmenter::segment_into` packs
//! into its context buffer, so matching walks the packed codes directly
//! without decoding the sentence again. The matching semantics (both edges
//! of an occurrence forced to be boundaries, no boundary inside it,
//! leftmost-longest resolution of overlapping occurrences, and the optional
//! gate) are documented on [`UserDictionary`] and implemented in
//! `crate::segmenter`.

use rustc_hash::FxHashMap;

use crate::error::{LitseaError, Result};

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
///   ([`add`](Self::add) returns `false` for them).
/// - The dictionary carries no part-of-speech information. In
///   `segment_with_pos`, a forced word that the lexicon does not know is
///   tagged by the stage-2 tagger's full argmax.
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
/// use litsea::user_dictionary::UserDictionary;
///
/// let mut dictionary = UserDictionary::from_words(["雪だるま", "新幹線"]);
/// assert_eq!(dictionary.len(), 2);
/// assert!(dictionary.contains("新幹線"));
/// assert!(!dictionary.add("あ")); // one character: not stored
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
    /// # Arguments
    /// * `word` - The word to register, matched later exactly as given.
    ///
    /// # Returns
    /// `true` if the word was newly registered; `false` if it has fewer
    /// than two characters (there is no inside to keep together) or was
    /// already registered.
    pub fn add(&mut self, word: &str) -> bool {
        let char_count = word.chars().count();
        if char_count < 2 {
            return false;
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
            return false;
        }
        *slot = true;
        self.len += 1;
        self.max_len = self.max_len.max(char_count);
        true
    }

    /// Returns whether a word is registered.
    ///
    /// # Arguments
    /// * `word` - The word to look up, compared exactly as given.
    ///
    /// # Returns
    /// `true` if `word` was registered through [`add`](Self::add) or
    /// [`from_words`](Self::from_words).
    #[must_use]
    pub fn contains(&self, word: &str) -> bool {
        let mut node = ROOT;
        for ch in word.chars() {
            match self.edges.get(&(node, u32::from(ch))) {
                Some(&next) => node = next,
                None => return false,
            }
        }
        self.terminal[node as usize]
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
        dictionary.set_gate(Some(0.25)).unwrap();
        let cloned = dictionary.clone();
        assert!(cloned.contains("あい"));
        assert_eq!(cloned.gate(), Some(0.25));
        let debug = format!("{dictionary:?}");
        assert!(debug.starts_with("UserDictionary"));
        assert!(debug.contains("gate: Some(0.25)"));
    }
}
