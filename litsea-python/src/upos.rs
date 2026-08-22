//! The `Upos` enum and the `Token` class.

use litsea::Upos;
use litsea_binding_core::TokenView;
use pyo3::prelude::*;

/// A Universal POS tag.
///
/// Exposed to Python as `litsea.Upos`, with the 17 UD tags as members
/// (`Upos.NOUN`, `Upos.VERB`, …).
// See the note on `Language`: no `eq_int`, so tags never compare equal to ints.
#[pyclass(name = "Upos", eq, frozen, hash, from_py_object, module = "litsea")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyUpos {
    /// Adjective.
    ADJ,
    /// Adposition.
    ADP,
    /// Adverb.
    ADV,
    /// Auxiliary.
    AUX,
    /// Coordinating conjunction.
    CCONJ,
    /// Determiner.
    DET,
    /// Interjection.
    INTJ,
    /// Noun.
    NOUN,
    /// Numeral.
    NUM,
    /// Particle.
    PART,
    /// Pronoun.
    PRON,
    /// Proper noun.
    PROPN,
    /// Punctuation.
    PUNCT,
    /// Subordinating conjunction.
    SCONJ,
    /// Symbol.
    SYM,
    /// Verb.
    VERB,
    /// Other.
    X,
}

#[pymethods]
impl PyUpos {
    /// Returns the tag name, for example `"NOUN"`.
    ///
    /// # Returns
    /// The canonical uppercase UPOS tag name.
    #[getter]
    fn name(&self) -> String {
        Upos::from(*self).to_string()
    }

    /// Returns every UPOS tag.
    ///
    /// # Returns
    /// All 17 tags, in UD order.
    #[staticmethod]
    fn all() -> Vec<Self> {
        Upos::ALL.into_iter().map(Self::from).collect()
    }

    /// Returns the tag name.
    ///
    /// # Returns
    /// The same value as the `name` property.
    fn __str__(&self) -> String {
        self.name()
    }
}

impl From<PyUpos> for Upos {
    /// Converts the Python-facing tag into `litsea`'s tag.
    ///
    /// # Arguments
    /// * `pos` - The Python-facing tag.
    ///
    /// # Returns
    /// The corresponding [`Upos`].
    fn from(pos: PyUpos) -> Self {
        match pos {
            PyUpos::ADJ => Upos::ADJ,
            PyUpos::ADP => Upos::ADP,
            PyUpos::ADV => Upos::ADV,
            PyUpos::AUX => Upos::AUX,
            PyUpos::CCONJ => Upos::CCONJ,
            PyUpos::DET => Upos::DET,
            PyUpos::INTJ => Upos::INTJ,
            PyUpos::NOUN => Upos::NOUN,
            PyUpos::NUM => Upos::NUM,
            PyUpos::PART => Upos::PART,
            PyUpos::PRON => Upos::PRON,
            PyUpos::PROPN => Upos::PROPN,
            PyUpos::PUNCT => Upos::PUNCT,
            PyUpos::SCONJ => Upos::SCONJ,
            PyUpos::SYM => Upos::SYM,
            PyUpos::VERB => Upos::VERB,
            PyUpos::X => Upos::X,
        }
    }
}

impl From<Upos> for PyUpos {
    /// Converts `litsea`'s tag into the Python-facing tag.
    ///
    /// # Arguments
    /// * `pos` - The `litsea` tag.
    ///
    /// # Returns
    /// The corresponding [`PyUpos`].
    fn from(pos: Upos) -> Self {
        match pos {
            Upos::ADJ => PyUpos::ADJ,
            Upos::ADP => PyUpos::ADP,
            Upos::ADV => PyUpos::ADV,
            Upos::AUX => PyUpos::AUX,
            Upos::CCONJ => PyUpos::CCONJ,
            Upos::DET => PyUpos::DET,
            Upos::INTJ => PyUpos::INTJ,
            Upos::NOUN => PyUpos::NOUN,
            Upos::NUM => PyUpos::NUM,
            Upos::PART => PyUpos::PART,
            Upos::PRON => PyUpos::PRON,
            Upos::PROPN => PyUpos::PROPN,
            Upos::PUNCT => PyUpos::PUNCT,
            Upos::SCONJ => PyUpos::SCONJ,
            Upos::SYM => PyUpos::SYM,
            Upos::VERB => PyUpos::VERB,
            Upos::X => PyUpos::X,
        }
    }
}

/// A segmented token.
///
/// `start` and `end` are byte offsets into the input string, so
/// `text.encode()[token.start:token.end].decode()` is `token.surface`.
/// They are exact for both segmentation and POS output.
#[pyclass(name = "Token", frozen, skip_from_py_object, module = "litsea")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyToken {
    /// The token's surface form.
    #[pyo3(get)]
    surface: String,
    /// The UPOS tag, or `None` for segmentation-only output.
    #[pyo3(get)]
    pos: Option<PyUpos>,
    /// Starting byte offset in the input string.
    #[pyo3(get)]
    start: usize,
    /// Ending byte offset (exclusive) in the input string.
    #[pyo3(get)]
    end: usize,
}

#[pymethods]
impl PyToken {
    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `Token('すもも', pos=NOUN, start=0, end=9)`.
    fn __repr__(&self) -> String {
        match self.pos {
            Some(pos) => format!(
                "Token({:?}, pos={}, start={}, end={})",
                self.surface,
                pos.name(),
                self.start,
                self.end
            ),
            None => format!("Token({:?}, start={}, end={})", self.surface, self.start, self.end),
        }
    }

    /// Compares two tokens by all four fields.
    ///
    /// # Arguments
    /// * `other` - The token to compare against.
    ///
    /// # Returns
    /// `true` when every field is equal.
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl From<TokenView> for PyToken {
    /// Converts a core token view into the Python-facing token.
    ///
    /// # Arguments
    /// * `view` - The token view.
    ///
    /// # Returns
    /// The corresponding [`PyToken`].
    fn from(view: TokenView) -> Self {
        Self {
            surface: view.surface,
            pos: view.pos.map(PyUpos::from),
            start: view.byte_start,
            end: view.byte_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_upos_tag_round_trips() {
        for pos in Upos::ALL {
            let member = PyUpos::from(pos);
            assert_eq!(Upos::from(member), pos);
            assert_eq!(member.name(), pos.to_string());
        }
        assert_eq!(PyUpos::all().len(), 17);
    }

    #[test]
    fn test_token_conversion_keeps_offsets() {
        let view = TokenView::new("すもも", 3, 12, Some(Upos::NOUN));
        let token = PyToken::from(view);
        assert_eq!(token.surface, "すもも");
        assert_eq!(token.start, 3);
        assert_eq!(token.end, 12);
        assert_eq!(token.pos, Some(PyUpos::NOUN));
        assert!(token.__repr__().contains("pos=NOUN"));
    }
}
