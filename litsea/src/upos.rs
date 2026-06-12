use std::fmt;
use std::str::FromStr;

/// Universal POS (UPOS) tag.
/// The 17 part-of-speech categories defined by Universal Dependencies.
/// <https://universaldependencies.org/u/pos/>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Upos {
    /// Adjective
    ADJ,
    /// Adposition
    ADP,
    /// Adverb
    ADV,
    /// Auxiliary
    AUX,
    /// Coordinating conjunction
    CCONJ,
    /// Determiner
    DET,
    /// Interjection
    INTJ,
    /// Noun
    NOUN,
    /// Numeral
    NUM,
    /// Particle
    PART,
    /// Pronoun
    PRON,
    /// Proper noun
    PROPN,
    /// Punctuation
    PUNCT,
    /// Subordinating conjunction
    SCONJ,
    /// Symbol
    SYM,
    /// Verb
    VERB,
    /// Other
    X,
}

impl Upos {
    /// All UPOS tags.
    pub const ALL: [Upos; 17] = [
        Upos::ADJ,
        Upos::ADP,
        Upos::ADV,
        Upos::AUX,
        Upos::CCONJ,
        Upos::DET,
        Upos::INTJ,
        Upos::NOUN,
        Upos::NUM,
        Upos::PART,
        Upos::PRON,
        Upos::PROPN,
        Upos::PUNCT,
        Upos::SCONJ,
        Upos::SYM,
        Upos::VERB,
        Upos::X,
    ];
}

impl fmt::Display for Upos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Upos::ADJ => "ADJ",
            Upos::ADP => "ADP",
            Upos::ADV => "ADV",
            Upos::AUX => "AUX",
            Upos::CCONJ => "CCONJ",
            Upos::DET => "DET",
            Upos::INTJ => "INTJ",
            Upos::NOUN => "NOUN",
            Upos::NUM => "NUM",
            Upos::PART => "PART",
            Upos::PRON => "PRON",
            Upos::PROPN => "PROPN",
            Upos::PUNCT => "PUNCT",
            Upos::SCONJ => "SCONJ",
            Upos::SYM => "SYM",
            Upos::VERB => "VERB",
            Upos::X => "X",
        };
        f.write_str(s)
    }
}

impl FromStr for Upos {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ADJ" => Ok(Upos::ADJ),
            "ADP" => Ok(Upos::ADP),
            "ADV" => Ok(Upos::ADV),
            "AUX" => Ok(Upos::AUX),
            "CCONJ" => Ok(Upos::CCONJ),
            "DET" => Ok(Upos::DET),
            "INTJ" => Ok(Upos::INTJ),
            "NOUN" => Ok(Upos::NOUN),
            "NUM" => Ok(Upos::NUM),
            "PART" => Ok(Upos::PART),
            "PRON" => Ok(Upos::PRON),
            "PROPN" => Ok(Upos::PROPN),
            "PUNCT" => Ok(Upos::PUNCT),
            "SCONJ" => Ok(Upos::SCONJ),
            "SYM" => Ok(Upos::SYM),
            "VERB" => Ok(Upos::VERB),
            "X" => Ok(Upos::X),
            _ => Err(format!("Unknown UPOS tag: '{}'", s)),
        }
    }
}

/// Combined segmentation + POS label assigned to each character position.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    /// First character of a word (boundary), carrying the POS tag.
    B(Upos),
    /// Continuation character of a word (non-boundary).
    O,
}

impl fmt::Display for SegmentLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentLabel::B(pos) => write!(f, "B-{}", pos),
            SegmentLabel::O => f.write_str("O"),
        }
    }
}

impl FromStr for SegmentLabel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "O" {
            Ok(SegmentLabel::O)
        } else if let Some(pos_str) = s.strip_prefix("B-") {
            let pos: Upos = pos_str.parse()?;
            Ok(SegmentLabel::B(pos))
        } else {
            Err(format!("Invalid segment label: '{}'. Expected 'O' or 'B-<UPOS>'", s))
        }
    }
}

impl SegmentLabel {
    /// Returns all labels (B-ADJ, B-ADP, ..., B-X, O).
    pub fn all_labels() -> Vec<SegmentLabel> {
        let mut labels: Vec<SegmentLabel> =
            Upos::ALL.iter().map(|&pos| SegmentLabel::B(pos)).collect();
        labels.push(SegmentLabel::O);
        labels
    }

    /// Returns true if this is a boundary label.
    pub fn is_boundary(&self) -> bool {
        matches!(self, SegmentLabel::B(_))
    }

    /// Returns the POS tag, or None for the non-boundary label (O).
    pub fn pos(&self) -> Option<Upos> {
        match self {
            SegmentLabel::B(pos) => Some(*pos),
            SegmentLabel::O => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upos_display_and_parse() {
        for &pos in &Upos::ALL {
            let s = pos.to_string();
            let parsed: Upos = s.parse().unwrap();
            assert_eq!(parsed, pos);
        }
    }

    #[test]
    fn test_upos_parse_error() {
        assert!("UNKNOWN".parse::<Upos>().is_err());
        assert!("noun".parse::<Upos>().is_err()); // upper case only
        assert!("".parse::<Upos>().is_err());
    }

    #[test]
    fn test_segment_label_display_and_parse() {
        // B-<POS> label
        let label = SegmentLabel::B(Upos::NOUN);
        assert_eq!(label.to_string(), "B-NOUN");
        assert_eq!("B-NOUN".parse::<SegmentLabel>().unwrap(), label);

        // O label
        let label_o = SegmentLabel::O;
        assert_eq!(label_o.to_string(), "O");
        assert_eq!("O".parse::<SegmentLabel>().unwrap(), label_o);
    }

    #[test]
    fn test_segment_label_parse_error() {
        assert!("B-UNKNOWN".parse::<SegmentLabel>().is_err());
        assert!("I-NOUN".parse::<SegmentLabel>().is_err());
        assert!("".parse::<SegmentLabel>().is_err());
    }

    #[test]
    fn test_all_labels() {
        let labels = SegmentLabel::all_labels();
        // 17 B-<POS> labels + O = 18 labels
        assert_eq!(labels.len(), 18);
        assert!(labels.last().unwrap() == &SegmentLabel::O);
        assert!(labels[0].is_boundary());
        assert!(!labels.last().unwrap().is_boundary());
    }

    #[test]
    fn test_segment_label_pos() {
        assert_eq!(SegmentLabel::B(Upos::VERB).pos(), Some(Upos::VERB));
        assert_eq!(SegmentLabel::O.pos(), None);
    }

    #[test]
    fn test_upos_all_count() {
        assert_eq!(Upos::ALL.len(), 17);
    }
}
