use std::fmt;
use std::str::FromStr;

/// Universal POS (UPOS) タグ。
/// Universal Dependencies で定義された17品詞分類。
/// <https://universaldependencies.org/u/pos/>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Upos {
    /// 形容詞 (Adjective)
    ADJ,
    /// 接置詞 (Adposition)
    ADP,
    /// 副詞 (Adverb)
    ADV,
    /// 助動詞 (Auxiliary)
    AUX,
    /// 等位接続詞 (Coordinating conjunction)
    CCONJ,
    /// 限定詞 (Determiner)
    DET,
    /// 間投詞 (Interjection)
    INTJ,
    /// 名詞 (Noun)
    NOUN,
    /// 数詞 (Numeral)
    NUM,
    /// 助詞・小辞 (Particle)
    PART,
    /// 代名詞 (Pronoun)
    PRON,
    /// 固有名詞 (Proper noun)
    PROPN,
    /// 句読点 (Punctuation)
    PUNCT,
    /// 従属接続詞 (Subordinating conjunction)
    SCONJ,
    /// 記号 (Symbol)
    SYM,
    /// 動詞 (Verb)
    VERB,
    /// その他 (Other)
    X,
}

impl Upos {
    /// 全UPOSタグの一覧を返す。
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

/// セグメンテーション+品詞の複合ラベル。
/// 各文字位置に割り当てられる。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SegmentLabel {
    /// 単語の先頭文字（境界）。品詞情報を持つ。
    B(Upos),
    /// 単語の継続文字（非境界）。
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
    /// 全ラベルの一覧を返す（B-ADJ, B-ADP, ..., B-X, O）。
    pub fn all_labels() -> Vec<SegmentLabel> {
        let mut labels: Vec<SegmentLabel> =
            Upos::ALL.iter().map(|&pos| SegmentLabel::B(pos)).collect();
        labels.push(SegmentLabel::O);
        labels
    }

    /// 境界ラベルかどうかを返す。
    pub fn is_boundary(&self) -> bool {
        matches!(self, SegmentLabel::B(_))
    }

    /// 品詞タグを返す。非境界（O）の場合はNone。
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
        assert!("noun".parse::<Upos>().is_err()); // 大文字のみ
        assert!("".parse::<Upos>().is_err());
    }

    #[test]
    fn test_segment_label_display_and_parse() {
        // B-品詞ラベル
        let label = SegmentLabel::B(Upos::NOUN);
        assert_eq!(label.to_string(), "B-NOUN");
        assert_eq!("B-NOUN".parse::<SegmentLabel>().unwrap(), label);

        // Oラベル
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
        // 17品詞のB + O = 18ラベル
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
