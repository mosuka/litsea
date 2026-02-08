use std::fmt;
use std::str::FromStr;

use regex::Regex;

/// Supported languages for word segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    /// Japanese (日本語)
    Japanese,
    /// Chinese (中文) - covers both Simplified and Traditional
    Chinese,
    /// Korean (한국어)
    Korean,
}

impl Default for Language {
    fn default() -> Self {
        Language::Japanese
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::Japanese => write!(f, "japanese"),
            Language::Chinese => write!(f, "chinese"),
            Language::Korean => write!(f, "korean"),
        }
    }
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "japanese" | "ja" => Ok(Language::Japanese),
            "chinese" | "zh" => Ok(Language::Chinese),
            "korean" | "ko" => Ok(Language::Korean),
            _ => Err(format!(
                "Unsupported language: '{}'. Supported: japanese (ja), chinese (zh), korean (ko)",
                s
            )),
        }
    }
}

impl Language {
    /// Creates the character type patterns for this language.
    pub fn char_type_patterns(&self) -> CharTypePatterns {
        match self {
            Language::Japanese => japanese_patterns(),
            Language::Chinese => chinese_patterns(),
            Language::Korean => korean_patterns(),
        }
    }
}

/// Character type classification patterns for a specific language.
/// Each pattern maps a regex to a type code string.
pub struct CharTypePatterns {
    patterns: Vec<(Regex, &'static str)>,
}

impl CharTypePatterns {
    /// Creates a new instance of [`CharTypePatterns`].
    pub fn new(patterns: Vec<(Regex, &'static str)>) -> Self {
        CharTypePatterns { patterns }
    }

    /// Gets the type of a character based on the language-specific patterns.
    ///
    /// # Arguments
    /// * `ch` - A string slice representing a single character.
    ///
    /// # Returns
    /// A string slice representing the type code of the character.
    /// Returns "O" (Other) if the character does not match any pattern.
    pub fn get_type(&self, ch: &str) -> &str {
        for (pattern, label) in &self.patterns {
            if pattern.is_match(ch) {
                return label;
            }
        }
        "O" // Other
    }
}

/// Creates character type patterns for Japanese.
///
/// Type codes:
/// - "M": Kanji numbers (一二三四五六七八九十百千万億兆)
/// - "H": Kanji (CJK ideographs)
/// - "I": Hiragana
/// - "K": Katakana
/// - "A": ASCII and full-width Latin characters
/// - "N": Digits (ASCII and full-width)
/// - "O": Other (fallback)
fn japanese_patterns() -> CharTypePatterns {
    CharTypePatterns::new(vec![
        (Regex::new(r"[一二三四五六七八九十百千万億兆]").unwrap(), "M"),
        (Regex::new(r"[一-龠々〆ヵヶ]").unwrap(), "H"),
        (Regex::new(r"[ぁ-ん]").unwrap(), "I"),
        (Regex::new(r"[ァ-ヴーｱ-ﾝﾞﾟ]").unwrap(), "K"),
        (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
        (Regex::new(r"[0-9０-９]").unwrap(), "N"),
    ])
}

/// Creates character type patterns for Chinese.
///
/// Type codes:
/// - "C": CJK Unified Ideographs (U+4E00..U+9FFF)
/// - "X": CJK Extension A (U+3400..U+4DBF)
/// - "R": CJK Radicals and Kangxi Radicals (U+2E80..U+2FDF)
/// - "P": Chinese punctuation and CJK symbols
/// - "B": Bopomofo (Zhuyin)
/// - "A": ASCII and full-width Latin characters
/// - "N": Digits (ASCII and full-width)
/// - "O": Other (fallback)
fn chinese_patterns() -> CharTypePatterns {
    CharTypePatterns::new(vec![
        // CJK Unified Ideographs
        (Regex::new(r"[\u{4E00}-\u{9FFF}]").unwrap(), "C"),
        // CJK Extension A
        (Regex::new(r"[\u{3400}-\u{4DBF}]").unwrap(), "X"),
        // CJK Radicals Supplement + Kangxi Radicals
        (Regex::new(r"[\u{2E80}-\u{2FDF}]").unwrap(), "R"),
        // Chinese punctuation: CJK Symbols and Punctuation + full-width punctuation
        (
            Regex::new(
                r"[\u{3000}-\u{303F}\u{FF01}-\u{FF0F}\u{FF1A}-\u{FF20}\u{FF3B}-\u{FF40}\u{FF5B}-\u{FF65}]",
            )
            .unwrap(),
            "P",
        ),
        // Bopomofo + Bopomofo Extended
        (Regex::new(r"[\u{3100}-\u{312F}\u{31A0}-\u{31BF}]").unwrap(), "B"),
        // ASCII + Full-width Latin
        (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
        // Numbers
        (Regex::new(r"[0-9０-９]").unwrap(), "N"),
    ])
}

/// Creates character type patterns for Korean.
///
/// Type codes:
/// - "S": Hangul Syllables (U+AC00..U+D7AF)
/// - "J": Hangul Jamo (U+1100..U+11FF)
/// - "G": Hangul Compatibility Jamo (U+3130..U+318F)
/// - "H": Hanja / CJK Ideographs (U+4E00..U+9FFF)
/// - "P": Korean punctuation and CJK symbols
/// - "A": ASCII and full-width Latin characters
/// - "N": Digits (ASCII and full-width)
/// - "O": Other (fallback)
fn korean_patterns() -> CharTypePatterns {
    CharTypePatterns::new(vec![
        // Hangul Syllables
        (Regex::new(r"[\u{AC00}-\u{D7AF}]").unwrap(), "S"),
        // Hangul Jamo
        (Regex::new(r"[\u{1100}-\u{11FF}]").unwrap(), "J"),
        // Hangul Compatibility Jamo
        (Regex::new(r"[\u{3130}-\u{318F}]").unwrap(), "G"),
        // Hanja (CJK Unified Ideographs)
        (Regex::new(r"[\u{4E00}-\u{9FFF}]").unwrap(), "H"),
        // Korean punctuation: CJK Symbols and Punctuation + full-width punctuation
        (
            Regex::new(
                r"[\u{3000}-\u{303F}\u{FF01}-\u{FF0F}\u{FF1A}-\u{FF20}]",
            )
            .unwrap(),
            "P",
        ),
        // ASCII + Full-width Latin
        (Regex::new(r"[a-zA-Zａ-ｚＡ-Ｚ]").unwrap(), "A"),
        // Numbers
        (Regex::new(r"[0-9０-９]").unwrap(), "N"),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Language enum tests ---

    #[test]
    fn test_language_from_str() {
        assert_eq!("japanese".parse::<Language>().unwrap(), Language::Japanese);
        assert_eq!("ja".parse::<Language>().unwrap(), Language::Japanese);
        assert_eq!("Japanese".parse::<Language>().unwrap(), Language::Japanese);
        assert_eq!("chinese".parse::<Language>().unwrap(), Language::Chinese);
        assert_eq!("zh".parse::<Language>().unwrap(), Language::Chinese);
        assert_eq!("Chinese".parse::<Language>().unwrap(), Language::Chinese);
        assert_eq!("korean".parse::<Language>().unwrap(), Language::Korean);
        assert_eq!("ko".parse::<Language>().unwrap(), Language::Korean);
        assert_eq!("KOREAN".parse::<Language>().unwrap(), Language::Korean);
        assert!("french".parse::<Language>().is_err());
        assert!("".parse::<Language>().is_err());
    }

    #[test]
    fn test_language_display() {
        assert_eq!(Language::Japanese.to_string(), "japanese");
        assert_eq!(Language::Chinese.to_string(), "chinese");
        assert_eq!(Language::Korean.to_string(), "korean");
    }

    #[test]
    fn test_language_default() {
        assert_eq!(Language::default(), Language::Japanese);
    }

    // --- Japanese pattern tests ---

    #[test]
    fn test_japanese_patterns() {
        let p = Language::Japanese.char_type_patterns();
        assert_eq!(p.get_type("三"), "M"); // Kanji number
        assert_eq!(p.get_type("漢"), "H"); // Kanji
        assert_eq!(p.get_type("あ"), "I"); // Hiragana
        assert_eq!(p.get_type("ア"), "K"); // Katakana
        assert_eq!(p.get_type("A"), "A"); // ASCII
        assert_eq!(p.get_type("ａ"), "A"); // Full-width Latin
        assert_eq!(p.get_type("5"), "N"); // Digit
        assert_eq!(p.get_type("５"), "N"); // Full-width digit
        assert_eq!(p.get_type("@"), "O"); // Other
    }

    // --- Chinese pattern tests ---

    #[test]
    fn test_chinese_patterns() {
        let p = Language::Chinese.char_type_patterns();
        assert_eq!(p.get_type("中"), "C"); // CJK Unified
        assert_eq!(p.get_type("国"), "C"); // CJK Unified
        assert_eq!(p.get_type("人"), "C"); // CJK Unified
        assert_eq!(p.get_type("。"), "P"); // Chinese punctuation (U+3002)
        assert_eq!(p.get_type("，"), "P"); // Full-width comma (U+FF0C)
        assert_eq!(p.get_type("A"), "A"); // ASCII
        assert_eq!(p.get_type("5"), "N"); // Digit
        assert_eq!(p.get_type("@"), "O"); // Other
    }

    // --- Korean pattern tests ---

    #[test]
    fn test_korean_patterns() {
        let p = Language::Korean.char_type_patterns();
        assert_eq!(p.get_type("한"), "S"); // Hangul Syllable
        assert_eq!(p.get_type("글"), "S"); // Hangul Syllable
        assert_eq!(p.get_type("ㄱ"), "G"); // Compatibility Jamo
        assert_eq!(p.get_type("ㅏ"), "G"); // Compatibility Jamo (vowel)
        assert_eq!(p.get_type("漢"), "H"); // Hanja
        assert_eq!(p.get_type("。"), "P"); // Punctuation (U+3002)
        assert_eq!(p.get_type("A"), "A"); // ASCII
        assert_eq!(p.get_type("5"), "N"); // Digit
        assert_eq!(p.get_type("@"), "O"); // Other
    }
}
