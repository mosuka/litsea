use std::fmt;
use std::str::FromStr;

/// Supported languages for word segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Language {
    /// Japanese (日本語)
    #[default]
    Japanese,
    /// Chinese (中文) - covers both Simplified and Traditional
    Chinese,
    /// Korean (한국어)
    Korean,
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
    /// Classifies a character into a language-specific type code.
    ///
    /// Returns "O" (Other) if the character does not belong to any class.
    /// Classification is a direct `match` on character ranges, so it is
    /// allocation-free and O(1).
    #[must_use]
    pub fn char_type(&self, c: char) -> &'static str {
        match self {
            Language::Japanese => japanese_char_type(c),
            Language::Chinese => chinese_char_type(c),
            Language::Korean => korean_char_type(c),
        }
    }
}

/// Classes shared by all languages, checked after the language-specific ones:
/// - "P": CJK Symbols and Punctuation + full-width punctuation
/// - "A": ASCII and full-width Latin characters
/// - "N": Digits (ASCII and full-width)
fn punct_latin_digit(c: char) -> Option<&'static str> {
    match c {
        '\u{3000}'..='\u{303F}'
        | '\u{FF01}'..='\u{FF0F}'
        | '\u{FF1A}'..='\u{FF20}'
        | '\u{FF3B}'..='\u{FF40}'
        | '\u{FF5B}'..='\u{FF65}' => Some("P"),
        'a'..='z' | 'A'..='Z' | 'ａ'..='ｚ' | 'Ａ'..='Ｚ' => Some("A"),
        '0'..='9' | '０'..='９' => Some("N"),
        _ => None,
    }
}

/// Character type classification for Japanese.
///
/// Type codes:
/// - "M": Kanji numbers (一二三四五六七八九十百千万億兆)
/// - "H": Kanji (CJK ideographs, 々〆ヵヶ)
/// - "I": Hiragana
/// - "K": Katakana (full-width and half-width)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O": Other (fallback)
fn japanese_char_type(c: char) -> &'static str {
    match c {
        '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千'
        | '万' | '億' | '兆' => "M",
        // 一-龠 plus 々〆ヵヶ
        '\u{4E00}'..='\u{9FA0}' | '々' | '〆' | 'ヵ' | 'ヶ' => "H",
        // ぁ-ん
        '\u{3041}'..='\u{3093}' => "I",
        // ァ-ヴ, ー, half-width ｱ-ﾝ and ﾞﾟ
        '\u{30A1}'..='\u{30F4}' | 'ー' | '\u{FF71}'..='\u{FF9D}' | 'ﾞ' | 'ﾟ' => "K",
        _ => punct_latin_digit(c).unwrap_or("O"),
    }
}

/// Character type classification for Chinese.
///
/// Type codes:
/// - "F": High-frequency function words (虚词: 的了在是和不也 etc.)
/// - "C": CJK Unified Ideographs (U+4E00..U+9FFF)
/// - "X": CJK Extension A (U+3400..U+4DBF)
/// - "R": CJK Radicals and Kangxi Radicals (U+2E80..U+2FDF)
/// - "B": Bopomofo (Zhuyin)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O": Other (fallback)
fn chinese_char_type(c: char) -> &'static str {
    match c {
        // High-frequency function words (虚词): structural particles,
        // aspect/modal particles, conjunctions, prepositions, and common
        // grammatical verbs/adverbs
        '的' | '地' | '得' | '了' | '着' | '过' | '吗' | '呢' | '吧' | '啊' | '嘛' | '和'
        | '与' | '或' | '但' | '而' | '且' | '及' | '在' | '从' | '到' | '把' | '被' | '对'
        | '向' | '给' | '是' | '有' | '不' | '也' | '都' | '就' | '要' | '会' | '能' | '可' => {
            "F"
        }
        '\u{4E00}'..='\u{9FFF}' => "C",
        '\u{3400}'..='\u{4DBF}' => "X",
        '\u{2E80}'..='\u{2FDF}' => "R",
        // Bopomofo + Bopomofo Extended
        '\u{3100}'..='\u{312F}' | '\u{31A0}'..='\u{31BF}' => "B",
        _ => punct_latin_digit(c).unwrap_or("O"),
    }
}

/// Character type classification for Korean.
///
/// Type codes:
/// - "E": High-frequency particles/endings (조사/어미: 은는을를의에)
/// - "SN": Hangul Syllable without 받침 (e.g., 가, 나, 하)
/// - "SF": Hangul Syllable with 받침 (e.g., 한, 글, 각)
/// - "J": Hangul Jamo (U+1100..U+11FF)
/// - "G": Hangul Compatibility Jamo (U+3130..U+318F)
/// - "H": Hanja / CJK Ideographs (U+4E00..U+9FFF)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O": Other (fallback)
fn korean_char_type(c: char) -> &'static str {
    match c {
        // Overwhelmingly used as grammatical particles:
        // 은/는 (topic), 을/를 (object), 의 (possessive), 에 (locative)
        '은' | '는' | '을' | '를' | '의' | '에' => "E",
        // Hangul Syllables: (codepoint - 0xAC00) % 28 == 0 means no 받침
        // (final consonant)
        '\u{AC00}'..='\u{D7AF}' => {
            if (c as u32 - 0xAC00).is_multiple_of(28) {
                "SN"
            } else {
                "SF"
            }
        }
        '\u{1100}'..='\u{11FF}' => "J",
        '\u{3130}'..='\u{318F}' => "G",
        '\u{4E00}'..='\u{9FFF}' => "H",
        _ => punct_latin_digit(c).unwrap_or("O"),
    }
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
    fn test_japanese_char_types() {
        let lang = Language::Japanese;
        assert_eq!(lang.char_type('三'), "M"); // Kanji number
        assert_eq!(lang.char_type('千'), "M"); // Kanji number (boundary)
        assert_eq!(lang.char_type('万'), "M"); // Kanji number (large unit)
        assert_eq!(lang.char_type('億'), "M"); // Kanji number (large unit)
        assert_eq!(lang.char_type('漢'), "H"); // Kanji
        assert_eq!(lang.char_type('々'), "H"); // Iteration mark
        assert_eq!(lang.char_type('あ'), "I"); // Hiragana
        assert_eq!(lang.char_type('ア'), "K"); // Katakana
        assert_eq!(lang.char_type('ー'), "K"); // Prolonged sound mark
        assert_eq!(lang.char_type('ｱ'), "K"); // Half-width Katakana
        assert_eq!(lang.char_type('。'), "P"); // CJK punctuation
        assert_eq!(lang.char_type('、'), "P"); // CJK punctuation
        assert_eq!(lang.char_type('「'), "P"); // CJK punctuation
        assert_eq!(lang.char_type('A'), "A"); // ASCII
        assert_eq!(lang.char_type('ａ'), "A"); // Full-width Latin
        assert_eq!(lang.char_type('5'), "N"); // Digit
        assert_eq!(lang.char_type('５'), "N"); // Full-width digit
        assert_eq!(lang.char_type('@'), "O"); // Other
    }

    // --- Chinese pattern tests ---

    #[test]
    fn test_chinese_char_types() {
        let lang = Language::Chinese;
        assert_eq!(lang.char_type('的'), "F"); // Function word (structural particle)
        assert_eq!(lang.char_type('了'), "F"); // Function word (aspect particle)
        assert_eq!(lang.char_type('在'), "F"); // Function word (preposition)
        assert_eq!(lang.char_type('是'), "F"); // Function word (verb)
        assert_eq!(lang.char_type('中'), "C"); // CJK Unified (not a function word)
        assert_eq!(lang.char_type('国'), "C"); // CJK Unified
        assert_eq!(lang.char_type('人'), "C"); // CJK Unified
        assert_eq!(lang.char_type('㐀'), "X"); // CJK Extension A (U+3400)
        assert_eq!(lang.char_type('⺀'), "R"); // CJK Radicals Supplement (U+2E80)
        assert_eq!(lang.char_type('ㄅ'), "B"); // Bopomofo (U+3105)
        assert_eq!(lang.char_type('。'), "P"); // Chinese punctuation (U+3002)
        assert_eq!(lang.char_type('，'), "P"); // Full-width comma (U+FF0C)
        assert_eq!(lang.char_type('A'), "A"); // ASCII
        assert_eq!(lang.char_type('5'), "N"); // Digit
        assert_eq!(lang.char_type('@'), "O"); // Other
    }

    // --- Korean pattern tests ---

    #[test]
    fn test_korean_char_types() {
        let lang = Language::Korean;
        assert_eq!(lang.char_type('는'), "E"); // Particle (topic marker)
        assert_eq!(lang.char_type('은'), "E"); // Particle (topic marker)
        assert_eq!(lang.char_type('을'), "E"); // Particle (object marker)
        assert_eq!(lang.char_type('를'), "E"); // Particle (object marker)
        assert_eq!(lang.char_type('의'), "E"); // Particle (possessive)
        assert_eq!(lang.char_type('에'), "E"); // Particle (locative)
        assert_eq!(lang.char_type('가'), "SN"); // Hangul Syllable without 받침
        assert_eq!(lang.char_type('나'), "SN"); // Hangul Syllable without 받침
        assert_eq!(lang.char_type('하'), "SN"); // Hangul Syllable without 받침
        assert_eq!(lang.char_type('한'), "SF"); // Hangul Syllable with 받침
        assert_eq!(lang.char_type('글'), "SF"); // Hangul Syllable with 받침
        assert_eq!(lang.char_type('각'), "SF"); // Hangul Syllable with 받침
        assert_eq!(lang.char_type('ㄱ'), "G"); // Compatibility Jamo (consonant)
        assert_eq!(lang.char_type('ㅏ'), "G"); // Compatibility Jamo (vowel)
        assert_eq!(lang.char_type('ㅎ'), "G"); // Compatibility Jamo (last consonant)
        assert_eq!(lang.char_type('\u{1100}'), "J"); // Hangul Jamo (choseong kiyeok)
        assert_eq!(lang.char_type('\u{1161}'), "J"); // Hangul Jamo (jungseong a)
        assert_eq!(lang.char_type('漢'), "H"); // Hanja
        assert_eq!(lang.char_type('。'), "P"); // Punctuation (U+3002)
        assert_eq!(lang.char_type('A'), "A"); // ASCII
        assert_eq!(lang.char_type('5'), "N"); // Digit
        assert_eq!(lang.char_type('@'), "O"); // Other
    }
}
