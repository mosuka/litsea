//! Supported languages and character type classification.
//!
//! Defines [`Language`] and the per-language character classification
//! rules: every character maps to a language-specific type code (e.g. "H"
//! for hiragana) through direct `match`-based Unicode range checks, feeding
//! the type-based feature templates of the segmenter.

use std::fmt;
use std::str::FromStr;

/// Error returned when parsing a [`Language`] from a string fails.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("Unsupported language: '{input}'. Supported: japanese (ja), chinese (zh), korean (ko)")]
pub struct ParseLanguageError {
    input: String,
}

impl ParseLanguageError {
    /// Returns the string that failed to parse.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }
}

/// Supported languages for word segmentation.
///
/// Marked `#[non_exhaustive]`: new languages are expected to be added (the
/// language-support guide documents the extension procedure), so external
/// matches must carry a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
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
    type Err = ParseLanguageError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "japanese" | "ja" => Ok(Language::Japanese),
            "chinese" | "zh" => Ok(Language::Chinese),
            "korean" | "ko" => Ok(Language::Korean),
            _ => Err(ParseLanguageError {
                input: s.to_string(),
            }),
        }
    }
}

/// Type id of the "O" (Other) code. Shared by every language: index 0 of
/// every [`Language::type_codes`] table. Also used by the segmenter as the
/// padding type id (padding is intentionally indistinguishable from a real
/// Other-class character, matching the string representation).
pub(crate) const OTHER_TYPE_ID: u8 = 0;
/// Type id of the shared "P" (punctuation) code (index 1 in every table).
const PUNCT_TYPE_ID: u8 = 1;
/// Type id of the shared "A" (Latin) code (index 2 in every table).
const LATIN_TYPE_ID: u8 = 2;
/// Type id of the shared "N" (digit) code (index 3 in every table).
const DIGIT_TYPE_ID: u8 = 3;

impl Language {
    /// Returns the ordered table of type codes this language can produce.
    ///
    /// The index of a code in this table is its type id as returned by
    /// [`char_type_id`](Self::char_type_id). The shared codes occupy fixed
    /// indices across all languages ("O" = 0, "P" = 1, "A" = 2, "N" = 3);
    /// language-specific codes follow from index 4.
    pub(crate) fn type_codes(self) -> &'static [&'static str] {
        match self {
            Language::Japanese => &["O", "P", "A", "N", "M", "H", "I", "K"],
            Language::Chinese => &["O", "P", "A", "N", "F", "C", "X", "R", "B"],
            Language::Korean => &["O", "P", "A", "N", "E", "SN", "SF", "J", "G", "H"],
        }
    }

    /// Classifies a character into a language-specific type id (an index
    /// into [`type_codes`](Self::type_codes)).
    ///
    /// Classification is a direct `match` on character ranges, so it is
    /// allocation-free and O(1).
    pub(crate) fn char_type_id(self, c: char) -> u8 {
        match self {
            Language::Japanese => japanese_char_type_id(c),
            Language::Chinese => chinese_char_type_id(c),
            Language::Korean => korean_char_type_id(c),
        }
    }

    /// Classifies a character into a language-specific type code.
    ///
    /// Returns "O" (Other) if the character does not belong to any class.
    /// Implemented as a table lookup over [`char_type_id`](Self::char_type_id),
    /// so the string codes and the numeric ids are consistent by construction.
    ///
    /// # Arguments
    /// * `c` - The character to classify.
    #[must_use]
    pub fn char_type(&self, c: char) -> &'static str {
        self.type_codes()[self.char_type_id(c) as usize]
    }
}

/// Classes shared by all languages, checked after the language-specific ones:
/// - "P" (id 1): CJK Symbols and Punctuation + full-width punctuation
/// - "A" (id 2): ASCII and full-width Latin characters
/// - "N" (id 3): Digits (ASCII and full-width)
fn punct_latin_digit(c: char) -> Option<u8> {
    match c {
        '\u{3000}'..='\u{303F}'
        | '\u{FF01}'..='\u{FF0F}'
        | '\u{FF1A}'..='\u{FF20}'
        | '\u{FF3B}'..='\u{FF40}'
        | '\u{FF5B}'..='\u{FF65}' => Some(PUNCT_TYPE_ID),
        'a'..='z' | 'A'..='Z' | 'ａ'..='ｚ' | 'Ａ'..='Ｚ' => Some(LATIN_TYPE_ID),
        '0'..='9' | '０'..='９' => Some(DIGIT_TYPE_ID),
        _ => None,
    }
}

/// Character type classification for Japanese, returning type ids.
///
/// Type codes (ids per the Japanese [`Language::type_codes`] table):
/// - "M" (4): Kanji numbers (一二三四五六七八九十百千万億兆)
/// - "H" (5): Kanji (CJK Unified Ideographs, U+4E00..=U+9FFF, plus 々〆ヵヶ)
/// - "I" (6): Hiragana
/// - "K" (7): Katakana (full-width and half-width)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O" (0): Other (fallback)
fn japanese_char_type_id(c: char) -> u8 {
    match c {
        '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千'
        | '万' | '億' | '兆' => 4, // "M"
        // CJK Unified Ideographs (U+4E00..=U+9FFF) plus 々〆ヵヶ
        '\u{4E00}'..='\u{9FFF}' | '々' | '〆' | 'ヵ' | 'ヶ' => 5, // "H"
        // ぁ-ん
        '\u{3041}'..='\u{3093}' => 6, // "I"
        // ァ-ヴ, ー, half-width ｱ-ﾝ and ﾞﾟ
        '\u{30A1}'..='\u{30F4}' | 'ー' | '\u{FF71}'..='\u{FF9D}' | 'ﾞ' | 'ﾟ' => 7, // "K"
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
    }
}

/// Character type classification for Chinese, returning type ids.
///
/// Type codes (ids per the Chinese [`Language::type_codes`] table):
/// - "F" (4): High-frequency function words (虚词: 的了在是和不也 etc.)
/// - "C" (5): CJK Unified Ideographs (U+4E00..U+9FFF)
/// - "X" (6): CJK Extension A (U+3400..U+4DBF)
/// - "R" (7): CJK Radicals and Kangxi Radicals (U+2E80..U+2FDF)
/// - "B" (8): Bopomofo (Zhuyin)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O" (0): Other (fallback)
fn chinese_char_type_id(c: char) -> u8 {
    match c {
        // High-frequency function words (虚词): structural particles,
        // aspect/modal particles, conjunctions, prepositions, and common
        // grammatical verbs/adverbs
        '的' | '地' | '得' | '了' | '着' | '过' | '吗' | '呢' | '吧' | '啊' | '嘛' | '和'
        | '与' | '或' | '但' | '而' | '且' | '及' | '在' | '从' | '到' | '把' | '被' | '对'
        | '向' | '给' | '是' | '有' | '不' | '也' | '都' | '就' | '要' | '会' | '能' | '可' =>
        {
            4 // "F"
        }
        '\u{4E00}'..='\u{9FFF}' => 5, // "C"
        '\u{3400}'..='\u{4DBF}' => 6, // "X"
        '\u{2E80}'..='\u{2FDF}' => 7, // "R"
        // Bopomofo + Bopomofo Extended
        '\u{3100}'..='\u{312F}' | '\u{31A0}'..='\u{31BF}' => 8, // "B"
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
    }
}

/// Character type classification for Korean, returning type ids.
///
/// Type codes (ids per the Korean [`Language::type_codes`] table):
/// - "E" (4): High-frequency particles/endings (조사/어미: 은는을를의에)
/// - "SN" (5): Hangul Syllable without 받침 (e.g., 가, 나, 하)
/// - "SF" (6): Hangul Syllable with 받침 (e.g., 한, 글, 각)
/// - "J" (7): Hangul Jamo (U+1100..U+11FF)
/// - "G" (8): Hangul Compatibility Jamo (U+3130..U+318F)
/// - "H" (9): Hanja / CJK Ideographs (U+4E00..U+9FFF)
/// - "P" / "A" / "N": see [`punct_latin_digit`]
/// - "O" (0): Other (fallback)
fn korean_char_type_id(c: char) -> u8 {
    match c {
        // Overwhelmingly used as grammatical particles:
        // 은/는 (topic), 을/를 (object), 의 (possessive), 에 (locative)
        '은' | '는' | '을' | '를' | '의' | '에' => 4, // "E"
        // Hangul Syllables: (codepoint - 0xAC00) % 28 == 0 means no 받침
        // (final consonant)
        '\u{AC00}'..='\u{D7AF}' => {
            if (c as u32 - 0xAC00).is_multiple_of(28) {
                5 // "SN"
            } else {
                6 // "SF"
            }
        }
        '\u{1100}'..='\u{11FF}' => 7, // "J"
        '\u{3130}'..='\u{318F}' => 8, // "G"
        '\u{4E00}'..='\u{9FFF}' => 9, // "H"
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
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
    fn test_parse_language_error_message() {
        // #128: the typed parse error must render the exact message the CLI
        // shows through clap (pinned end-to-end by the CLI integration test).
        let err = "french".parse::<Language>().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported language: 'french'. Supported: japanese (ja), chinese (zh), korean (ko)"
        );
        assert_eq!(err.input(), "french");
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

    // --- Type-code table tests (#136) ---

    const ALL_LANGUAGES: [Language; 3] = [Language::Japanese, Language::Chinese, Language::Korean];

    #[test]
    fn test_type_codes_unique_and_shared_prefix() {
        // The packed-key encoding relies on ids (table indices) mapping
        // one-to-one to codes, and on the shared codes sitting at fixed
        // indices so punct_latin_digit can stay language-agnostic.
        for lang in ALL_LANGUAGES {
            let codes = lang.type_codes();
            for (i, a) in codes.iter().enumerate() {
                for b in &codes[i + 1..] {
                    assert_ne!(a, b, "duplicate type code in {lang} table");
                }
            }
            assert_eq!(codes[OTHER_TYPE_ID as usize], "O");
            assert_eq!(codes[1], "P");
            assert_eq!(codes[2], "A");
            assert_eq!(codes[3], "N");
        }
    }

    #[test]
    fn test_type_codes_prefix_free() {
        // The load-time feature parser decodes concatenated type codes left
        // to right; that is deterministic only while no code is a prefix of
        // another. In particular "S" alone must not be a Korean code (SN/SF
        // are the only multi-character codes).
        for lang in ALL_LANGUAGES {
            let codes = lang.type_codes();
            for (i, a) in codes.iter().enumerate() {
                for (j, b) in codes.iter().enumerate() {
                    if i != j {
                        assert!(!b.starts_with(a), "{lang}: code {a:?} is a prefix of {b:?}");
                    }
                }
            }
            assert!(!codes.contains(&"S"));
        }
    }

    #[test]
    fn test_char_type_id_consistent_with_char_type() {
        // char_type is a table lookup over char_type_id, so consistency is
        // structural; this pins the id values for a representative sample.
        let samples = ['あ', 'ア', '漢', '的', '中', '는', '한', '가', 'A', '5', '。', '@'];
        for lang in ALL_LANGUAGES {
            for c in samples {
                let id = lang.char_type_id(c) as usize;
                assert!(id < lang.type_codes().len());
                assert_eq!(lang.type_codes()[id], lang.char_type(c));
            }
        }
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

    #[test]
    fn test_japanese_kanji_range_covers_full_cjk_block() {
        // #130: the Japanese classifier previously stopped at U+9FA0 while
        // Chinese and Korean already covered the full CJK Unified
        // Ideographs block (U+4E00..=U+9FFF). Pin the full range for all
        // three languages so they can't drift apart again.
        assert_eq!(Language::Japanese.char_type('\u{9FA0}'), "H"); // old upper bound
        assert_eq!(Language::Japanese.char_type('\u{9FA1}'), "H"); // 龡, first newly-included
        assert_eq!(Language::Japanese.char_type('\u{9FFF}'), "H"); // 鿿, block end

        assert_eq!(Language::Chinese.char_type('\u{9FA1}'), "C");
        assert_eq!(Language::Korean.char_type('\u{9FA1}'), "H");
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
