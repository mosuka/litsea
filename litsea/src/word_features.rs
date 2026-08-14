//! Word-level feature templates for the two-stage POS tagger (issue #147).
//!
//! This module is the single source of truth for the stage-2 feature
//! inventory: the training extractor writes feature *strings* with
//! [`write_word_features`], and the packed runtime compiles the same
//! strings back into integer keys / dense indices with
//! [`parse_word_feature`]. The two directions are pinned against each other
//! by a round-trip unit test, and the runtime builds its probe keys with
//! the same helpers ([`hash_key`], [`ts_payload`]) that the parser
//! produces.
//!
//! # Templates
//!
//! For a word spanning `[start, end)` of a sentence (`w` = surface,
//! `n = end - start`):
//!
//! | prefix | value | representation |
//! |--------|-------|----------------|
//! | `WS` | the surface itself | surface string |
//! | `WL` | `min(n, 4)` | dense (word length) |
//! | `FC` / `LC` | first / last char | hashed char key |
//! | `ft` / `lt` | first / last char type | dense (type) |
//! | `TS` | type codes of the first ≤8 chars | hashed key |
//! | `L1`-`L3` / `R1`-`R3` | context chars at distance 1-3 | hashed char key |
//! | `cl1`-`cl3` / `cr1`-`cr3` | context char types | dense (type) |
//! | `LB` / `RB` | context bigrams (distance 2+1 / 1+2) | hashed pair key |
//! | `P2` / `S2` | first / last two chars (words with n ≥ 2) | hashed pair key |
//!
//! Context positions beyond the sentence use the sentinel characters
//! [`BOS_CHAR`] / [`EOS_CHAR`] (U+0001 / U+0002) in feature strings — a
//! single character, so parsing is unambiguous, and never produced by real
//! text — and the out-of-Unicode codes [`BOS_CODE`] / [`EOS_CODE`] in
//! packed keys (the same trick as `packed_model::SENTINEL_BASE`). Type
//! strings use the language's type codes, which are prefix-free by design,
//! so the concatenated `TS` payload parses unambiguously.

use crate::language::Language;

/// Sentinel character standing for "before the sentence" in feature
/// strings (control character U+0001; real text never contains it).
pub(crate) const BOS_CHAR: char = '\u{1}';
/// Sentinel character standing for "after the sentence" in feature
/// strings (control character U+0002).
pub(crate) const EOS_CHAR: char = '\u{2}';
/// Packed char code of [`BOS_CHAR`] (first code above Unicode).
pub(crate) const BOS_CODE: u64 = 0x11_0000;
/// Packed char code of [`EOS_CHAR`].
pub(crate) const EOS_CODE: u64 = 0x11_0001;

/// Word-length cap of the `WL` template.
pub(crate) const WL_CAP: usize = 4;
/// Character cap of the `TS` (word type string) template.
pub(crate) const TS_CAP: usize = 8;
/// Context window of the `L*`/`R*`/`cl*`/`cr*` templates.
pub(crate) const CONTEXT_WINDOW: usize = 3;

/// Number of word-feature templates.
pub(crate) const N_WORD_TEMPLATES: usize = 23;

/// Template prefixes, indexed by template id. The id order is load-bearing
/// for packed hash keys (`id << 56`); append-only.
pub(crate) const WORD_TEMPLATE_PREFIXES: [&str; N_WORD_TEMPLATES] = [
    "WS", "WL", "FC", "LC", "ft", "lt", "TS", "L1", "L2", "L3", "R1", "R2", "R3", "cl1", "cl2",
    "cl3", "cr1", "cr2", "cr3", "LB", "RB", "P2", "S2",
];

/// Template ids (indices into [`WORD_TEMPLATE_PREFIXES`]).
pub(crate) const T_WS: usize = 0;
pub(crate) const T_WL: usize = 1;
pub(crate) const T_FC: usize = 2;
pub(crate) const T_LC: usize = 3;
pub(crate) const T_FT: usize = 4;
pub(crate) const T_LT: usize = 5;
pub(crate) const T_TS: usize = 6;
/// First of the three left-context char templates (`L1`..`L3`).
pub(crate) const T_L1: usize = 7;
/// First of the three right-context char templates (`R1`..`R3`).
pub(crate) const T_R1: usize = 10;
/// First of the three left-context type templates (`cl1`..`cl3`).
pub(crate) const T_CL1: usize = 13;
/// First of the three right-context type templates (`cr1`..`cr3`).
pub(crate) const T_CR1: usize = 16;
pub(crate) const T_LB: usize = 19;
pub(crate) const T_RB: usize = 20;
pub(crate) const T_P2: usize = 21;
pub(crate) const T_S2: usize = 22;

/// Number of dense type-valued families (`ft`, `lt`, `cl1`-`cl3`,
/// `cr1`-`cr3`), in that family order.
pub(crate) const N_TYPE_FAMILIES: usize = 8;
/// Dense family index of `ft`.
pub(crate) const F_FT: usize = 0;
/// Dense family index of `lt`.
pub(crate) const F_LT: usize = 1;
/// Dense family index of `cl1` (`cl2`/`cl3` follow).
pub(crate) const F_CL1: usize = 2;
/// Dense family index of `cr1` (`cr2`/`cr3` follow).
pub(crate) const F_CR1: usize = 5;

/// A parsed word feature, routed to its storage class by the packed
/// runtime.
#[derive(Debug, PartialEq)]
pub(crate) enum WordFeature<'a> {
    /// `WS`: the word surface (stored on the surface map).
    Surface(&'a str),
    /// `WL`: capped word length in `1..=WL_CAP` (dense).
    WordLen(usize),
    /// Type-valued template (dense): family index and type index, where
    /// the type index is a language type id, `radix` (BOS), or `radix + 1`
    /// (EOS).
    TypeDense { family: usize, type_idx: usize },
    /// Char-valued or type-string template: a packed hash key
    /// (`template_id << 56 | payload`).
    Hash(u64),
}

/// Builds a packed hash key for a template id and payload.
#[inline]
pub(crate) fn hash_key(template_id: usize, payload: u64) -> u64 {
    ((template_id as u64) << 56) | payload
}

/// Packed code of a real or sentinel context character.
#[inline]
pub(crate) fn char_code(c: char) -> u64 {
    match c {
        BOS_CHAR => BOS_CODE,
        EOS_CHAR => EOS_CODE,
        _ => c as u64,
    }
}

/// Builds the `TS` payload from the word's leading type ids
/// (`len <= TS_CAP`, each id `< 16`): the length in the high nibble
/// (bits 32+), one 4-bit id per character.
#[inline]
pub(crate) fn ts_payload(type_ids: &[u8]) -> u64 {
    let mut payload = (type_ids.len() as u64) << 32;
    for (i, &id) in type_ids.iter().enumerate() {
        payload |= u64::from(id) << (4 * i);
    }
    payload
}

/// Parses one feature string written by [`write_word_features`] back into
/// its routed form.
///
/// # Arguments
/// * `language` - The language whose type codes to parse against.
/// * `feature` - The feature string (`prefix:payload`).
///
/// # Returns
/// The parsed feature, or `None` if the string does not match any word
/// template for this language (such features are unreachable at inference
/// and are skipped by the packed build, mirroring
/// `packed_model::parse_feature_keys`).
pub(crate) fn parse_word_feature(language: Language, feature: &str) -> Option<WordFeature<'_>> {
    let (prefix, payload) = feature.split_once(':')?;
    let tid = WORD_TEMPLATE_PREFIXES.iter().position(|p| *p == prefix)?;
    match tid {
        T_WS => Some(WordFeature::Surface(payload)),
        T_WL => {
            let len: usize = payload.parse().ok()?;
            (1..=WL_CAP).contains(&len).then_some(WordFeature::WordLen(len))
        }
        T_FT | T_LT => {
            let family = if tid == T_FT { F_FT } else { F_LT };
            Some(WordFeature::TypeDense {
                family,
                type_idx: parse_type_idx(language, payload)?,
            })
        }
        _ if (T_CL1..T_CL1 + CONTEXT_WINDOW).contains(&tid) => Some(WordFeature::TypeDense {
            family: F_CL1 + (tid - T_CL1),
            type_idx: parse_type_idx(language, payload)?,
        }),
        _ if (T_CR1..T_CR1 + CONTEXT_WINDOW).contains(&tid) => Some(WordFeature::TypeDense {
            family: F_CR1 + (tid - T_CR1),
            type_idx: parse_type_idx(language, payload)?,
        }),
        T_TS => {
            let mut ids = Vec::with_capacity(TS_CAP);
            let mut rest = payload;
            while !rest.is_empty() {
                if ids.len() == TS_CAP {
                    return None;
                }
                // Type-code sets are prefix-free, so greedy matching is
                // the unique parse.
                let (id, r) = parse_type_code(language, rest)?;
                ids.push(id);
                rest = r;
            }
            (!ids.is_empty()).then(|| WordFeature::Hash(hash_key(T_TS, ts_payload(&ids))))
        }
        T_LB | T_RB | T_P2 | T_S2 => {
            let mut chars = payload.chars();
            let c1 = chars.next()?;
            let c2 = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(WordFeature::Hash(hash_key(tid, (char_code(c1) << 24) | char_code(c2))))
        }
        // FC / LC / L1..L3 / R1..R3: exactly one (possibly sentinel) char.
        _ => {
            let mut chars = payload.chars();
            let c = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(WordFeature::Hash(hash_key(tid, char_code(c))))
        }
    }
}

/// Parses a full payload as a single type code or sentinel, returning the
/// dense type index.
fn parse_type_idx(language: Language, payload: &str) -> Option<usize> {
    let (id, rest) = parse_type_code(language, payload)?;
    rest.is_empty().then_some(usize::from(id))
}

/// Greedily parses one leading type code (or sentinel char) off `s`,
/// returning its index and the remainder. Sentinels map to `radix` /
/// `radix + 1`.
fn parse_type_code(language: Language, s: &str) -> Option<(u8, &str)> {
    if let Some(rest) = s.strip_prefix(BOS_CHAR) {
        return Some((language.type_codes().len() as u8, rest));
    }
    if let Some(rest) = s.strip_prefix(EOS_CHAR) {
        return Some(((language.type_codes().len() + 1) as u8, rest));
    }
    for (id, code) in language.type_codes().iter().enumerate() {
        if let Some(rest) = s.strip_prefix(code) {
            return Some((id as u8, rest));
        }
    }
    None
}

/// Writes the word features of the word spanning `[start, end)` as
/// strings, in template order, restricted to templates for which
/// `select(template_id)` returns true.
///
/// This is the training-side twin of the packed runtime's key computation;
/// both are pinned together by the round-trip test in this module (with
/// `select` always true — a real extractor passes
/// `TwoStageFeatureSet::includes`).
///
/// # Arguments
/// * `language` - The language whose type codes to write.
/// * `sent` - The sentence characters (no sentinels).
/// * `type_ids` - The per-character language type ids of `sent`.
/// * `start` / `end` - The word's char span (`start < end <= sent.len()`).
/// * `select` - Called with each template id ([`T_WS`] etc.); only
///   templates for which it returns true are written.
/// * `push` - Receives each selected feature string.
pub(crate) fn write_word_features(
    language: Language,
    sent: &[char],
    type_ids: &[u8],
    start: usize,
    end: usize,
    select: impl Fn(usize) -> bool,
    push: &mut impl FnMut(String),
) {
    let codes = language.type_codes();
    let n = end - start;
    let lc = |k: usize| if start >= k { sent[start - k] } else { BOS_CHAR };
    let rc = |k: usize| sent.get(end + k - 1).copied().unwrap_or(EOS_CHAR);
    let type_code = |c: char| -> &str {
        match c {
            BOS_CHAR => "\u{1}",
            EOS_CHAR => "\u{2}",
            _ => codes[language.char_type_id(c) as usize],
        }
    };
    let mut emit = |tid: usize, s: String| {
        if select(tid) {
            push(s);
        }
    };

    emit(T_WS, format!("WS:{}", sent[start..end].iter().collect::<String>()));
    emit(T_WL, format!("WL:{}", n.min(WL_CAP)));
    emit(T_FC, format!("FC:{}", sent[start]));
    emit(T_LC, format!("LC:{}", sent[end - 1]));
    emit(T_FT, format!("ft:{}", codes[type_ids[start] as usize]));
    emit(T_LT, format!("lt:{}", codes[type_ids[end - 1] as usize]));
    let ts: String = type_ids[start..end.min(start + TS_CAP)]
        .iter()
        .map(|&t| codes[t as usize])
        .collect();
    emit(T_TS, format!("TS:{}", ts));
    for k in 1..=CONTEXT_WINDOW {
        emit(T_L1 + k - 1, format!("L{}:{}", k, lc(k)));
    }
    for k in 1..=CONTEXT_WINDOW {
        emit(T_R1 + k - 1, format!("R{}:{}", k, rc(k)));
    }
    for k in 1..=CONTEXT_WINDOW {
        emit(T_CL1 + k - 1, format!("cl{}:{}", k, type_code(lc(k))));
    }
    for k in 1..=CONTEXT_WINDOW {
        emit(T_CR1 + k - 1, format!("cr{}:{}", k, type_code(rc(k))));
    }
    emit(T_LB, format!("LB:{}{}", lc(2), lc(1)));
    emit(T_RB, format!("RB:{}{}", rc(1), rc(2)));
    if n >= 2 {
        emit(T_P2, format!("P2:{}{}", sent[start], sent[start + 1]));
        emit(T_S2, format!("S2:{}{}", sent[end - 2], sent[end - 1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Type index of a context character for the dense type families:
    /// a language type id, or the BOS/EOS sentinel indices `radix` /
    /// `radix + 1`.
    fn context_type_idx(language: Language, c: char) -> usize {
        match c {
            BOS_CHAR => language.type_codes().len(),
            EOS_CHAR => language.type_codes().len() + 1,
            _ => language.char_type_id(c) as usize,
        }
    }

    fn context(language: Language, text: &str) -> (Vec<char>, Vec<u8>) {
        let chars: Vec<char> = text.chars().collect();
        let type_ids = chars.iter().map(|&c| language.char_type_id(c)).collect();
        (chars, type_ids)
    }

    /// Computes the packed-runtime view of every feature of a span the
    /// same way `PackedTwoStageModel` does, for the round-trip test.
    fn runtime_features(
        language: Language,
        sent: &[char],
        type_ids: &[u8],
        start: usize,
        end: usize,
    ) -> Vec<WordFeature<'static>> {
        let n = end - start;
        let lc = |k: usize| if start >= k { sent[start - k] } else { BOS_CHAR };
        let rc = |k: usize| sent.get(end + k - 1).copied().unwrap_or(EOS_CHAR);
        let mut out: Vec<WordFeature<'static>> = vec![
            WordFeature::WordLen(n.min(WL_CAP)),
            WordFeature::Hash(hash_key(T_FC, char_code(sent[start]))),
            WordFeature::Hash(hash_key(T_LC, char_code(sent[end - 1]))),
            WordFeature::TypeDense {
                family: F_FT,
                type_idx: type_ids[start] as usize,
            },
            WordFeature::TypeDense {
                family: F_LT,
                type_idx: type_ids[end - 1] as usize,
            },
            WordFeature::Hash(hash_key(
                T_TS,
                ts_payload(&type_ids[start..end.min(start + TS_CAP)]),
            )),
        ];
        for k in 1..=CONTEXT_WINDOW {
            out.push(WordFeature::Hash(hash_key(T_L1 + k - 1, char_code(lc(k)))));
        }
        for k in 1..=CONTEXT_WINDOW {
            out.push(WordFeature::Hash(hash_key(T_R1 + k - 1, char_code(rc(k)))));
        }
        for k in 1..=CONTEXT_WINDOW {
            out.push(WordFeature::TypeDense {
                family: F_CL1 + k - 1,
                type_idx: context_type_idx(language, lc(k)),
            });
        }
        for k in 1..=CONTEXT_WINDOW {
            out.push(WordFeature::TypeDense {
                family: F_CR1 + k - 1,
                type_idx: context_type_idx(language, rc(k)),
            });
        }
        out.push(WordFeature::Hash(hash_key(T_LB, (char_code(lc(2)) << 24) | char_code(lc(1)))));
        out.push(WordFeature::Hash(hash_key(T_RB, (char_code(rc(1)) << 24) | char_code(rc(2)))));
        if n >= 2 {
            out.push(WordFeature::Hash(hash_key(
                T_P2,
                (char_code(sent[start]) << 24) | char_code(sent[start + 1]),
            )));
            out.push(WordFeature::Hash(hash_key(
                T_S2,
                (char_code(sent[end - 2]) << 24) | char_code(sent[end - 1]),
            )));
        }
        out
    }

    /// The writer's strings, parsed back, must equal the runtime's direct
    /// key computation for every span — including sentence edges (BOS/EOS
    /// sentinels) and multi-char Korean type codes.
    #[test]
    fn test_writer_parser_round_trip() {
        for (language, text) in [
            (Language::Japanese, "これは犬です"),
            (Language::Japanese, "3匹のネコ"),
            (Language::Korean, "고양이가 3마리"),
            (Language::Chinese, "我爱北京"),
        ] {
            let (sent, type_ids) = context(language, text);
            for start in 0..sent.len() {
                for end in start + 1..=sent.len() {
                    let mut written = Vec::new();
                    write_word_features(
                        language,
                        &sent,
                        &type_ids,
                        start,
                        end,
                        |_| true,
                        &mut |f| {
                            written.push(f);
                        },
                    );
                    let parsed: Vec<WordFeature> = written
                        .iter()
                        .map(|f| {
                            parse_word_feature(language, f)
                                .unwrap_or_else(|| panic!("unparsable feature {:?}", f))
                        })
                        .collect();
                    // The first parsed feature is the surface; the rest
                    // must equal the runtime computation in order.
                    let surface: String = sent[start..end].iter().collect();
                    assert_eq!(parsed[0], WordFeature::Surface(surface.as_str()));
                    let expected = runtime_features(language, &sent, &type_ids, start, end);
                    assert_eq!(
                        &parsed[1..],
                        &expected[..],
                        "span {}..{} of {:?}",
                        start,
                        end,
                        text
                    );
                }
            }
        }
    }

    #[test]
    fn test_parse_rejects_unknown_and_malformed() {
        let language = Language::Japanese;
        for feature in [
            "XX:foo",       // unknown prefix
            "WSfoo",        // no colon
            "WL:0",         // below range
            "WL:5",         // above cap
            "WL:x",         // not a number
            "FC:ab",        // two chars where one is expected
            "FC:",          // empty payload
            "ft:Z",         // unknown type code
            "TS:",          // empty type string
            "TS:HHHHHHHHH", // more than TS_CAP codes
            "LB:a",         // one char where two are expected
            "LB:abc",       // three chars where two are expected
        ] {
            assert!(
                parse_word_feature(language, feature).is_none(),
                "{:?} should not parse",
                feature
            );
        }
    }

    #[test]
    fn test_sentinel_codes_are_outside_unicode() {
        assert!(BOS_CODE > char::MAX as u64);
        assert_eq!(char_code(BOS_CHAR), BOS_CODE);
        assert_eq!(char_code(EOS_CHAR), EOS_CODE);
        assert_eq!(char_code('あ'), 'あ' as u64);
    }
}
