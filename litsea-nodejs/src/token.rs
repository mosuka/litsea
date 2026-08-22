//! The token object handed to JavaScript.

use litsea_binding_core::TokenView;

/// A segmented token.
///
/// `start` and `end` are byte offsets into the input string, so
/// `Buffer.from(text).subarray(token.start, token.end).toString()` returns
/// the surface. They are exact for both segmentation and POS output,
/// including for space-preserving languages, where the whitespace itself is
/// a token.
///
/// Note that JavaScript string indices are UTF-16 code units, so these
/// offsets are not directly usable with `String.prototype.slice`.
#[napi(object)]
pub struct Token {
    /// The token's surface form.
    pub surface: String,
    /// The UPOS tag name (for example `"NOUN"`), or `undefined` when the
    /// segmenter has no POS model (napi maps `None` to `undefined`).
    pub pos: Option<String>,
    /// Starting byte offset in the input string.
    pub start: u32,
    /// Ending byte offset (exclusive) in the input string.
    pub end: u32,
}

impl From<TokenView> for Token {
    /// Converts a core token view into the JavaScript-facing object.
    ///
    /// # Arguments
    /// * `view` - The token view.
    ///
    /// # Returns
    /// The corresponding [`Token`].
    fn from(view: TokenView) -> Self {
        Self {
            surface: view.surface,
            // The tag travels as its name: JavaScript has no enum type, and
            // a string union is what the generated `.d.ts` can express.
            pos: view.pos.map(|pos| pos.to_string()),
            start: view.byte_start as u32,
            end: view.byte_end as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use litsea::Upos;

    use super::*;

    #[test]
    fn test_conversion_keeps_offsets_and_tag_name() {
        let token = Token::from(TokenView::new("テスト", 9, 18, Some(Upos::NOUN)));
        assert_eq!(token.surface, "テスト");
        assert_eq!(token.pos.as_deref(), Some("NOUN"));
        assert_eq!(token.start, 9);
        assert_eq!(token.end, 18);
    }

    #[test]
    fn test_untagged_token_has_no_pos() {
        let token = Token::from(TokenView::new("テスト", 0, 9, None));
        assert_eq!(token.pos, None);
    }
}
