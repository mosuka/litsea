//! The token object handed to JavaScript.

use litsea_binding_core::TokenView;
use wasm_bindgen::prelude::*;

/// A segmented token.
///
/// `start` and `end` are byte offsets into the UTF-8 encoding of the input.
/// JavaScript string indices are UTF-16 code units, so slice with
/// `TextEncoder` / `TextDecoder` rather than `String.prototype.slice`:
///
/// ```js
/// const bytes = new TextEncoder().encode(text)
/// new TextDecoder().decode(bytes.subarray(token.start, token.end))  // === token.surface
/// ```
#[wasm_bindgen]
pub struct Token {
    /// The token's surface form.
    surface: String,
    /// The UPOS tag name, or `undefined` when the segmenter has no POS model.
    pos: Option<String>,
    /// Starting byte offset in the input.
    start: u32,
    /// Ending byte offset (exclusive) in the input.
    end: u32,
}

#[wasm_bindgen]
impl Token {
    /// The token's surface form.
    #[wasm_bindgen(getter)]
    pub fn surface(&self) -> String {
        self.surface.clone()
    }

    /// The UPOS tag name (for example `"NOUN"`), or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn pos(&self) -> Option<String> {
        self.pos.clone()
    }

    /// Starting byte offset in the input.
    #[wasm_bindgen(getter)]
    pub fn start(&self) -> u32 {
        self.start
    }

    /// Ending byte offset (exclusive) in the input.
    #[wasm_bindgen(getter)]
    pub fn end(&self) -> u32 {
        self.end
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `これ/PRON [0..6]`.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_js_string(&self) -> String {
        match &self.pos {
            Some(pos) => format!("{}/{} [{}..{}]", self.surface, pos, self.start, self.end),
            None => format!("{} [{}..{}]", self.surface, self.start, self.end),
        }
    }
}

impl From<TokenView> for Token {
    /// Converts a core token view into the JavaScript-facing token.
    ///
    /// # Arguments
    /// * `view` - The token view.
    ///
    /// # Returns
    /// The corresponding [`Token`].
    fn from(view: TokenView) -> Self {
        Self {
            surface: view.surface,
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
        assert_eq!(token.surface(), "テスト");
        assert_eq!(token.pos().as_deref(), Some("NOUN"));
        assert_eq!(token.start(), 9);
        assert_eq!(token.end(), 18);
        assert_eq!(token.to_js_string(), "テスト/NOUN [9..18]");
    }

    #[test]
    fn test_untagged_token() {
        let token = Token::from(TokenView::new("テスト", 0, 9, None));
        assert_eq!(token.pos(), None);
        assert_eq!(token.to_js_string(), "テスト [0..9]");
    }
}
