//! The token class handed to PHP.

use ext_php_rs::prelude::*;
use litsea_binding_core::TokenView;

/// A segmented token.
///
/// `start` and `end` are byte offsets into the input string. PHP strings are
/// byte strings, so `substr($text, $token->start, $token->end - $token->start)`
/// returns the surface directly.
#[php_class]
#[php(name = "Litsea\\Token")]
#[derive(Default)]
pub struct Token {
    /// The token's surface form.
    #[php(prop)]
    pub surface: String,
    /// The UPOS tag name (for example `"NOUN"`), or `null` when the
    /// segmenter has no POS model.
    #[php(prop)]
    pub pos: Option<String>,
    /// Starting byte offset in the input string.
    #[php(prop)]
    pub start: u64,
    /// Ending byte offset (exclusive) in the input string.
    #[php(prop)]
    pub end: u64,
}

#[php_impl]
impl Token {
    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `これ/PRON [0..6]`.
    pub fn __to_string(&self) -> String {
        match &self.pos {
            Some(pos) => format!("{}/{} [{}..{}]", self.surface, pos, self.start, self.end),
            None => format!("{} [{}..{}]", self.surface, self.start, self.end),
        }
    }
}

impl From<TokenView> for Token {
    /// Converts a core token view into the PHP-facing token.
    ///
    /// # Arguments
    /// * `view` - The token view.
    ///
    /// # Returns
    /// The corresponding [`Token`].
    fn from(view: TokenView) -> Self {
        Self {
            surface: view.surface,
            // The tag travels as its name: PHP enums cannot be registered
            // from ext-php-rs, and a string matches what the other bindings
            // expose on the token.
            pos: view.pos.map(|pos| pos.to_string()),
            start: view.byte_start as u64,
            end: view.byte_end as u64,
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
        assert_eq!(token.__to_string(), "テスト/NOUN [9..18]");
    }

    #[test]
    fn test_untagged_token() {
        let token = Token::from(TokenView::new("テスト", 0, 9, None));
        assert_eq!(token.pos, None);
        assert_eq!(token.__to_string(), "テスト [0..9]");
    }
}
