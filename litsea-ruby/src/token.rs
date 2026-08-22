//! The token class handed to Ruby.

use litsea_binding_core::TokenView;
use magnus::{Module, RModule, Ruby, error::Error};

/// A segmented token.
///
/// `start` and `end` are byte offsets into the input string, so
/// `text.byteslice(token.start, token.end - token.start)` returns the
/// surface. Ruby's `String#[]` works in characters, hence `byteslice`.
#[magnus::wrap(class = "Litsea::Token", free_immediately, size)]
pub struct Token {
    /// The token's surface form.
    surface: String,
    /// The UPOS tag name (for example `"NOUN"`), or `nil` when the segmenter
    /// has no POS model.
    pos: Option<String>,
    /// Starting byte offset in the input string.
    start: usize,
    /// Ending byte offset (exclusive) in the input string.
    end: usize,
}

impl Token {
    /// Returns the token's surface form.
    ///
    /// # Returns
    /// The surface string.
    fn surface(&self) -> String {
        self.surface.clone()
    }

    /// Returns the UPOS tag name.
    ///
    /// # Returns
    /// The tag name, or `nil` for segmentation-only output.
    fn pos(&self) -> Option<String> {
        self.pos.clone()
    }

    /// Returns the starting byte offset.
    ///
    /// # Returns
    /// The offset into the input string.
    fn start(&self) -> usize {
        self.start
    }

    /// Returns the ending byte offset (exclusive).
    ///
    /// # Returns
    /// The offset into the input string.
    fn end(&self) -> usize {
        self.end
    }

    /// Returns a readable representation.
    ///
    /// # Returns
    /// For example `#<Litsea::Token これ/PRON [0..6]>`.
    fn inspect(&self) -> String {
        match &self.pos {
            Some(pos) => {
                format!("#<Litsea::Token {}/{} [{}..{}]>", self.surface, pos, self.start, self.end)
            }
            None => format!("#<Litsea::Token {} [{}..{}]>", self.surface, self.start, self.end),
        }
    }
}

impl From<TokenView> for Token {
    /// Converts a core token view into the Ruby-facing token.
    ///
    /// # Arguments
    /// * `view` - The token view.
    ///
    /// # Returns
    /// The corresponding [`Token`].
    fn from(view: TokenView) -> Self {
        Self {
            surface: view.surface,
            // The tag travels as its name, matching the other bindings'
            // token shape.
            pos: view.pos.map(|pos| pos.to_string()),
            start: view.byte_start,
            end: view.byte_end,
        }
    }
}

/// Defines `Litsea::Token`.
///
/// # Arguments
/// * `ruby` - The Ruby handle for the current thread.
/// * `module` - The `Litsea` module to define the class on.
///
/// # Returns
/// `()` on success.
///
/// # Errors
/// Returns a Ruby exception if the class cannot be defined.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("Token", ruby.class_object())?;
    class.define_method("surface", magnus::method!(Token::surface, 0))?;
    class.define_method("pos", magnus::method!(Token::pos, 0))?;
    class.define_method("start", magnus::method!(Token::start, 0))?;
    class.define_method("end", magnus::method!(Token::end, 0))?;
    class.define_method("inspect", magnus::method!(Token::inspect, 0))?;
    class.define_method("to_s", magnus::method!(Token::inspect, 0))?;
    Ok(())
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
        assert_eq!(token.inspect(), "#<Litsea::Token テスト/NOUN [9..18]>");
    }

    #[test]
    fn test_untagged_token() {
        let token = Token::from(TokenView::new("テスト", 0, 9, None));
        assert_eq!(token.pos(), None);
        assert_eq!(token.inspect(), "#<Litsea::Token テスト [0..9]>");
    }
}
