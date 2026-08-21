//! FFI-independent token representation.

use litsea::Upos;

/// A segmented token as the bindings expose it.
///
/// `litsea` returns tokens as `Vec<String>` (segmentation) or
/// `Vec<(String, Upos)>` (POS tagging), neither of which carries the token's
/// position in the input. Byte offsets are useful to host-language callers
/// (highlighting, span alignment), and they are recoverable because the
/// tokens tile the input exactly, so [`TokenView`] carries them for both
/// modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenView {
    /// The token's surface form.
    pub surface: String,
    /// Starting byte offset in the input string.
    pub byte_start: usize,
    /// Ending byte offset (exclusive) in the input string.
    pub byte_end: usize,
    /// The UPOS tag, or `None` when the segmenter has no POS model.
    pub pos: Option<Upos>,
}

impl TokenView {
    /// Creates a token view.
    ///
    /// # Arguments
    /// * `surface` - The token's surface form.
    /// * `byte_start` - Starting byte offset in the input string.
    /// * `byte_end` - Ending byte offset (exclusive) in the input string.
    /// * `pos` - The UPOS tag, or `None` for segmentation-only output.
    ///
    /// # Returns
    /// The new [`TokenView`].
    pub fn new(
        surface: impl Into<String>,
        byte_start: usize,
        byte_end: usize,
        pos: Option<Upos>,
    ) -> Self {
        Self {
            surface: surface.into(),
            byte_start,
            byte_end,
            pos,
        }
    }

    /// Returns the UPOS tag as its canonical uppercase name.
    ///
    /// # Returns
    /// The tag name (for example `"NOUN"`), or `None` when the token has no
    /// POS tag.
    #[must_use]
    pub fn pos_name(&self) -> Option<String> {
        self.pos.map(|pos| pos.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_name() {
        let tagged = TokenView::new("すもも", 0, 9, Some(Upos::NOUN));
        assert_eq!(tagged.pos_name().as_deref(), Some("NOUN"));

        let untagged = TokenView::new("すもも", 0, 9, None);
        assert_eq!(untagged.pos_name(), None);
    }

    #[test]
    fn test_offsets_slice_the_input() {
        let sentence = "すももももも";
        let token = TokenView::new("すもも", 0, 9, None);
        assert_eq!(&sentence[token.byte_start..token.byte_end], token.surface);
    }
}
