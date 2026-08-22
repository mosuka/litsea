//! Language arguments.
//!
//! Ruby callers reach for a Symbol (`:japanese`) as readily as a String, and
//! magnus converts neither into the other automatically, so both are
//! accepted here.

use litsea::Language;
use litsea_binding_core::parse_language;
use magnus::{Symbol, TryConvert, Value, error::Error};

use crate::error::map_err;

/// Converts a Ruby String or Symbol into a [`Language`].
///
/// # Arguments
/// * `value` - The language name or ISO 639-1 code, as a String or Symbol.
///
/// # Returns
/// The parsed language.
///
/// # Errors
/// Raises `Litsea::InvalidArgumentError` for an unknown language, or a
/// `TypeError` if the value is neither a String nor a Symbol.
pub fn language_from_value(value: Value) -> Result<Language, Error> {
    let name = match Symbol::from_value(value) {
        Some(symbol) => symbol.name()?.to_string(),
        None => String::try_convert(value)?,
    };

    map_err(parse_language(&name))
}
