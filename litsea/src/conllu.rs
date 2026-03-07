use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

/// Represents a single token line in a CoNLL-U file.
#[derive(Debug, Clone)]
struct Token {
    /// Surface form (FORM).
    form: String,
    /// UPOS part-of-speech tag.
    upos: String,
}

/// Converts a CoNLL-U file to Litsea corpus format.
///
/// CoNLL-U format:
/// ```text
/// # text = 太郎は花子が読んでいる本を次郎に渡した。
/// 1    太郎    太郎    PROPN    _    _    12    nsubj    _    SpaceAfter=No
/// 2    は    は    ADP    _    _    1    case    _    SpaceAfter=No
/// ...
/// ```
///
/// Output format depends on the `with_pos` flag:
///
/// When `with_pos` is `true` (POS corpus format):
/// ```text
/// 太郎/PROPN は/ADP 花子/PROPN が/ADP ...
/// ```
///
/// When `with_pos` is `false` (word segmentation corpus format):
/// ```text
/// 太郎 は 花子 が ...
/// ```
///
/// # Arguments
///
/// * `input_path` - Path to the CoNLL-U file.
/// * `output_path` - Path to the output Litsea corpus file.
/// * `with_pos` - If `true`, output includes POS tags (`word/POS`).
///   If `false`, output is space-separated words only.
///
/// # Returns
///
/// The number of sentences converted.
pub fn convert_conllu(
    input_path: &Path,
    output_path: &Path,
    with_pos: bool,
) -> Result<usize, Box<dyn Error>> {
    let input_file = File::open(input_path)?;
    let reader = io::BufReader::new(input_file);

    let output_file = File::create(output_path)?;
    let mut writer = io::BufWriter::new(output_file);

    let mut sentence_tokens: Vec<Token> = Vec::new();
    let mut sentence_count = 0usize;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() {
            // Empty line = sentence boundary
            if !sentence_tokens.is_empty() {
                write_sentence(&mut writer, &sentence_tokens, with_pos)?;
                sentence_count += 1;
                sentence_tokens.clear();
            }
            continue;
        }

        // Skip comment lines
        if line.starts_with('#') {
            continue;
        }

        // Skip lines with fewer than 4 tab-separated fields
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 4 {
            continue;
        }

        let id = fields[0];
        // Skip multi-word tokens ("1-2") and empty nodes ("1.1")
        if id.contains('-') || id.contains('.') {
            continue;
        }

        let form = fields[1];
        let upos = fields[3];

        // Skip unannotated tokens (UPOS = "_")
        if upos == "_" {
            continue;
        }

        sentence_tokens.push(Token {
            form: form.to_string(),
            upos: upos.to_string(),
        });
    }

    // Handle remaining tokens at end of file (files without trailing newline)
    if !sentence_tokens.is_empty() {
        write_sentence(&mut writer, &sentence_tokens, with_pos)?;
        sentence_count += 1;
    }

    writer.flush()?;
    Ok(sentence_count)
}

/// Writes a single sentence in Litsea corpus format.
///
/// # Arguments
///
/// * `writer` - The writer to write to.
/// * `tokens` - The tokens in the sentence.
/// * `with_pos` - If `true`, writes `word/POS` format. If `false`, writes space-separated words.
fn write_sentence<W: Write>(writer: &mut W, tokens: &[Token], with_pos: bool) -> io::Result<()> {
    if with_pos {
        let formatted: Vec<String> =
            tokens.iter().map(|t| format!("{}/{}", t.form, t.upos)).collect();
        writeln!(writer, "{}", formatted.join(" "))
    } else {
        let words: Vec<&str> = tokens.iter().map(|t| t.form.as_str()).collect();
        writeln!(writer, "{}", words.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::io::Write;

    use tempfile::NamedTempFile;

    fn sample_conllu_input() -> String {
        "# text = 太郎は走った。\n\
         1\t太郎\t太郎\tPROPN\t_\t_\t3\tnsubj\t_\tSpaceAfter=No\n\
         2\tは\tは\tADP\t_\t_\t1\tcase\t_\tSpaceAfter=No\n\
         3\t走っ\t走る\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No\n\
         4\tた\tた\tAUX\t_\t_\t3\taux\t_\tSpaceAfter=No\n\
         5\t。\t。\tPUNCT\t_\t_\t3\tpunct\t_\tSpaceAfter=No\n\
         \n"
        .to_string()
    }

    #[test]
    fn test_convert_conllu_with_pos() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(input, "{}", sample_conllu_input())?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), true)?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "太郎/PROPN は/ADP 走っ/VERB た/AUX 。/PUNCT");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_without_pos() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(input, "{}", sample_conllu_input())?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), false)?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "太郎 は 走っ た 。");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_multiple_sentences() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(
            input,
            "# sent_id = 1\n\
             1\t猫\t猫\tNOUN\t_\t_\t0\troot\t_\t_\n\
             \n\
             # sent_id = 2\n\
             1\t犬\t犬\tNOUN\t_\t_\t0\troot\t_\t_\n\
             \n"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), true)?;

        assert_eq!(count, 2);

        let content = fs::read_to_string(output.path())?;
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "猫/NOUN");
        assert_eq!(lines[1], "犬/NOUN");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_multiple_sentences_without_pos() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(
            input,
            "# sent_id = 1\n\
             1\t猫\t猫\tNOUN\t_\t_\t0\troot\t_\t_\n\
             \n\
             # sent_id = 2\n\
             1\t犬\t犬\tNOUN\t_\t_\t0\troot\t_\t_\n\
             \n"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), false)?;

        assert_eq!(count, 2);

        let content = fs::read_to_string(output.path())?;
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "猫");
        assert_eq!(lines[1], "犬");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_skip_multiword() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        // Multi-word token lines ("1-2") are skipped
        write!(
            input,
            "1-2\tでは\t_\t_\t_\t_\t_\t_\t_\t_\n\
             1\tで\tで\tADP\t_\t_\t0\troot\t_\t_\n\
             2\tは\tは\tADP\t_\t_\t1\tcase\t_\t_\n\
             \n"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), true)?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "で/ADP は/ADP");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_empty_file() -> Result<(), Box<dyn Error>> {
        let input = NamedTempFile::new()?;
        let output = NamedTempFile::new()?;

        let count = convert_conllu(input.path(), output.path(), true)?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_convert_conllu_no_trailing_newline() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        // File without trailing newline
        write!(input, "1\t花\t花\tNOUN\t_\t_\t0\troot\t_\t_")?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), true)?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "花/NOUN");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_no_trailing_newline_without_pos() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(input, "1\t花\t花\tNOUN\t_\t_\t0\troot\t_\t_")?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path(), false)?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "花");

        Ok(())
    }
}
