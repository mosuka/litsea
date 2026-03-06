use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

/// CoNLL-Uファイルの1トークン行を表す。
#[derive(Debug, Clone)]
struct Token {
    /// 表層形 (FORM)
    form: String,
    /// UPOS品詞タグ
    upos: String,
}

/// CoNLL-UファイルをLitsea品詞コーパス形式に変換する。
///
/// CoNLL-Uフォーマット:
/// ```text
/// # text = 太郎は花子が読んでいる本を次郎に渡した。
/// 1	太郎	太郎	PROPN	_	_	12	nsubj	_	SpaceAfter=No
/// 2	は	は	ADP	_	_	1	case	_	SpaceAfter=No
/// ...
/// ```
///
/// 出力フォーマット（Litseaコーパス形式）:
/// ```text
/// 太郎/PROPN は/ADP 花子/PROPN が/ADP ...
/// ```
///
/// # Arguments
/// * `input_path` - CoNLL-Uファイルのパス
/// * `output_path` - Litseaコーパス出力ファイルのパス
pub fn convert_conllu(input_path: &Path, output_path: &Path) -> Result<usize, Box<dyn Error>> {
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
            // 空行 = 文の区切り
            if !sentence_tokens.is_empty() {
                write_sentence(&mut writer, &sentence_tokens)?;
                sentence_count += 1;
                sentence_tokens.clear();
            }
            continue;
        }

        // コメント行はスキップ
        if line.starts_with('#') {
            continue;
        }

        // マルチワードトークン行（"1-2"のようなID）はスキップ
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 4 {
            continue;
        }

        let id = fields[0];
        // マルチワードトークン（"1-2"）や空ノード（"1.1"）はスキップ
        if id.contains('-') || id.contains('.') {
            continue;
        }

        let form = fields[1];
        let upos = fields[3];

        // UPOSが "_" の場合はスキップ（未注釈トークン）
        if upos == "_" {
            continue;
        }

        sentence_tokens.push(Token {
            form: form.to_string(),
            upos: upos.to_string(),
        });
    }

    // ファイル末尾の残り（空行なしで終わるファイル対応）
    if !sentence_tokens.is_empty() {
        write_sentence(&mut writer, &sentence_tokens)?;
        sentence_count += 1;
    }

    writer.flush()?;
    Ok(sentence_count)
}

/// 1文をLitseaコーパス形式で書き出す。
fn write_sentence<W: Write>(writer: &mut W, tokens: &[Token]) -> io::Result<()> {
    let formatted: Vec<String> = tokens
        .iter()
        .map(|t| format!("{}/{}", t.form, t.upos))
        .collect();
    writeln!(writer, "{}", formatted.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::io::Write;

    use tempfile::NamedTempFile;

    #[test]
    fn test_convert_conllu_basic() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        write!(
            input,
            "# text = 太郎は走った。\n\
             1\t太郎\t太郎\tPROPN\t_\t_\t3\tnsubj\t_\tSpaceAfter=No\n\
             2\tは\tは\tADP\t_\t_\t1\tcase\t_\tSpaceAfter=No\n\
             3\t走っ\t走る\tVERB\t_\t_\t0\troot\t_\tSpaceAfter=No\n\
             4\tた\tた\tAUX\t_\t_\t3\taux\t_\tSpaceAfter=No\n\
             5\t。\t。\tPUNCT\t_\t_\t3\tpunct\t_\tSpaceAfter=No\n\
             \n"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path())?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "太郎/PROPN は/ADP 走っ/VERB た/AUX 。/PUNCT");

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
        let count = convert_conllu(input.path(), output.path())?;

        assert_eq!(count, 2);

        let content = fs::read_to_string(output.path())?;
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "猫/NOUN");
        assert_eq!(lines[1], "犬/NOUN");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_skip_multiword() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        // マルチワードトークン行（1-2）はスキップされる
        write!(
            input,
            "1-2\tでは\t_\t_\t_\t_\t_\t_\t_\t_\n\
             1\tで\tで\tADP\t_\t_\t0\troot\t_\t_\n\
             2\tは\tは\tADP\t_\t_\t1\tcase\t_\t_\n\
             \n"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path())?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "で/ADP は/ADP");

        Ok(())
    }

    #[test]
    fn test_convert_conllu_empty_file() -> Result<(), Box<dyn Error>> {
        let input = NamedTempFile::new()?;
        let output = NamedTempFile::new()?;

        let count = convert_conllu(input.path(), output.path())?;
        assert_eq!(count, 0);

        Ok(())
    }

    #[test]
    fn test_convert_conllu_no_trailing_newline() -> Result<(), Box<dyn Error>> {
        let mut input = NamedTempFile::new()?;
        // 末尾に空行がないファイル
        write!(
            input,
            "1\t花\t花\tNOUN\t_\t_\t0\troot\t_\t_"
        )?;
        input.as_file().sync_all()?;

        let output = NamedTempFile::new()?;
        let count = convert_conllu(input.path(), output.path())?;

        assert_eq!(count, 1);

        let content = fs::read_to_string(output.path())?;
        assert_eq!(content.trim(), "花/NOUN");

        Ok(())
    }
}
