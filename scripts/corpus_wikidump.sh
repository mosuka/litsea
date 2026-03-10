#!/bin/bash
set -euo pipefail

# Corpus size guidelines (sentence lines after filtering):
#   ~10,000    Minimum for prototyping and smoke tests
#   50,000-100,000   Practical range for model training
#   100,000-500,000  High-quality, robust models
#   0 (unlimited)    Use full dump for maximum accuracy

max_lines=100000
lang="ja"

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l lang] [-n max_lines] <dump_file> <output_file>"
    echo ""
    echo "Extract plain text from a Wikipedia dump using wicket,"
    echo "tokenize with lindera, and produce a corpus file."
    echo ""
    echo "  <dump_file>     Path to Wikipedia dump file (e.g. jawiki-latest-pages-articles.xml.bz2)"
    echo "  <output_file>   Path to output corpus file"
    echo "  -l lang         Language code: ja, ko, zh (default: ja)"
    echo "  -n max_lines    Maximum number of text lines to process (default: 100000, 0 = unlimited)"
    exit 1
}

while getopts "hl:n:" opt; do
    case "$opt" in
        h) usage ;;
        l) lang="$OPTARG" ;;
        n) max_lines="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage
fi

dump_file="$1"
output_file="$2"

###############################################################################
# Language-specific settings
###############################################################################
case "${lang}" in
    ja)
        lindera_dict="embedded://unidic"
        lindera_filters='-t japanese_compound_word:{"kind":"unidic","tags":["名詞,数詞"],"new_tag":"複合語"}'
        ;;
    ko)
        lindera_dict="embedded://ko-dic"
        lindera_filters=""
        ;;
    zh)
        lindera_dict="embedded://cc-cedict"
        lindera_filters=""
        ;;
    *)
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh" >&2
        exit 1
        ;;
esac

if [ ! -f "${dump_file}" ]; then
    echo "Error: Dump file not found: ${dump_file}" >&2
    exit 1
fi

###############################################################################
# Check required commands
###############################################################################
for cmd in wicket lindera; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: '${cmd}' is not installed or not in PATH." >&2
        exit 1
    fi
done

###############################################################################
# Create temporary directory and register cleanup
###############################################################################
tmp_dir=$(mktemp -d /tmp/litsea-wikidump.XXXXXX)

cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT
trap 'exit 1' INT TERM

###############################################################################
# Extract plain text from Wikipedia dump using wicket
###############################################################################
echo "Extracting text from ${dump_file} to ${tmp_dir} ..." >&2
wicket "${dump_file}" -o "${tmp_dir}"
echo "Extraction completed." >&2

###############################################################################
# Build corpus: collect text lines, shuffle, limit, tokenize in one pipeline
###############################################################################
echo "Building corpus: ${output_file} (lang=${lang}, max_lines=${max_lines}) ..." >&2

# Collect wiki files in shuffled order to reduce category bias
mapfile -t wiki_files < <(find "${tmp_dir}" -type f -name 'wiki_*' | shuf)

# Stream pipeline:
#   1. cat all wiki files (shuffled order)
#   2. Strip <doc> / </doc> tag lines and empty lines
#   3. Keep only lines that end with sentence-ending punctuation
#   4. Keep only lines with 20+ characters (filter out short headers)
#   5. Apply line limit (head -n; 0 = unlimited via cat)
#   6. Tokenize all lines in a single lindera process
if [ "${max_lines}" -gt 0 ]; then
    limit_cmd="head -n ${max_lines}"
else
    limit_cmd="cat"
fi

# Build lindera command
lindera_cmd="lindera tokenize -d \"${lindera_dict}\" -o wakati"
if [ -n "${lindera_filters}" ]; then
    lindera_cmd="${lindera_cmd} -t '${lindera_filters}'"
fi

cat "${wiki_files[@]}" \
    | grep -v '^<doc ' \
    | grep -v '^</doc>$' \
    | grep -v '^[[:space:]]*$' \
    | grep '[。.!?]$' \
    | awk 'length >= 20' \
    | ${limit_cmd} \
    | eval ${lindera_cmd} \
    > "${output_file}"

line_count=$(wc -l < "${output_file}")
echo "Corpus created: ${output_file} (${line_count} lines)" >&2
echo "Done." >&2
