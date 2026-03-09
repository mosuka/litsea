#!/bin/bash
set -euo pipefail

pos=false

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-p] <conllu_file> <output_file>"
    echo ""
    echo "Convert a CoNLL-U file to Litsea corpus format."
    echo ""
    echo "  <conllu_file>   Path to the input CoNLL-U file"
    echo "  <output_file>   Path to the output corpus file"
    echo "  -p              Output POS-tagged corpus (word/POS format)"
    echo "                  Without -p, outputs space-separated words"
    exit 1
}

while getopts "hp" opt; do
    case "$opt" in
        h) usage ;;
        p) pos=true ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage
fi

conllu_file="$1"
output_file="$2"

if [ ! -f "${conllu_file}" ]; then
    echo "Error: CoNLL-U file not found: ${conllu_file}" >&2
    exit 1
fi

###############################################################################
# convert_conllu function
# Converts a CoNLL-U file to Litsea corpus format using awk.
#
# Skips:
#   - Comment lines (starting with '#')
#   - Multi-word tokens (ID contains '-', e.g. "1-2")
#   - Empty nodes (ID contains '.', e.g. "1.1")
#   - Tokens with unannotated UPOS ('_')
# Blank lines mark sentence boundaries.
#
# Usage: convert_conllu <input_file> <output_file> [--pos]
#   --pos: output "word/POS" format instead of space-separated words
###############################################################################
convert_conllu() {
    local input_file="$1"
    local output_file="$2"
    local with_pos="${3:-}"

    awk -F'\t' -v with_pos="${with_pos}" '
    BEGIN { sentence = ""; count = 0 }
    /^[[:space:]]*$/ {
        # Blank line = sentence boundary
        if (sentence != "") {
            print sentence > output
            count++
            sentence = ""
        }
        next
    }
    /^#/ { next }
    {
        if (NF < 4) next
        id = $1; form = $2; upos = $4
        if (index(id, "-") > 0) next
        if (index(id, ".") > 0) next
        if (upos == "_") next
        if (with_pos == "--pos") {
            token = form "/" upos
        } else {
            token = form
        }
        if (sentence == "") {
            sentence = token
        } else {
            sentence = sentence " " token
        }
    }
    END {
        # Handle remaining tokens at end of file (files without trailing newline)
        if (sentence != "") {
            print sentence > output
            count++
        }
        printf "Converted %d sentences.\n", count > "/dev/stderr"
    }
    ' output="${output_file}" "${input_file}"
}

###############################################################################
# Convert CoNLL-U to Litsea corpus format
###############################################################################
if [ "${pos}" = true ]; then
    echo "Converting to POS corpus: ${output_file}"
    convert_conllu "${conllu_file}" "${output_file}" --pos
else
    echo "Converting to word segmentation corpus: ${output_file}"
    convert_conllu "${conllu_file}" "${output_file}"
fi

echo "Done."
