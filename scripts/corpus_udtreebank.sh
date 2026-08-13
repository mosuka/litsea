#!/bin/bash
set -euo pipefail

pos=false
spaced=false

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-p|-s] <conllu_file> <output_file>"
    echo ""
    echo "Convert a CoNLL-U file to Litsea corpus format."
    echo ""
    echo "  <conllu_file>   Path to the input CoNLL-U file"
    echo "  <output_file>   Path to the output corpus file"
    echo "  -p              Output POS-tagged corpus (word/POS format)"
    echo "                  Without -p, outputs space-separated words"
    echo "  -s              Output space-preserving TSV corpus: tab-separated"
    echo "                  tokens, with a literal space token wherever the"
    echo "                  original text had a space (from SpaceAfter=No"
    echo "                  annotations). For use with 'litsea extract"
    echo "                  --format tsv'. Cannot be combined with -p."
    exit 1
}

while getopts "hps" opt; do
    case "$opt" in
        h) usage ;;
        p) pos=true ;;
        s) spaced=true ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage
fi

if [ "${pos}" = true ] && [ "${spaced}" = true ]; then
    echo "Error: -p and -s cannot be combined (the POS pipeline has no TSV variant)" >&2
    exit 1
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
# Usage: convert_conllu <input_file> <output_file> [--pos|--spaced]
#   --pos:    output "word/POS" format instead of space-separated words
#   --spaced: output tab-separated tokens, inserting a literal space token
#             between tokens wherever the original text had a space
#             (i.e. the previous token's MISC lacks SpaceAfter=No). Note:
#             multi-word token ranges are skipped, so SpaceAfter carried on
#             a range line is not seen; the GSD treebanks used here do not
#             rely on that.
###############################################################################
convert_conllu() {
    local input_file="$1"
    local output_file="$2"
    local mode="${3:-}"

    awk -F'\t' -v mode="${mode}" '
    BEGIN { sentence = ""; count = 0; prev_space = 0 }
    /^[[:space:]]*$/ {
        # Blank line = sentence boundary
        if (sentence != "") {
            print sentence > output
            count++
            sentence = ""
        }
        prev_space = 0
        next
    }
    /^#/ { next }
    {
        if (NF < 4) next
        id = $1; form = $2; upos = $4
        if (index(id, "-") > 0) next
        if (index(id, ".") > 0) next
        if (upos == "_") next
        if (mode == "--pos") {
            token = form "/" upos
        } else {
            token = form
        }
        if (mode == "--spaced") {
            if (sentence == "") {
                sentence = token
            } else if (prev_space) {
                sentence = sentence "\t \t" token
            } else {
                sentence = sentence "\t" token
            }
            prev_space = (index($10, "SpaceAfter=No") == 0) ? 1 : 0
        } else if (sentence == "") {
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
elif [ "${spaced}" = true ]; then
    echo "Converting to space-preserving TSV corpus: ${output_file}"
    convert_conllu "${conllu_file}" "${output_file}" --spaced
else
    echo "Converting to word segmentation corpus: ${output_file}"
    convert_conllu "${conllu_file}" "${output_file}"
fi

echo "Done."
