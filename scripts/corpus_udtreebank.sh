#!/bin/bash
set -euo pipefail

pos=false
spaced=false

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-p] [-s] <conllu_file> <output_file>"
    echo ""
    echo "Convert a CoNLL-U file to Litsea corpus format. -p and -s are"
    echo "independent and may be combined."
    echo ""
    echo "  <conllu_file>   Path to the input CoNLL-U file"
    echo "  <output_file>   Path to the output corpus file"
    echo "  -p              Output word/POS tokens instead of plain words"
    echo "  -s              Output a space-preserving TSV corpus: tab-separated"
    echo "                  tokens, with a literal space token wherever the"
    echo "                  original text had a space (from SpaceAfter=No"
    echo "                  annotations). For use with 'litsea extract"
    echo "                  --format tsv'."
    echo "  -p -s together  Output a space-preserving TSV corpus of word/POS"
    echo "                  tokens (issue #196), for measuring a two-stage"
    echo "                  POS model's real-world (spaced) quality with"
    echo "                  'litsea evaluate --pos --format tsv'. Space"
    echo "                  tokens carry no /POS suffix (CoNLL-U has no UPOS"
    echo "                  annotation for whitespace itself)."
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
#   - Multi-word token range lines (ID contains '-', e.g. "1-2"; but see
#     the spaced argument below for how their surface spacing is handled)
#   - Empty nodes (ID contains '.', e.g. "1.1")
#   - Tokens with unannotated UPOS ('_')
# Blank lines mark sentence boundaries.
#
# Usage: convert_conllu <input_file> <output_file> <pos:true|false> <spaced:true|false>
#   pos:    when true, emit "word/POS" tokens instead of plain words. A
#           literal space token (see spaced below) never gets a /POS
#           suffix, since CoNLL-U has no UPOS annotation for whitespace.
#   spaced: when true, emit tab-separated tokens, inserting a literal space
#           token between tokens wherever the original text had a space
#           (i.e. the previous token's MISC lacks SpaceAfter=No).
#           Multi-word token (MWT) ranges like "3-4 don't" carry the
#           surface spacing: member words are emitted with no space
#           tokens between them (concatenating the member FORMs must
#           reproduce the range FORM -- true for every MWT in UD English
#           EWT; if a future treebank violates this, fall back to
#           emitting the range FORM as a single token), and the range
#           line's own SpaceAfter applies after the last member word.
#           Known limitation: a rare MISC field, SpacesAfter=<escaped>,
#           can specify a non-space separator (e.g. a no-break space);
#           this script always inserts an ordinary U+0020 space token,
#           so the reconstructed text can differ by that one character.
#           Whitespace tokens are excluded from evaluation scoring, so
#           this has no effect on measured quality.
#   pos and spaced are independent and may both be true (issue #196), in
#   which case the output is a space-preserving TSV corpus of "word/POS"
#   tokens.
###############################################################################
convert_conllu() {
    local input_file="$1"
    local output_file="$2"
    local pos_flag="${3:-false}"
    local spaced_flag="${4:-false}"

    awk -F'\t' -v pos="${pos_flag}" -v spaced="${spaced_flag}" '
    BEGIN { sentence = ""; count = 0; prev_space = 0; in_mwt = 0; mwt_end = 0; mwt_space = 0 }
    /^[[:space:]]*$/ {
        # Blank line = sentence boundary
        if (sentence != "") {
            print sentence > output
            count++
            sentence = ""
        }
        prev_space = 0
        in_mwt = 0
        next
    }
    /^#/ { next }
    {
        if (NF < 4) next
        id = $1; form = $2; upos = $4
        if (index(id, "-") > 0) {
            if (spaced == "true") {
                # Multi-word token range (e.g. a contraction spanning ids
                # 3-4): the surface spacing belongs to this line. Member
                # words join with no space tokens in between; the SpaceAfter
                # of the range applies after the last member word.
                split(id, range, "-")
                mwt_end = range[2] + 0
                mwt_space = (index($10, "SpaceAfter=No") == 0) ? 1 : 0
                in_mwt = 1
            }
            next
        }
        if (index(id, ".") > 0) next
        if (upos == "_") next
        token = (pos == "true") ? form "/" upos : form
        if (spaced == "true") {
            if (sentence == "") {
                sentence = token
            } else if (prev_space) {
                sentence = sentence "\t \t" token
            } else {
                sentence = sentence "\t" token
            }
            if (in_mwt) {
                if (id + 0 == mwt_end) {
                    prev_space = mwt_space
                    in_mwt = 0
                } else {
                    prev_space = 0
                }
            } else {
                prev_space = (index($10, "SpaceAfter=No") == 0) ? 1 : 0
            }
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
if [ "${pos}" = true ] && [ "${spaced}" = true ]; then
    echo "Converting to space-preserving POS corpus: ${output_file}"
elif [ "${pos}" = true ]; then
    echo "Converting to POS corpus: ${output_file}"
elif [ "${spaced}" = true ]; then
    echo "Converting to space-preserving TSV corpus: ${output_file}"
else
    echo "Converting to word segmentation corpus: ${output_file}"
fi
convert_conllu "${conllu_file}" "${output_file}" "${pos}" "${spaced}"

echo "Done."
