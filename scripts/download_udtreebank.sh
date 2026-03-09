#!/bin/bash
set -euo pipefail

lang="${LANG_CODE:-ja}"
output_dir="."

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l lang] [-o output_dir]"
    echo ""
    echo "Download a UD Treebank and print the path to the training CoNLL-U file."
    echo ""
    echo "  -l lang        Language code: ja, ko, zh (default: ja)"
    echo "  -o output_dir  Directory to clone the treebank into (default: current directory)"
    exit 1
}

while getopts "hl:o:" opt; do
    case "$opt" in
        h) usage ;;
        l) lang="$OPTARG" ;;
        o) output_dir="${OPTARG%/}" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

###############################################################################
# Set language-specific UD Treebank variables
###############################################################################
case "$lang" in
    ja)
        ud_repo="UD_Japanese-GSD"
        ud_prefix="ja_gsd-ud"
        ;;
    ko)
        ud_repo="UD_Korean-GSD"
        ud_prefix="ko_gsd-ud"
        ;;
    zh)
        ud_repo="UD_Chinese-GSD"
        ud_prefix="zh_gsd-ud"
        ;;
    *)
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh" >&2
        exit 1
        ;;
esac

ud_url="https://github.com/UniversalDependencies/${ud_repo}.git"
ud_dir="${output_dir}/${ud_repo}"
conllu_file="${ud_dir}/${ud_prefix}-train.conllu"

###############################################################################
# Download UD Treebank (clone if not already present)
###############################################################################
if [ -d "${ud_dir}" ]; then
    echo "UD Treebank already exists at ${ud_dir}, skipping download." >&2
else
    echo "Cloning ${ud_url} ..." >&2
    git clone --depth 1 "${ud_url}" "${ud_dir}" >&2
    echo "Cloning completed." >&2
fi

if [ ! -f "${conllu_file}" ]; then
    echo "Error: CoNLL-U file not found: ${conllu_file}" >&2
    exit 1
fi

# Print the path to stdout so callers can capture it
echo "${conllu_file}"
