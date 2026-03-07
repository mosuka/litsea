#!/bin/bash
set -euo pipefail

lang="${LANG_CODE:-ja}"
corpus_file="${CORPUS_FILE:-corpus.txt}"
pos_corpus_file="${POS_CORPUS_FILE:-pos_corpus.txt}"
litsea_cli="${LITSEA_CLI:-cargo run --bin litsea --}"

###############################################################################
# usage function
# Displays the usage information for the script.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l lang] [-c corpus_file] [-p pos_corpus_file]"
    echo ""
    echo "Download a UD Treebank and convert it to Litsea corpus format."
    echo ""
    echo "  -l lang             Language code: ja, ko, zh (default: ja)"
    echo "  -c corpus_file      Output corpus file for word segmentation (default: corpus.txt)"
    echo "  -p pos_corpus_file  Output POS corpus file (default: pos_corpus.txt)"
    exit 1
}

while getopts "hl:c:p:" opt; do
    case "$opt" in
        h) usage ;;
        l) lang="$OPTARG" ;;
        c) corpus_file="$OPTARG" ;;
        p) pos_corpus_file="$OPTARG" ;;
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
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh"
        exit 1
        ;;
esac

ud_url="https://github.com/UniversalDependencies/${ud_repo}.git"
ud_dir="/tmp/${ud_repo}"
conllu_file="${ud_dir}/${ud_prefix}-train.conllu"

echo "Language: ${lang}"
echo "UD Treebank: ${ud_repo}"
echo "Corpus file: ${corpus_file}"
echo "POS corpus file: ${pos_corpus_file}"

###############################################################################
# Download UD Treebank (clone if not already present)
###############################################################################
if [ -d "${ud_dir}" ]; then
    echo "UD Treebank already exists at ${ud_dir}, skipping download."
else
    echo "Cloning ${ud_url} ..."
    git clone --depth 1 "${ud_url}" "${ud_dir}"
    echo "Cloning completed."
fi

if [ ! -f "${conllu_file}" ]; then
    echo "Error: CoNLL-U file not found: ${conllu_file}"
    exit 1
fi

###############################################################################
# Convert CoNLL-U to word segmentation corpus (space-separated words)
###############################################################################
echo "Converting to word segmentation corpus: ${corpus_file}"
${litsea_cli} convert-conllu "${conllu_file}" "${corpus_file}"

###############################################################################
# Convert CoNLL-U to POS corpus (word/POS format)
###############################################################################
echo "Converting to POS corpus: ${pos_corpus_file}"
${litsea_cli} convert-conllu --pos "${conllu_file}" "${pos_corpus_file}"

echo "Done."
