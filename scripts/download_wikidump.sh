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
    echo "Download a Wikipedia dump and print the path to the downloaded file."
    echo ""
    echo "  -l lang        Language code: ja, ko, zh (default: ja)"
    echo "  -o output_dir  Directory to save the dump file (default: current directory)"
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
# Set language-specific Wikipedia dump variables
###############################################################################
case "$lang" in
    ja)
        wiki_prefix="jawiki"
        ;;
    ko)
        wiki_prefix="kowiki"
        ;;
    zh)
        wiki_prefix="zhwiki"
        ;;
    *)
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh" >&2
        exit 1
        ;;
esac

dump_url="https://dumps.wikimedia.org/${wiki_prefix}/latest/${wiki_prefix}-latest-pages-articles.xml.bz2"
dump_file="${output_dir}/${wiki_prefix}-latest-pages-articles.xml.bz2"

###############################################################################
# Download Wikipedia dump (skip if already present)
###############################################################################
mkdir -p "${output_dir}"

if [ -f "${dump_file}" ]; then
    echo "Wikipedia dump already exists at ${dump_file}, skipping download." >&2
else
    echo "Downloading ${dump_url} ..." >&2
    curl -L -o "${dump_file}" "${dump_url}" >&2
    echo "Download completed." >&2
fi

# Print the path to stdout so callers can capture it
echo "${dump_file}"
