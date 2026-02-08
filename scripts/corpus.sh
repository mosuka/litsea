#!/bin/bash

lang="${WIKI_LANG:-ja}"
texts_file="${WIKI_TEXTS_FILE:-texts.txt}"
corpus_file="${WIKI_CORPUS_FILE:-corpus.txt}"

###############################################################################
# usage function
# Displays the usage information for the script.
# Usage: usage
# This function is called when the script is run with the -h option or when an invalid option is provided.
# It prints the usage information and exits the script with a status code of 1.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l lang] [-t texts_file] [-c corpus_file]"
    echo "  -l lang         Language code: ja, ko, zh (default: ja)"
    echo "  -t texts_file   Input texts file (default: texts.txt)"
    echo "  -c corpus_file  Output corpus file (default: corpus.txt)"
    exit 1
}

while getopts "hl:t:c:" opt; do
    case "$opt" in
        h) usage ;;
        l) lang="$OPTARG" ;;
        t) texts_file="$OPTARG" ;;
        c) corpus_file="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

###############################################################################
# Set language-specific lindera options
###############################################################################
case "$lang" in
    ja)
        lindera_dict="embedded://unidic"
        lindera_filters=( \
            --token-filter 'japanese_compound_word:{"kind":"unidic","tags":["名詞,数詞"],"new_tag":"複合語"}' \
            --token-filter 'japanese_compound_word:{"kind":"unidic","tags":["記号,文字"],"new_tag":"複合語"}' \
        )
        ;;
    ko)
        lindera_dict="embedded://ko-dic"
        lindera_filters=()
        ;;
    zh)
        lindera_dict="embedded://cc-cedict"
        lindera_filters=()
        ;;
    *)
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh"
        exit 1
        ;;
esac

echo "Language: ${lang}"
echo "Lindera dict: ${lindera_dict}"
echo "Texts file: ${texts_file}"
echo "Corpus file: ${corpus_file}"

###############################################################################
# spinner definition
###############################################################################
spinner=( '|' '/' '-' '\' )
spin_idx=0

###############################################################################
# cleanup function
# This function is called when the script exits or receives a signal.
# It kills the spinner process and exits the script.
# It is used to ensure that the spinner stops when the script is interrupted.
# Usage: cleanup
###############################################################################
cleanup() {
    if [[ -n "$spinner_pid" ]]; then
        kill "$spinner_pid" 2>/dev/null
    fi
}

###############################################################################
# Call cleanup when SIGINT, SIGTERM, or EXIT is received.
###############################################################################
trap cleanup EXIT
trap 'exit 1' INT TERM


###############################################################################
# spinner_loop function
# This function displays a spinner while a task is running.
# It takes a message as an argument to display.
# Usage: spinner_loop "Your message here"
###############################################################################
spinner_loop() {
    local msg="$1"
    while true; do
        echo -ne "${msg} ... ${spinner[spin_idx]} \r"
        spin_idx=$(( (spin_idx + 1) % ${#spinner[@]} ))
        sleep 0.1
    done
}


###############################################################################
# Create the corpus file
###############################################################################
spinner_loop "Creating ${corpus_file}" &
spinner_pid=$!

# Pre-process the texts file (normalize spaces, remove empty lines),
# then tokenize all lines at once with Lindera, and normalize the output.
sed 's/^[[:space:]]*//;s/[[:space:]]*$//' "$texts_file" | \
    tr -s ' ' | \
    sed '/^$/d' | \
    lindera tokenize --dict "${lindera_dict}" \
        --output wakati \
        "${lindera_filters[@]}" | \
    tr -s ' ' > "$corpus_file"

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Creating ${corpus_file} completed."
