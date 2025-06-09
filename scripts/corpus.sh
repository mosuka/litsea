#!/bin/bash

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
    echo "Usage: $0 [-h] [-t texts_file] [-c corpus_file]"
    exit 1
}

while getopts "ht:c:" opt; do
    case "$opt" in
        h) usage ;;
        t) texts_file="$OPTARG" ;;
        c) corpus_file="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

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
    exit 1
}

###############################################################################
# Call cleanup when SIGINT, SIGTERM, or EXIT is received.
###############################################################################
trap cleanup INT TERM EXIT


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

# Read one line at a time
while IFS= read -r sentence; do
    ## Replace consecutive spaces with a single space
    sentence=$(echo "$sentence" | tr -s ' ')

    # Remove leading and trailing whitespace
    sentence=$(echo "${sentence}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # Skip empty lines
    if [[ -z "$sentence" ]]; then
        continue
    fi

    # Segment the sentence into words using Lindera
    words=$(echo "$sentence" | lindera tokenize -k unidic \
        -o wakati \
        -t 'japanese_compound_word:{"kind":"unidic","tags":["名詞,数詞"],"new_tag":"複合語"}' \
        -t 'japanese_compound_word:{"kind":"unidic","tags":["記号,文字"],"new_tag":"複合語"}')

    ## Replace consecutive spaces with a single space
    words=$(echo "$words" | tr -s ' ')

    # Append the segmented words to the corpus file
    echo "$words" >> "$corpus_file"
done < "$texts_file"

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Creating ${corpus_file} completed."
