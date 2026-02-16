#!/bin/bash
set -euo pipefail

# Default value (uses the value defined in the environment variables, if defined)
lang="${WIKI_LANG:-ja}"
timestamp="${WIKI_TIMESTAMP:-latest}"
title_count="${WIKI_TITLE_COUNT:-1000}"
texts_file="${WIKI_TEXTS_FILE:-wiki_texts.txt}"
litsea_cli="${LITSEA_CLI:-cargo run --bin litsea --}"

###############################################################################
# usage function
# Displays the usage information for the script.
# Usage: usage
# This function is called when the script is run with the -h option or when an invalid option is provided.
# It prints the usage information and exits the script with a status code of 1.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l lang] [-t timestamp] [-c title_count] [-o texts_file]"
    echo "  -l lang         Language code: ja, ko, zh (default: ja)"
    echo "  -t timestamp    Wikipedia dump timestamp (default: latest)"
    echo "  -c title_count  Number of titles to fetch (default: 1000)"
    echo "  -o texts_file   Output texts file (default: wiki_texts.txt)"
    exit 1
}

while getopts "hl:t:c:o:" opt; do
    case "$opt" in
        h) usage ;;
        l) lang="$OPTARG" ;;
        t) timestamp="$OPTARG" ;;
        c) title_count="$OPTARG" ;;
        o) texts_file="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

###############################################################################
# Set language-specific variables
###############################################################################
case "$lang" in
    ja)
        wiki_prefix="jawiki"
        wiki_domain="ja.wikipedia.org"
        ;;
    ko)
        wiki_prefix="kowiki"
        wiki_domain="ko.wikipedia.org"
        ;;
    zh)
        wiki_prefix="zhwiki"
        wiki_domain="zh.wikipedia.org"
        ;;
    *)
        echo "Error: Unsupported language '${lang}'. Supported: ja, ko, zh"
        exit 1
        ;;
esac

echo "Language: ${lang}"
echo "Timestamp: ${timestamp}"
echo "Title count: ${title_count}"
echo "Texts file: ${texts_file}"


file_name="${wiki_prefix}-${timestamp}-pages-articles-multistream-index.txt"
download_dir=/tmp
download_file="${file_name}.bz2"
download_url="https://dumps.wikimedia.org/${wiki_prefix}/${timestamp}/${download_file}"

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
    if [[ -n "${tmpfile:-}" && -f "$tmpfile" ]]; then
        rm -f "$tmpfile"
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
# Download dump file
###############################################################################
spinner_loop "Downloading ${download_url}" &
spinner_pid=$!

# Start curl in the background and obtain the process ID.
curl -s -o "${download_dir}/${download_file}" "${download_url}"

# Stop the spinner after the download is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Downloading ${download_url} completed."


###############################################################################
# Decompressing dump file
###############################################################################
spinner_loop "Decompressing ${download_dir}/${download_file}" &
spinner_pid=$!

# Start bunzip2 in the background and obtain the process ID.
bunzip2 -q "${download_dir}/${download_file}" 2>/dev/null

# Stop the spinner after decompression is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Decompressing ${download_dir}/${download_file} completed."


###############################################################################
# Read the dump file, exclude unnecessary lines, extract titles,
# and save them to a temporary file.
###############################################################################
spinner_loop "Extracting titles from ${download_dir}/${file_name}" &
spinner_pid=$!

tmpfile=$(mktemp "/tmp/${file_name}.XXXXXX")

# Read one line at a time
while IFS= read -r line; do
    # Ignore empty lines
    if [[ -z "${line}" ]]; then
        continue
    fi

    # If the line contains ":Category:", ignore it.
    if [[ "${line}" == *":Category:"* ]]; then
        continue
    fi

    # If the line contains ":Template:", ignore it.
    if [[ "${line}" == *":Template:"* ]]; then
        continue
    fi

    # If the line contains ":Wikipedia:", ignore it.
    if [[ "${line}" == *":Wikipedia:"* ]]; then
        continue
    fi

    # If the line contains ":Portal:", ignore it.
    if [[ "${line}" == *":Portal:"* ]]; then
        continue
    fi

    # Split the lines with ':' and get the rightmost part as the title.
    title="${line##*:}"

    # Ignore empty titles
    if [[ -z "${title}" ]]; then
        continue
    fi

    # Ignore titles containing "Help"
    if [[ "${title}" == Help* ]]; then
        continue
    fi

    # Language-specific title filters
    case "$lang" in
        ja)
            # Ignore Japanese list/disambiguation/deletion pages
            if [[ "${title}" == *"一覧"* ]]; then continue; fi
            if [[ "${title}" == *"曖昧さ回避"* ]]; then continue; fi
            if [[ "${title}" == *"削除依頼"* ]]; then continue; fi
            if [[ "${title}" == *"削除記録"* ]]; then continue; fi
            ;;
        ko)
            # Ignore Korean list/disambiguation pages
            if [[ "${title}" == *"목록"* ]]; then continue; fi
            if [[ "${title}" == *"동음이의"* ]]; then continue; fi
            ;;
        zh)
            # Ignore Chinese list/disambiguation pages
            if [[ "${title}" == *"列表"* ]]; then continue; fi
            if [[ "${title}" == *"消歧义"* ]]; then continue; fi
            if [[ "${title}" == *"消歧義"* ]]; then continue; fi
            ;;
    esac

    # Write title to file one line at a time
    echo "${title}" >> "${tmpfile}"
done < <(grep -Ev ':[^:]*[a-zA-Z][^:]*:' "${download_dir}/${file_name}")

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Extracting titles from ${download_dir}/${file_name} completed."


###############################################################################
# Select N titles at random
###############################################################################
spinner_loop "Creating ${texts_file}" &
spinner_pid=$!

shuf -n "${title_count}" "${tmpfile}" | while read -r title; do
    # If the title is blank, ignore it.
    if [[ -z "${title}" ]]; then
        continue
    fi

    # URL encode title
    encoded_title=$(echo -n "${title}" | jq -sRr @uri)

    # Generate Wikipedia API URL
    url="https://${wiki_domain}/w/api.php?action=query&prop=extracts&format=json&explaintext=1&redirects=1&titles=${encoded_title}"

    # Retrieve data from API and extract text
    response=$(curl -s --retry 2 --retry-delay 1 "${url}")

    # Validate JSON response before parsing
    if ! echo "${response}" | jq empty 2>/dev/null; then
        sleep 1
        continue
    fi

    text=$(echo "${response}" | jq -r '.query.pages[] | .extract')

    # If the text is empty, ignore it.
    if [[ -z "${text}" ]]; then
        continue
    fi

    # If the text is "null," ignore it.
    if [[ "${text}" == "null" ]]; then
        continue
    fi

    # Rate limit: small delay between API requests
    sleep 0.5

    # Extract the longest line
    longest_line=$(echo "${text}" | awk 'length > max_length { max_length = length; longest = $0 } END { print longest }')

    # Split text into sentences using ICU4X SentenceSegmenter (Unicode UAX #29)
    readarray -t sentences < <(echo "${longest_line}" | ${litsea_cli} split-sentences)

    for sentence in "${sentences[@]}"; do
        ## Replace consecutive spaces with a single space
        sentence=$(echo "$sentence" | tr -s ' ')

        # Trim sentence
        sentence=$(echo "${sentence}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # If the sentence is empty, ignore it.
        if [[ -z "${sentence}" ]]; then
            continue
        fi

        # Skip ASCII-only lines
        if [[ "${sentence}" =~ ^[a-zA-Z0-9[:space:][:punct:]]+$ ]]; then
            continue
        fi

        # Append the sentence to the texts file
        echo "${sentence}" >> "${texts_file}"
    done
done

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Creating ${texts_file} completed."
