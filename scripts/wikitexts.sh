#!/bin/bash

# Default value (uses the value defined in the environment variables, if defined)
timestamp="${WIKI_TIMESTAMP:-20250601}"
title_count="${WIKI_TITLE_COUNT:-1000}"
texts_file="${WIKI_TEXTS_FILE:-texts.txt}"

###############################################################################
# usage function
# Displays the usage information for the script.
# Usage: usage
# This function is called when the script is run with the -h option or when an invalid option is provided.
# It prints the usage information and exits the script with a status code of 1.
###############################################################################
usage() {
    echo "Usage: $0 [-h] [-l language] [-t timestamp] [-c title_count] [-o texts_file]"
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

echo "Timestamp: ${timestamp}"
echo "Title count: ${title_count}"
echo "Texts file: ${texts_file}"


file_name="jawiki-${timestamp}-pages-articles-multistream-index.txt"
download_dir=/tmp
download_file="${file_name}.bz2"
download_url="https://dumps.wikimedia.org/jawiki/${timestamp}/${download_file}"

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

tmpfile=$(mktemp /tmp/${file_name}.XXXXXX)

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

    # Ignore titles containing "一覧"
    if [[ "${title}" == *"一覧"* ]]; then
        continue
    fi

    # Ignore titles containing "曖昧さ回避"
    if [[ "${title}" == *"曖昧さ回避"* ]]; then
        continue
    fi

    # Ignore titles containing "削除依頼"
    if [[ "${title}" == *"削除依頼"* ]]; then
        continue
    fi

    # Ignore titles containing "削除記録"
    if [[ "${title}" == *"削除記録"* ]]; then
        continue
    fi

    # Write title to file one line at a time
    echo "${title}" >> ${tmpfile}
done < <(grep -Ev ':[^:]*[a-zA-Z][^:]*:' ${download_dir}/${file_name})

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Extracting titles from ${download_dir}/${file_name} completed."


###############################################################################
# Select N titles at random
###############################################################################
spinner_loop "Creating ${texts_file}" &
spinner_pid=$!

shuf -n ${title_count} ${tmpfile} | while read -r title; do
    # If the title is blank, ignore it.
    if [[ -z "${title}" ]]; then
        continue
    fi

    # URL encode title
    encoded_title=$(echo -n "${title}" | jq -sRr @uri)
    # echo "Processing title: ${title} (encoded: ${encoded_title})"

    # Generate Wikipedia URL
    url="https://ja.wikipedia.org/wiki/${encoded_title}"

    # Generate Wikipedia API URL
    url="https://ja.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&explaintext=1&redirects=1&titles=${encoded_title}"

    # Retrieve data from API and extract text
    text=$(curl -s "${url}" | jq -r '.query.pages[] | .extract')
    # echo "Extracted text: ${text}"

    # If the text is empty, ignore it.
    if [[ -z "${text}" ]]; then
        continue
    fi

    # If the text is “null,” ignore it.
    if [[ "${text}" == "null" ]]; then
        continue
    fi

    # Extract the longest line
    longest_line=$(echo "${text}" | awk 'length > max_length { max_length = length; longest = $0 } END { print longest }')
    # echo "Longest line: ${longest_line}"

    # Split text into sentences
    readarray -t sentences < <(echo "${longest_line}" | sed -E 's/([!?\！？。]+)/\1\n/g')

    for sentence in "${sentences[@]}"; do
        # 文章が空だったら無視
        if [[ -z "${sentence}" ]]; then
            continue
        fi

        echo "${sentence}" >> "${texts_file}"
    done
done

# Stop the spinner after the loop is complete.
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Creating ${texts_file} completed."
