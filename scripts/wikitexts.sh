#!/bin/bash

lang=ja
timestamp=20250601
file_name="${lang}wiki-${timestamp}-pages-articles-multistream-index.txt"
download_dir=/tmp
download_file="${file_name}.bz2"
download_url="https://dumps.wikimedia.org/${lang}wiki/${timestamp}/${download_file}"

title_count=10

texts_file="texts.txt"

##################
# spinner の定義
##################
spinner=( '|' '/' '-' '\' )
spin_idx=0


#################
# spinner の停止
#################
cleanup() {
    if [[ -n "$spinner_pid" ]]; then
        kill "$spinner_pid" 2>/dev/null
    fi
    exit 1
}


################################################
# SIGINT, SIGTERM, EXIT 受信時に cleanup を呼ぶ
################################################
trap cleanup INT TERM EXIT


#######################################
# 非同期で spinner を更新する関数を定義
#######################################
spinner_loop() {
    local msg="$1"
    while true; do
        echo -ne "${msg} ... ${spinner[spin_idx]} \r"
        spin_idx=$(( (spin_idx + 1) % ${#spinner[@]} ))
        sleep 0.1
    done
}


#############################
# ダンプファイルのダウンロード
#############################
spinner_loop "Downloading ${download_url}" &
spinner_pid=$!

# curl をバックグラウンドで起動し、プロセスIDを取得
curl -s -o "${download_dir}/${download_file}" "${download_url}" 

# ループ完了後に spinner を停止
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Downloading ${download_url} completed."


######################
# ダンプファイルの解凍
######################
spinner_loop "Decompressing ${download_dir}/${download_file}" &
spinner_pid=$!

# bunzip2 をバックグラウンドで起動し、プロセスIDを取得
bunzip2 -q "${download_dir}/${download_file}" 2>/dev/null

# ループ完了後に spinner を停止
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Decompressing ${download_dir}/${download_file} completed."


#########################################################################
# ダンプファイルを読込、不要な行を除外し、タイトルを抽出して一時ファイルに保存
#########################################################################
spinner_loop "Extracting titles from ${download_dir}/${file_name}" &
spinner_pid=$!

tmpfile=$(mktemp /tmp/${file_name}.XXXXXX)

# 1行ずつ読み込む
while IFS= read -r line; do
    # 行が空は無視
    if [[ -z "${line}" ]]; then
        continue
    fi

    # 行に:Category:が含まれる場合は無視
    if [[ "${line}" == *":Category:"* ]]; then
        continue
    fi

    # 行に:Template:が含まれる場合は無視
    if [[ "${line}" == *":Template:"* ]]; then
        continue
    fi

    # 行に:Wikipedia:が含まれる場合は無視
    if [[ "${line}" == *":Wikipedia:"* ]]; then
        continue
    fi

    # 行に:Portal:が含まれる場合は無視
    if [[ "${line}" == *":Portal:"* ]]; then
        continue
    fi

    # 行を:で分割して一番右の部分をタイトルとして取得
    title="${line##*:}"

    # タイトルが空は無視
    if [[ -z "${title}" ]]; then
        continue
    fi

    # タイトルがHelpは無視
    if [[ "${title}" == Help* ]]; then
        continue
    fi

    # タイトルに一覧が含まれる場合は無視
    if [[ "${title}" == *"一覧"* ]]; then
        continue
    fi

    # タイトルに曖昧さ回避が含まれる場合は無視
    if [[ "${title}" == *"曖昧さ回避"* ]]; then
        continue
    fi

    # タイトルに削除依頼が含まれる場合は無視
    if [[ "${title}" == *"削除依頼"* ]]; then
        continue
    fi

    # タイトルに削除記録が含まれる場合は無視
    if [[ "${title}" == *"削除記録"* ]]; then
        continue
    fi

    # titleをファイルに1行ずつ書き込む
    echo "${title}" >> ${tmpfile}
done < <(grep -Ev ':[^:]*[a-zA-Z][^:]*:' ${download_dir}/${file_name})

# ループ完了後に spinner を停止
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Extracting titles from ${download_dir}/${file_name} completed."


#############################
# ランダムにタイトルをN個選ぶ
#############################

spinner_loop "Creating ${texts_file}" &
spinner_pid=$!

shuf -n ${title_count} ${tmpfile} | while read -r title; do
    # タイトルが空だったら無視
    if [[ -z "${title}" ]]; then
        continue
    fi

    # titleをURLエンコード
    encoded_title=$(echo -n "${title}" | jq -sRr @uri)
    # echo "Processing title: ${title} (encoded: ${encoded_title})"

    # WikipediaのURLを生成
    url="https://${lang}.wikipedia.org/wiki/${encoded_title}"

    # Wikipedia APIのURLを生成
    url="https://${lang}.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&explaintext=1&redirects=1&titles=${encoded_title}"

    # APIからデータを取得し、テキストを抽出
    text=$(curl -s "${url}" | jq -r '.query.pages[] | .extract')
    # echo "Extracted text: ${text}"

    # テキストが空だったら無視
    if [[ -z "${text}" ]]; then
        continue
    fi

    # テキストがnullだったら無視
    if [[ "${text}" == "null" ]]; then
        continue
    fi

    # 一番長い行を抽出
    longest_line=$(echo "${text}" | awk 'length > max_length { max_length = length; longest = $0 } END { print longest }')
    # echo "Longest line: ${longest_line}"

    # テキストを文章で分割
    readarray -t sentences < <(echo "${longest_line}" | sed -E 's/([.!?\！？。．]+)/\1\n/g')

    for sentence in "${sentences[@]}"; do
        # 文章が空だったら無視
        if [[ -z "${sentence}" ]]; then
            continue
        fi

        echo "${sentence}" >> "${texts_file}"
    done
done

# ループ完了後に spinner を停止
kill "${spinner_pid}" 2>/dev/null
wait "${spinner_pid}" 2>/dev/null
echo "Creating ${texts_file} completed."
