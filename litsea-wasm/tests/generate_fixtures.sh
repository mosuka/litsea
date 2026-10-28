#!/usr/bin/env bash
# Generates the expected segmentation output for the wasm tests.
#
# The wasm tests run in a browser and cannot spawn a process, so the CLI --
# the reference implementation the other four bindings compare against -- is
# run here instead, and the test asserts equality against what it wrote.
#
# The result is committed, because `include_str!` needs it to exist for any
# build of the test target (including `cargo clippy --all-targets` on a fresh
# checkout). CI regenerates it and fails on a diff, so a retrained model
# cannot leave a stale expectation behind -- the same guard the Node.js
# binding uses for its generated entry points.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(dirname "${BASH_SOURCE[0]}")/fixtures.tsv"
cli="${repo_root}/target/debug/litsea"

# Always build, even when a binary exists: one left in target/ (a CI cache
# restored by its Cargo.lock key, or an older checkout) can predate the
# source under test, and Cargo makes the build a no-op when it is current.
cargo build --quiet -p litsea-cli --manifest-path "${repo_root}/Cargo.toml"

# The user dictionary is written to a file for `--user-words`.
words_file="$(mktemp)"
trap 'rm -f "${words_file}"' EXIT

# language <TAB> model <TAB> mode <TAB> words <TAB> sentence, where mode is
# "seg" or "pos" and words is a comma-separated user dictionary (#263), with
# `word=TAG` for a word that carries a UPOS tag (#264; written to the CLI's
# file as `word<TAB>TAG`), or "-" for none. Explicit tokens rather than
# empty fields: tab counts as whitespace for IFS, so `read` collapses
# consecutive tabs and an empty field would shift every column after it.
cases=(
  $'japanese\tjapanese.model\tseg\t-\tこれはテストです。'
  $'chinese\tchinese.model\tseg\t-\t我喜欢吃中国菜。'
  $'korean\tkorean.model\tseg\t-\t안녕하세요 반갑습니다'
  $'english\tenglish.model\tseg\t-\tThe quick brown fox jumps over the lazy dog.'
  $'japanese\tjapanese_pos.model\tpos\t-\tこれはテストです。'
  $'korean\tkorean_pos.model\tpos\t-\t안녕하세요 반갑습니다'
  $'japanese\tjapanese.model\tseg\t雪だるま,新幹線\t雪だるまを作る'
  $'japanese\tjapanese.model\tseg\t雪だるま,新幹線\t新幹線で帰る'
  $'english\tenglish.model\tseg\tNew York\tI live in New York.'
  $'japanese\tjapanese_pos.model\tpos\t雪だるま\t雪だるまを作る'
  $'japanese\tjapanese_pos.model\tpos\t雪だるま,チャットGPT=PROPN,ポケモン=PROPN\tチャットGPTに聞く'
  $'japanese\tjapanese_pos.model\tpos\t雪だるま,チャットGPT=PROPN,ポケモン=PROPN\tポケモンを集める'
  $'japanese\tjapanese_pos.model\tpos\t雪だるま,楓=PROPN\t楓が笑った。'
)

: > "${out}"
for row in "${cases[@]}"; do
  IFS=$'\t' read -r language model mode words sentence <<< "${row}"
  args=(segment -l "${language}")
  [ "${mode}" = "pos" ] && args+=(--pos)
  if [ "${words}" != "-" ]; then
    entries="${words//,/$'\n'}"
    printf '%s\n' "${entries//=/$'\t'}" > "${words_file}"
    args+=(--user-words "${words_file}")
  fi
  args+=("${repo_root}/models/${model}")
  expected="$(printf '%s\n' "${sentence}" | "${cli}" "${args[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${language}" "${model}" "${mode}" "${words}" "${sentence}" "${expected}" >> "${out}"
done

echo "wrote ${out} ($(wc -l < "${out}") cases)"
