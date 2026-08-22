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

[ -x "${cli}" ] || cargo build --quiet -p litsea-cli --manifest-path "${repo_root}/Cargo.toml"

# language <TAB> model <TAB> mode <TAB> sentence, where mode is "seg" or
# "pos". An explicit token rather than an empty field: tab counts as
# whitespace for IFS, so `read` collapses consecutive tabs and an empty
# field would shift every column after it.
cases=(
  $'japanese\tjapanese.model\tseg\tこれはテストです。'
  $'chinese\tchinese.model\tseg\t我喜欢吃中国菜。'
  $'korean\tkorean.model\tseg\t안녕하세요 반갑습니다'
  $'english\tenglish.model\tseg\tThe quick brown fox jumps over the lazy dog.'
  $'japanese\tjapanese_pos.model\tpos\tこれはテストです。'
  $'korean\tkorean_pos.model\tpos\t안녕하세요 반갑습니다'
)

: > "${out}"
for row in "${cases[@]}"; do
  IFS=$'\t' read -r language model mode sentence <<< "${row}"
  args=(segment -l "${language}")
  [ "${mode}" = "pos" ] && args+=(--pos)
  args+=("${repo_root}/models/${model}")
  expected="$(printf '%s\n' "${sentence}" | "${cli}" "${args[@]}")"
  printf '%s\t%s\t%s\t%s\t%s\n' "${language}" "${model}" "${mode}" "${sentence}" "${expected}" >> "${out}"
done

echo "wrote ${out} ($(wc -l < "${out}") cases)"
