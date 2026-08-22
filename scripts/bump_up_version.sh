#!/usr/bin/env bash
#
# Bumps the workspace version everywhere it is stated, in preparation for a
# release.
#
# Usage:
#   scripts/bump_up_version.sh 0.14.0
#   scripts/bump_up_version.sh 0.14.0 --keep-changelog
#
# What it touches:
#
#   Cargo.toml                          workspace.package.version, plus the two
#                                       [workspace.dependencies] entries that
#                                       duplicate it
#   Cargo.lock                          via `cargo update --workspace`
#   litsea-nodejs/package.json          npm restates the version
#   litsea-ruby/lib/litsea/version.rb   Litsea::VERSION, read by the gemspec
#   docs/{src,ja/src}/...               dependency snippets, 6 files per language
#   CHANGELOG.md                        `## Unreleased` -> `## <version> (<date>)`
#
# Three bindings need no edit and are deliberately absent from the list:
# litsea-python takes the version from Cargo through maturin's
# `dynamic = ["version"]`, litsea-wasm from Cargo.toml, and litsea-php has no
# version field at all - Packagist reads the git tag.
#
# The script finishes by grepping the tree for the old version. Anything it
# reports is a place this script does not know about yet: add it to FILES
# rather than editing it by hand, so the next bump stays a one-liner.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Every file that states the version literally.
readonly FILES=(
  "Cargo.toml"
  "litsea-nodejs/package.json"
  "litsea-ruby/lib/litsea/version.rb"
  "docs/src/litsea.md"
  "docs/src/README.md"
  "docs/src/bindings/binding-core.md"
  "docs/src/getting-started/installation.md"
  "docs/src/advanced/remote-model-loading.md"
  "docs/ja/src/litsea.md"
  "docs/ja/src/README.md"
  "docs/ja/src/bindings/binding-core.md"
  "docs/ja/src/getting-started/installation.md"
  "docs/ja/src/advanced/remote-model-loading.md"
)

usage() {
  cat >&2 <<'EOF'
usage: scripts/bump_up_version.sh <new-version> [--keep-changelog]

  <new-version>      the version to move to, e.g. 0.14.0
  --keep-changelog   leave the `## Unreleased` heading alone

Run from anywhere; paths are resolved relative to the repository root.
EOF
  exit 2
}

new_version=""
keep_changelog=false

while [ $# -gt 0 ]; do
  case "$1" in
    --keep-changelog) keep_changelog=true ;;
    -h | --help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *)
      [ -n "${new_version}" ] && { echo "unexpected argument: $1" >&2; usage; }
      new_version="$1"
      ;;
  esac
  shift
done

[ -n "${new_version}" ] || usage

# Semver, optionally with a pre-release suffix (0.14.0-rc.1).
if ! printf '%s' "${new_version}" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'; then
  echo "not a version: ${new_version}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

# The workspace version is the source of truth for what we are moving from.
current_version="$(
  grep -m1 -E '^version = "' Cargo.toml | sed -E 's/^version = "(.*)"/\1/'
)"

if [ -z "${current_version}" ]; then
  echo "could not read the current version from Cargo.toml" >&2
  exit 1
fi

if [ "${current_version}" = "${new_version}" ]; then
  echo "already at ${new_version}; nothing to do" >&2
  exit 1
fi

echo "bumping ${current_version} -> ${new_version}"

for file in "${FILES[@]}"; do
  if [ ! -f "${file}" ]; then
    echo "  missing: ${file} (has it moved? update FILES in this script)" >&2
    exit 1
  fi

  count="$(grep -c -F "${current_version}" "${file}" || true)"
  if [ "${count}" -eq 0 ]; then
    echo "  ${file}: no occurrence (update FILES in this script if it no longer states the version)" >&2
    continue
  fi

  # -F is not available to sed, so the version is escaped for the pattern;
  # only the dots need it.
  escaped_current="$(printf '%s' "${current_version}" | sed 's/\./\\./g')"
  sed -i "s/${escaped_current}/${new_version}/g" "${file}"
  echo "  ${file}: ${count}"
done

# Refresh the lockfile's workspace members. `--workspace` leaves third-party
# dependencies where they are, so a version bump does not smuggle in updates.
echo "updating Cargo.lock"
cargo update --workspace --quiet

if [ "${keep_changelog}" = false ]; then
  if grep -q '^## Unreleased$' CHANGELOG.md; then
    today="$(date +%Y-%m-%d)"
    sed -i "0,/^## Unreleased$/s//## ${new_version} (${today})/" CHANGELOG.md
    echo "CHANGELOG.md: ## Unreleased -> ## ${new_version} (${today})"
  else
    echo "CHANGELOG.md: no '## Unreleased' heading; left alone" >&2
  fi
fi

# napi writes the package version into the loader it generates, so the
# committed entry point goes stale on every bump - and CI's freshness check
# fails on the difference. Regenerate it here when the toolchain is available;
# the leftover check below is what catches it if this does not happen.
if grep -q -F "${current_version}" litsea-nodejs/index.js 2>/dev/null; then
  if command -v npx >/dev/null 2>&1; then
    echo "regenerating litsea-nodejs/index.js (napi embeds the version)"
    (
      cd litsea-nodejs
      npm install --silent --no-audit --no-fund
      npx napi build --platform -p litsea-nodejs
    ) >/dev/null 2>&1 || echo "  regeneration failed; see the command below" >&2
  else
    echo "npx not found; cannot regenerate litsea-nodejs/index.js" >&2
  fi
fi

# Anything left is a place this script does not know about.
#
# Only tracked files are searched: build output states versions of its own
# (a Python venv's dependency metadata, wasm-pack's generated package.json,
# vendored PHP packages) and none of it is ours to edit. `git ls-files` gets
# that right by construction, where an ignore list would drift.
#
# CHANGELOG.md and Cargo.lock are excluded too: the former names older
# versions in its history on purpose, and the latter was just regenerated
# but still lists third-party crates that happen to share the number.
echo
echo "checking for leftovers"
leftovers="$(
  git ls-files -z |
    grep -zZv -E '^(CHANGELOG\.md|Cargo\.lock)$' |
    xargs -0 grep -n -F "${current_version}" /dev/null 2>/dev/null || true
)"

if [ -n "${leftovers}" ]; then
  echo "${leftovers}" >&2
  cat >&2 <<EOF

The lines above still name ${current_version}.

If litsea-nodejs/index.js is among them, it is generated - regenerate and
commit it, or CI's freshness check will fail:

    cd litsea-nodejs && npm install && npx napi build --platform -p litsea-nodejs

Anything else is a place this script does not know about: add its file to
FILES here, so the next bump covers it.
EOF
  exit 1
fi

echo "  none"
echo
echo "done. Next:"
echo "  1. review the diff"
echo "  2. cargo test --all-features && cargo clippy --workspace --all-targets -- -D warnings"
echo "  3. commit, and tag once merged"
