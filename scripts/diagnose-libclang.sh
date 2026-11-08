#!/usr/bin/env bash
# Temporary diagnosis for #299, run by .github/workflows/diagnose-libclang.yml
# on the ubuntu-latest runner. Removed before the fix is reviewed. Adapted from
# the diagnosis of lindera/lindera#1326.
#
# Lists the clang, llvm-config and libclang instances on the runner, then
# builds the given crate once per environment, each in its own target
# directory so that every build script runs again, and records which libclang
# the build script of the crate opens and whether the bindings are generated.
#
# Usage: scripts/diagnose-libclang.sh <crate>
#   <crate> is a dependency that generates bindings with bindgen in its build
#   script: ext-php-rs (litsea-php) or rb-sys (litsea-ruby).
set -uo pipefail

crate=${1:?usage: scripts/diagnose-libclang.sh <crate>}
results=()

section() {
    printf '\n=== %s ===\n' "$1"
}

# Builds the crate with the given environment assignments and reports the
# libclang it used and the result as a "CASE_RESULT <crate> <name> PASS|FAIL"
# line.
#
# Arguments: a case name, then NAME=value assignments for `env`.
run_case() {
    local name=$1
    shift
    local target="/tmp/diag-target-${crate}-${name}"
    local log="/tmp/diag-${crate}-${name}.log"
    section "Case ${name}: ${*:-(default environment)}"
    env "$@" CARGO_TARGET_DIR="${target}" cargo build --locked -p "${crate}" >"${log}" 2>&1
    local status=$?
    echo "exit status: ${status}"
    echo "clang-sys link directives:"
    cat "${target}"/debug/build/clang-sys-*/output 2>/dev/null | grep -E 'rustc-link-(search|lib)' || echo "  (none)"
    echo "libclang linked into the build script of ${crate}:"
    for bin in "${target}"/debug/build/"${crate}"-*/build-script-build; do
        [ -e "${bin}" ] && ldd "${bin}" | grep -i clang
    done
    echo "errors:"
    grep -E 'error:|error\[|Unable to generate' "${log}" | head -5 || echo "  (none)"
    local result=FAIL
    [ "${status}" -eq 0 ] && result=PASS
    echo "CASE_RESULT ${crate} ${name} ${result}"
    results+=("CASE_RESULT ${crate} ${name} ${result}")
    rm -rf "${target}"
}

section "System"
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
uname -m

section "clang and llvm-config on PATH"
for tool in clang llvm-config; do
    path=$(command -v "${tool}" || true)
    if [ -n "${path}" ]; then
        echo "${tool}: ${path} -> $(readlink -f "${path}")"
    else
        echo "${tool}: not found"
    fi
done
clang --version | head -1 || true
llvm-config --version --prefix --bindir --libdir || true

section "Installed LLVM"
for dir in /usr/lib/llvm-*; do
    echo "${dir}:"
    find "${dir}/bin" -maxdepth 1 -name clang -printf '  %p -> %l\n'
    stat -c '  %N (%F)' "${dir}/lib"
    for library in "${dir}"/lib/libclang*; do
        [ -e "${library}" ] && stat -c '  %N' "${library}"
    done
done
echo "libclang in the library directories clang-sys searches:"
find /usr/lib/x86_64-linux-gnu /usr/lib -maxdepth 1 -name 'libclang*' -printf '  %p -> %l\n'

# The headers the build script hands to bindgen, and whether they pull in the
# x86 SIMD headers of clang, where the ext-php-rs build failed.
case "${crate}" in
ext-php-rs)
    section "PHP"
    php --version | head -1
    header_dirs=("$(php-config --include-dir)")
    ;;
rb-sys)
    section "Ruby"
    ruby --version
    header_dirs=(
        "$(ruby -e 'print RbConfig::CONFIG["rubyhdrdir"]')"
        "$(ruby -e 'print RbConfig::CONFIG["rubyarchhdrdir"]')"
    )
    ;;
*)
    header_dirs=()
    ;;
esac
for dir in "${header_dirs[@]}"; do
    echo "headers under ${dir} that include the x86 SIMD headers:"
    grep -rlE '#\s*include\s*<(x|e|im)mintrin\.h>' "${dir}" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
done

# clang-sys's Clang::find, which bindgen calls, takes the clang in
# `llvm-config --bindir` before the one on PATH (CLANG_PATH unset).
bindgen_clang=""
if bindir=$(llvm-config --bindir 2>/dev/null) && [ -x "${bindir}/clang" ]; then
    bindgen_clang=$(readlink -f "${bindir}/clang")
else
    bindgen_clang=$(readlink -f "$(command -v clang)")
fi
bindgen_llvm=$(dirname "$(dirname "${bindgen_clang}")")
echo
echo "clang that bindgen asks for the include paths: ${bindgen_clang}"
echo "its LLVM: ${bindgen_llvm}"

run_case default
# The default case again, with the dynamic loader logging the files each
# process opens, to see which libclang the build script dlopens.
run_case default-traced LD_DEBUG=files LD_DEBUG_OUTPUT="/tmp/diag-lddebug-${crate}"
echo "libclang opened in the default-traced case:"
grep -h 'file=.*libclang' /tmp/diag-lddebug-"${crate}".* 2>/dev/null | sed -E 's/^ *[0-9]+: */  /' | sort -u || echo "  (none)"
run_case libclang-of-bindgen-clang LIBCLANG_PATH="${bindgen_llvm}/lib"
for dir in /usr/lib/llvm-*; do
    version=${dir##*/llvm-}
    run_case "pair-${version}" LIBCLANG_PATH="${dir}/lib" CLANG_PATH="${dir}/bin/clang"
done
newest=$(find /usr/lib -maxdepth 1 -name 'llvm-*' | sort -V | tail -1)
if [ "${newest}" != "${bindgen_llvm}" ]; then
    run_case "mismatch-${newest##*/llvm-}-lib-with-bindgen-clang" \
        LIBCLANG_PATH="${newest}/lib" CLANG_PATH="${bindgen_clang}"
fi

section "Summary"
printf '%s\n' "${results[@]}"
echo "DIAGNOSIS_DONE ${crate}"
