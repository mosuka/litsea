# Makefile.frag for the litsea PHP extension. config.m4 appends it to the
# phpize-generated Makefile after Makefile.global.
#
# Variables from that Makefile: top_srcdir / top_builddir (absolute), CARGO and
# PHP_CONFIG (substituted by config.m4), PHP_EXECUTABLE and SHLIB_SUFFIX_NAME
# ("so" on Linux, "dylib" on macOS) from phpize. $(top_srcdir) is used rather
# than $(srcdir) because PHP_ADD_MAKEFILE_FRAGMENT rewrites the literal
# $(srcdir) with sed while appending this file.

LITSEA_WORKSPACE   = $(top_srcdir)/..
# Extra flags for `cargo build`, e.g. `make LITSEA_CARGO_FLAGS=--locked`.
LITSEA_CARGO_FLAGS =

# Plain `make` builds the module. With no C sources $(PHP_MODULES) is empty and
# Makefile.global's `all` has no prerequisites; adding one merges into that
# rule without redefining its recipe, so make prints no "overriding recipe"
# warning.
all: modules/litsea.so

# `make install` -> install-modules -> modules/litsea.so. The module is a real
# file target with no prerequisites, so an existing file is always up to date
# and cargo is NOT re-run: `sudo make install` after `make` is a pure copy
# (root usually has no cargo or rustup). build-modules is left alone on
# purpose; hooking cargo there would re-trigger it from install-modules.
install-modules: modules/litsea.so

# Consequence: after editing the Rust sources, `make` does nothing until the
# file is removed (`make clean` deletes modules/*). PHP dlopens ".so" on macOS
# as well, so the .dylib that cargo produces is copied under the .so name.
modules/litsea.so:
	@set -e; \
	suffix="$(SHLIB_SUFFIX_NAME)"; \
	if test -z "$$suffix"; then \
	  case "$$(uname -s)" in Darwin) suffix=dylib;; *) suffix=so;; esac; \
	fi; \
	cd "$(LITSEA_WORKSPACE)"; \
	echo "==> cargo build --release -p litsea-php (PHP_CONFIG=$(PHP_CONFIG), PHP=$(PHP_EXECUTABLE))"; \
	PHP_CONFIG="$(PHP_CONFIG)" PHP="$(PHP_EXECUTABLE)" \
	  "$(CARGO)" build --release -p litsea-php $(LITSEA_CARGO_FLAGS) || { \
	  echo "error: cargo build failed." >&2; \
	  echo "hint: ext-php-rs needs a Rust toolchain and libclang (bindgen)." >&2; \
	  echo "hint: for 'sudo make install', run 'make' as your normal user first." >&2; \
	  exit 1; }; \
	lib="$${CARGO_TARGET_DIR:-target}/release/liblitsea_php.$$suffix"; \
	if test ! -f "$$lib"; then \
	  echo "error: cargo finished but $$lib does not exist (cwd: $$(pwd))." >&2; \
	  echo "hint: a target dir set in .cargo/config.toml is not detected; export CARGO_TARGET_DIR instead." >&2; \
	  exit 1; \
	fi; \
	mkdir -p "$(top_builddir)/modules"; \
	cp -f "$$lib" "$(top_builddir)/$@"; \
	echo "==> wrote $(top_builddir)/$@"
