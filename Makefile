LITSEA_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea") | .version')
LITSEA_BINDING_CORE_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea-binding-core") | .version')
LITSEA_CLI_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea-cli") | .version')

# Python tooling for the litsea-python binding, kept in a venv inside the
# crate so it never touches the user's interpreter.
PYTHON_VENV_DIR := litsea-python/.venv
PYTHON          := $(PYTHON_VENV_DIR)/bin/python
PIP             := $(PYTHON_VENV_DIR)/bin/pip
MATURIN         := $(PYTHON_VENV_DIR)/bin/maturin
PYTEST          := $(PYTHON_VENV_DIR)/bin/pytest

# `litsea-ruby` links against libruby, and rb-sys reads the interpreter's
# configuration from RBCONFIG_* environment variables.
CARGO_WITH_RBCONFIG = ruby -rrbconfig -e 'RbConfig::CONFIG.each { |k, v| ENV["RBCONFIG_\#{k.upcase}"] = v }; exec(*ARGV)' --

USER_AGENT ?= $(shell curl --version | head -n1 | awk '{print $1"/"$2}')
USER ?= $(shell whoami)
HOSTNAME ?= $(shell hostname)

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'

clean: clean-litsea-python clean-litsea-nodejs clean-litsea-php clean-litsea-ruby clean-litsea-wasm ## Clean the project
	cargo clean

clean-litsea-python: ## Clean litsea-python build artifacts
	rm -rf $(PYTHON_VENV_DIR)
	rm -rf litsea-python/dist
	rm -rf litsea-python/.pytest_cache
	rm -rf litsea-python/python/litsea/__pycache__
	rm -rf litsea-python/tests/__pycache__

format: ## Format the project
	cargo fmt

lint: ## Lint the project
	cargo clippy --workspace --all-targets -- -D warnings

test: ## Test the project (Rust workspace only; see test-litsea-python)
	cargo test --workspace

$(PYTHON_VENV_DIR):
	python3 -m venv $(PYTHON_VENV_DIR)
	$(PIP) install --quiet --upgrade pip

setup-venv: $(PYTHON_VENV_DIR) ## Create the litsea-python venv and install dev tools
	$(PIP) install --quiet maturin pytest mypy ruff

test-litsea-python: setup-venv ## Test litsea-python (Rust unit tests + pytest)
	cargo test -p litsea-python --lib
	cd litsea-python && VIRTUAL_ENV=$(abspath $(PYTHON_VENV_DIR)) $(abspath $(MATURIN)) develop --quiet && $(abspath $(PYTEST)) tests/ -v

clean-litsea-nodejs: ## Clean litsea-nodejs build artifacts
	rm -rf litsea-nodejs/node_modules
	rm -rf litsea-nodejs/npm
	rm -f litsea-nodejs/*.node
	rm -f litsea-nodejs/package-lock.json

test-litsea-nodejs: ## Test litsea-nodejs (Rust unit tests + node --test)
	cargo test -p litsea-nodejs --lib
	cd litsea-nodejs && npm install --silent --no-audit --no-fund && npx napi build --platform -p litsea-nodejs && npm test

lint-litsea-nodejs: ## Lint litsea-nodejs (clippy)
	cargo clippy -p litsea-nodejs --all-targets -- -D warnings

clean-litsea-php: ## Clean litsea-php build artifacts
	rm -rf litsea-php/vendor
	rm -f litsea-php/composer.lock

test-litsea-php: ## Test litsea-php (Rust unit tests + PHPUnit)
	cargo test -p litsea-php --lib
	cargo build -p litsea-php
	cd litsea-php && composer install --quiet --no-interaction && \
		LIB=$$(find ../target/debug -maxdepth 1 \( -name 'liblitsea_php.so' -o -name 'liblitsea_php.dylib' \) | head -1) && \
		php -d extension=$$LIB vendor/bin/phpunit

lint-litsea-php: ## Lint litsea-php (clippy)
	cargo clippy -p litsea-php --all-targets -- -D warnings

clean-litsea-ruby: ## Clean litsea-ruby build artifacts
	rm -rf litsea-ruby/tmp
	rm -rf litsea-ruby/pkg
	rm -f litsea-ruby/lib/litsea/litsea_ruby.so
	rm -f litsea-ruby/Gemfile.lock

# Fails early with an actionable message when the active Ruby has no
# bundler (rbenv's `system` Ruby often does not).
check-bundler:
	@bundle --version >/dev/null 2>&1 || { \
		echo "bundler is not usable with $$(ruby --version)."; \
		echo "A version manager's shim can exist while the selected Ruby has no bundler."; \
		echo "Select a Ruby that has it, e.g. 'rbenv local 3.4.9', or run 'gem install bundler'."; \
		exit 1; \
	}

test-litsea-ruby: check-bundler ## Test litsea-ruby (Rust unit tests + minitest)
	$(CARGO_WITH_RBCONFIG) cargo test -p litsea-ruby --lib
	cd litsea-ruby && bundle install --quiet && bundle exec rake compile && bundle exec rake test

lint-litsea-ruby: check-bundler ## Lint litsea-ruby (clippy + rubocop)
	$(CARGO_WITH_RBCONFIG) cargo clippy -p litsea-ruby --all-targets -- -D warnings
	cd litsea-ruby && bundle exec rubocop

clean-litsea-wasm: ## Clean litsea-wasm build artifacts
	rm -rf litsea-wasm/pkg
	rm -rf litsea-wasm/pkg-node
	rm -f litsea-wasm/tests/fixtures.tsv

# Which browser drives the headless tests. Firefox by default; override with
# `make test-litsea-wasm WASM_BROWSER=chrome`.
WASM_BROWSER ?= firefox

test-litsea-wasm: ## Test litsea-wasm (Rust unit tests + headless browser tests)
	cargo test -p litsea-wasm --lib
	./litsea-wasm/tests/generate_fixtures.sh
	cd litsea-wasm && wasm-pack test --headless --$(WASM_BROWSER)

lint-litsea-wasm: ## Lint litsea-wasm (clippy on wasm32)
	cargo clippy -p litsea-wasm --target wasm32-unknown-unknown -- -D warnings

lint-litsea-python: setup-venv ## Lint litsea-python (clippy + ruff + mypy)
	cargo clippy -p litsea-python --all-targets -- -D warnings
	cd litsea-python && $(abspath $(PYTHON_VENV_DIR))/bin/ruff check python/ tests/ examples/
	cd litsea-python && $(abspath $(PYTHON_VENV_DIR))/bin/ruff format --check python/ tests/ examples/

build: ## Build the project
	cargo build --release

build-litsea-python: setup-venv ## Build a release wheel for litsea-python
	cd litsea-python && VIRTUAL_ENV=$(abspath $(PYTHON_VENV_DIR)) $(abspath $(MATURIN)) build --release --out dist

build-litsea-ruby: check-bundler ## Build litsea-ruby (release)
	cd litsea-ruby && bundle install --quiet && bundle exec rake compile -- --release

build-litsea-wasm: ## Build litsea-wasm (wasm-pack, --target web)
	cd litsea-wasm && wasm-pack build --release --target web --out-dir pkg
	cp litsea-wasm/js/cache.js litsea-wasm/js/cache.d.ts litsea-wasm/pkg/

build-litsea-php: ## Build litsea-php (release)
	cargo build -p litsea-php --release

build-litsea-nodejs: ## Build litsea-nodejs (release)
	cd litsea-nodejs && npm install --silent --no-audit --no-fund && npx napi build --platform --release -p litsea-nodejs

check-wasm: ## Check the wasm32 build (litsea and the binding core)
	cargo check -p litsea --target wasm32-unknown-unknown --no-default-features
	cargo check -p litsea-binding-core --target wasm32-unknown-unknown --no-default-features

bench: ## Benchmark the project
	cargo bench --bench bench

tag: ## Make a new tag for the current version
	git tag v$(LITSEA_VERSION)
	git push origin v$(LITSEA_VERSION)

publish: ## Publish the crate to crates.io
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/litsea | jq -r 'select(.versions != null) | .versions[].num' 2>/dev/null | grep -Fx "$(LITSEA_VERSION)"),)
	(cd litsea && cargo package && cargo publish)
	sleep 10
endif
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/litsea-binding-core | jq -r 'select(.versions != null) | .versions[].num' 2>/dev/null | grep -Fx "$(LITSEA_BINDING_CORE_VERSION)"),)
	(cd litsea-binding-core && cargo package && cargo publish)
	sleep 10
endif
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/litsea-cli | jq -r 'select(.versions != null) | .versions[].num' 2>/dev/null | grep -Fx "$(LITSEA_CLI_VERSION)"),)
	(cd litsea-cli && cargo package && cargo publish)
endif
