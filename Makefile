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

USER_AGENT ?= $(shell curl --version | head -n1 | awk '{print $1"/"$2}')
USER ?= $(shell whoami)
HOSTNAME ?= $(shell hostname)

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'

clean: clean-litsea-python clean-litsea-nodejs clean-litsea-php ## Clean the project
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

lint-litsea-python: setup-venv ## Lint litsea-python (clippy + ruff + mypy)
	cargo clippy -p litsea-python --all-targets -- -D warnings
	cd litsea-python && $(abspath $(PYTHON_VENV_DIR))/bin/ruff check python/ tests/ examples/
	cd litsea-python && $(abspath $(PYTHON_VENV_DIR))/bin/ruff format --check python/ tests/ examples/

build: ## Build the project
	cargo build --release

build-litsea-python: setup-venv ## Build a release wheel for litsea-python
	cd litsea-python && VIRTUAL_ENV=$(abspath $(PYTHON_VENV_DIR)) $(abspath $(MATURIN)) build --release --out dist

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
