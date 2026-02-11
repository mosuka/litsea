LITSEA_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea") | .version')
LITSEA_CLI_VERSION ?= $(shell cargo metadata --no-deps --format-version=1 | jq -r '.packages[] | select(.name=="litsea-cli") | .version')

USER_AGENT ?= $(shell curl --version | head -n1 | awk '{print $1"/"$2}')
USER ?= $(shell whoami)
HOSTNAME ?= $(shell hostname)

.DEFAULT_GOAL := help

help: ## Show help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

clean: ## Clean the project
	cargo clean

format: ## Format the project
	cargo fmt

lint: ## Lint the project
	cargo clippy

test: ## Test the project
	cargo test

build: ## Build the project
	cargo build --release

bench: ## Benchmark the project
	cargo bench --bench bench

tag: ## Make a new tag for the current version
	git tag v$(LITSEA_VERSION)
	git push origin v$(LITSEA_VERSION)

publish: ## Publish the crate to crates.io
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/litsea | jq -r '.versions[].num' | grep $(LITSEA_VERSION)),)
	(cd litsea && cargo package && cargo publish)
	sleep 10
endif
ifeq ($(shell curl -s -XGET -H "User-Agent: $(USER_AGENT) ($(USER)@$(HOSTNAME))" https://crates.io/api/v1/crates/litsea-cli | jq -r '.versions[].num' | grep $(LITSEA_CLI_VERSION)),)
	(cd litsea-cli && cargo package && cargo publish)
endif
