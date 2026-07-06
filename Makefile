# ── Variables ─────────────────────────────────────────────────────────────────
LEAPFLOW_DATA_DIR ?= $(HOME)/.leapflow

.PHONY: setup sync test brain lint cua-check

setup:  ## Setup scripts permissions and environment
	chmod +x scripts/setup.sh scripts/run.sh
	./scripts/setup.sh

sync:  ## Sync all dependencies
	uv sync --all-extras

test:  ## Run tests
	uv run pytest tests/ -q

lint:  ## Lint source code
	uv run ruff check src/leapflow/ tests/

# LeapFlow CLI (pass PROMPT via ARGS, e.g. make brain ARGS='--prompt "hello"')
brain:  ## Start Brain process
	uv run leap $(ARGS)

cua-check:  ## Check cua-driver installation status
	@which cua-driver > /dev/null 2>&1 && echo "✓ cua-driver: $$(cua-driver --version 2>/dev/null || echo 'installed')" || echo "✗ cua-driver not found. Install: brew install trycua/tap/cua-driver"
