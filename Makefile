# ── Variables ─────────────────────────────────────────────────────────────────
LEAPFLOW_DATA_DIR ?= $(HOME)/.leapflow

# Change-scoped selection compares against this ref (see tools/impact.py).
BASE ?= origin/main
# Parallelism for the mock layer. The real layer runs at -n 4 (few, heavy cases).
JOBS ?= auto

.PHONY: setup sync space-sync test test-unit test-e2e test-live test-impact test-full \
        record-traffic seed-cassettes sync-fixtures lint brain cua-check help

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Setup scripts permissions and environment
	chmod +x scripts/setup.sh scripts/run.sh
	./scripts/setup.sh

sync:  ## Sync all dependencies
	uv sync --all-extras

space-sync:  ## Sync dependencies including LeapSpace (requires Python >= 3.12)
	uv sync --all-extras --group leapspace

lint:  ## Lint source code
	uv run ruff check src/ tests/ tools/

# ── Test layers ───────────────────────────────────────────────────────────────
# The mock layer is broad and fast; the real layer is small, coarse, and never
# skipped. Both run offline: the LLM boundary is served from committed cassettes.

test: test-unit test-e2e  ## Default gate: mock layer + real journeys (offline)

test-unit:  ## Mock layer — hermetic units and components
	uv run pytest tests/ -q -m "not e2e" -n $(JOBS)

test-e2e:  ## Real layer — coarse journeys against a real leapd, cassette replay
	uv run pytest tests/journeys tests/regression -q -m "e2e or invariant" -n 4

test-full:  ## Everything, unselected
	uv run pytest tests/ -q -n $(JOBS)

test-impact:  ## Mock layer scoped to what changed since $(BASE), plus always-on tiers
	@uv run python tools/impact.py --base $(BASE) --run

test-live:  ## Real layer against a real provider (needs LEAPFLOW_LLM_* credentials)
	LEAPFLOW_TEST_LLM_MODE=live uv run pytest tests/journeys -q -m e2e

# ── Recorded provider traffic ─────────────────────────────────────────────────
# Two stores, two jobs. `cassettes/` holds the deterministic inputs the offline
# lanes replay; `recordings/` holds real provider traffic, which is evidence of
# wire shape rather than a replay input — a multi-turn agent conversation cannot
# be replayed from a recording, because each turn's prompt embeds the exact
# round-by-round history of the turns before it.

seed-cassettes:  ## Rebuild the offline replay store from each journey's script
	LEAPFLOW_TEST_LLM_MODE=seed uv run pytest tests/journeys -q -m e2e

record-traffic:  ## Capture real provider traffic into recordings/ (needs credentials)
	LEAPFLOW_TEST_LLM_MODE=record uv run pytest tests/journeys -q -m e2e

sync-fixtures:  ## Derive mock-layer response shapes from both stores
	uv run python tools/sync_fixtures.py

# ── LeapFlow CLI (pass PROMPT via ARGS, e.g. make brain ARGS='--prompt "hello"')
brain:  ## Start Brain process
	uv run leap $(ARGS)

cua-check:  ## Check cua-driver installation status
	@which cua-driver > /dev/null 2>&1 && echo "✓ cua-driver: $$(cua-driver --version 2>/dev/null || echo 'installed')" || echo "✗ cua-driver not found. Install: brew install trycua/tap/cua-driver"
