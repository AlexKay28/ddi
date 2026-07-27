# DDI — Decoupled Document Inference
# Primary command surface. See paper/RESEARCH_PLAN.md and issue #1 for the roadmap.

# Overridable — invoke as e.g. `make run CONFIG=foo.yaml SEED=3`.
PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
PYTEST  ?= $(PYTHON) -m pytest
RUFF    ?= $(PYTHON) -m ruff

SEED    ?= 0
CONFIG  ?= ddi/configs/example.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this list.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: PYTHON=$(PYTHON)  CONFIG=$(CONFIG)  SEED=$(SEED)"

# --- setup ---

.PHONY: install
install:  ## Editable install with dev extras.
	$(PIP) install -e ".[dev]"

.PHONY: install-fa2
install-fa2:  ## Add FlashAttention-2 extras.
	$(PIP) install -e ".[dev,fa2]"

.PHONY: install-report
install-report:  ## Add report extras (pandas, matplotlib).
	$(PIP) install -e ".[dev,report]"

.PHONY: install-all
install-all:  ## Every optional extra (fa2, triton, report).
	$(PIP) install -e ".[dev,fa2,triton,report]"

# --- quality ---

.PHONY: test
test:  ## Run the test suite.
	$(PYTEST)

.PHONY: test-fast
test-fast:  ## Stop on first failure.
	$(PYTEST) -x

.PHONY: lint
lint:  ## ruff check.
	$(RUFF) check .

.PHONY: format
format:  ## ruff format.
	$(RUFF) format .

# --- experiments ---

.PHONY: run
run:  ## Run one experiment. Override CONFIG=<path> SEED=<n>.
	$(PYTHON) -m ddi.cli --config $(CONFIG) --seed $(SEED)

# Phase-specific targets get added here as sub-issues land:
#   gates              # Phase 0.5, issue #4 — three correctness gates
#   report             # Phase 0.7, issue #8 — regenerate tables/figures from results/runs.jsonl
#   sweep-train-free   # Phase 2.1, issue #13 — primary quality result
#   sweep-bridge       # Phase 2.3, issue #15 — Pareto frontier (paper centerpiece)
#   timing-sweep       # Phase 3.2, issue #18 — analytical vs realized speedup
#   caching-throughput # Phase 4.4, issue #25

# --- housekeeping ---

.PHONY: clean
clean:  ## Remove Python caches and build artifacts. Never touches results/.
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
