# DDI — Decoupled Document Inference
# Central command surface. Every experiment on a serverless GPU box is invoked as
# `make <target> [VAR=value ...]` inside a container that has `uv` and `git` available.
# See paper/RESEARCH_PLAN.md and issue #1 for the roadmap.

# Overridable — e.g. `make run CONFIG=foo.yaml SEED=3`.
UV      ?= uv
PYTHON  ?= python3
PYTEST  ?= $(UV) run pytest
RUFF    ?= $(UV) run ruff
RUN     ?= $(UV) run ddi-run

SEED    ?= 0
CONFIG  ?= ddi/configs/example.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this list.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: UV=$(UV)  PYTHON=$(PYTHON)  CONFIG=$(CONFIG)  SEED=$(SEED)"

# --- setup (uv-based; first step on any serverless container) ---

.PHONY: setup
setup:  ## Install with dev extras (uv pip install -e ".[dev]").
	$(UV) pip install --system -e ".[dev]"

.PHONY: setup-fa2
setup-fa2:  ## setup + FlashAttention-2 (needs CUDA at install time).
	$(UV) pip install --system -e ".[dev,fa2]"

.PHONY: setup-report
setup-report:  ## setup + pandas/matplotlib for report generation.
	$(UV) pip install --system -e ".[dev,report]"

.PHONY: setup-all
setup-all:  ## setup + every optional extra (fa2, triton, report).
	$(UV) pip install --system -e ".[dev,fa2,triton,report]"

# --- quality (CI runs `make lint test`; serverless can re-run before an experiment) ---

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

# --- experiments (serverless entry points) ---
# One serverless invocation = one `make <experiment-target>`. Overrides via VAR=value.
# Every target writes to results/runs.jsonl (ground rule 2) with full provenance
# (ground rule 1); tables and figures regenerate from that store (ground rule 3).

.PHONY: run
run:  ## Run one experiment. Override CONFIG=<path> SEED=<n>.
	$(RUN) --config $(CONFIG) --seed $(SEED)

# Phase-specific serverless targets get added here as sub-issues land. Each will
# expand to one or more `run` invocations and live behind a task registered in
# ddi.cli.register_runner. Kept as documentation until then:
#
#   gates              # Phase 0.5, issue #4 — three correctness gates (harness sanity)
#   report             # Phase 0.7, issue #8 — regenerate tables/figures from the store
#   sweep-train-free   # Phase 2.1, issue #13 — primary quality result
#   sweep-bridge       # Phase 2.3, issue #15 — Pareto frontier (paper centerpiece)
#   timing-sweep       # Phase 3.2, issue #18 — analytical vs realized speedup
#   caching-throughput # Phase 4.4, issue #25 — per-document cache throughput

# --- housekeeping ---

.PHONY: clean
clean:  ## Remove Python caches and build artifacts. Never touches results/.
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
