# DDI — Decoupled Document Inference
# Central command surface for serverless experiment runs.
# `make setup` once per container; then `make <experiment> [SEED=...] [CONFIG=...]`.
# See paper/RESEARCH_PLAN.md and issue #1 for the roadmap.
#
# Every experiment target invokes `uv run ddi-run --config <yaml> --seed <n>`.
# Configs live under `ddi/configs/<name>.yaml` — created as each phase's runner
# lands. Override CONFIG=<path> to point at a custom config.

# Overridable.
UV      ?= uv
PYTEST  ?= $(UV) run pytest
RUN     ?= $(UV) run ddi-run
SEED    ?= 0
SEEDS   ?= 0 1 2
CONFIG  ?= ddi/configs/example.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show every target grouped by section.
	@awk 'BEGIN {FS = ":.*?## "} \
		/^# ==/ {print ""; getline; sub(/^# */, "", $$0); printf "\033[1m%s\033[0m\n", $$0; next} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: UV=$(UV)  CONFIG=$(CONFIG)  SEED=$(SEED)  SEEDS=$(SEEDS)"

# ===================================================================
# setup
# ===================================================================

.PHONY: setup
setup:  ## Install CPU dev environment (uv pip install -e ".[dev]").
	$(UV) pip install --system -e ".[dev]"

.PHONY: setup-gpu
setup-gpu:  ## setup + FlashAttention-2 + report extras (for serverless GPU boxes).
	$(UV) pip install --system -e ".[dev,fa2,report]"

# ===================================================================
# dev
# ===================================================================

.PHONY: test
test:  ## Full test suite.
	$(PYTEST)

.PHONY: lint
lint:  ## ruff check.
	$(UV) run ruff check .

.PHONY: format
format:  ## ruff format.
	$(UV) run ruff format .

# ===================================================================
# generic runner
# ===================================================================

.PHONY: run
run:  ## One run. Override CONFIG=<path> SEED=<n>.
	$(RUN) --config $(CONFIG) --seed $(SEED)

.PHONY: run-seeds
run-seeds:  ## Same config across SEEDS (default: 0 1 2). Useful on one big VM.
	@for seed in $(SEEDS); do $(RUN) --config $(CONFIG) --seed $$seed; done

# ===================================================================
# Phase 2 — mask family quality (C1, C2, C6)
# ===================================================================

.PHONY: sweep-train-free
sweep-train-free:  ## [2.1] Train-free mask sweep — primary quality result (issue #13).
	$(RUN) --config ddi/configs/phase2_train_free.yaml --seed $(SEED)

.PHONY: sweep-finetuned
sweep-finetuned:  ## [2.2] Fine-tuned mask sweep — conditional on 2.1 (issue #14).
	$(RUN) --config ddi/configs/phase2_finetuned.yaml --seed $(SEED)

.PHONY: sweep-bridge
sweep-bridge:  ## [2.3] Bridge-width curve — paper centerpiece (issue #15).
	$(RUN) --config ddi/configs/phase2_bridge.yaml --seed $(SEED)

.PHONY: probe-instruct
probe-instruct:  ## [2.4] Base vs instruct mechanism probe (issue #16).
	$(RUN) --config ddi/configs/phase2_base_vs_instruct.yaml --seed $(SEED)

# ===================================================================
# Phase 3 — efficiency (C3)
# ===================================================================

.PHONY: timing-N
timing-N:  ## [3.2] Document-count sweep vs analytical (issue #18).
	$(RUN) --config ddi/configs/phase3_timing_N.yaml --seed $(SEED)

.PHONY: timing-L
timing-L:  ## [3.3] Document-length sweep — find the crossover (issue #19).
	$(RUN) --config ddi/configs/phase3_timing_L.yaml --seed $(SEED)

.PHONY: backends-NxL
backends-NxL:  ## [3.4] Backend comparison across N × L (issue #20).
	$(RUN) --config ddi/configs/phase3_backends_NxL.yaml --seed $(SEED)

.PHONY: realization
realization:  ## [3.5] Realization-ratio analysis — the 9× gap (issue #21).
	$(RUN) --config ddi/configs/phase3_realization.yaml --seed $(SEED)

# ===================================================================
# Phase 4 — caching + order invariance (C4) — co-headline
# ===================================================================

.PHONY: cache-restart
cache-restart:  ## [4.1] Position-restart quality — does restart cost anything? (issue #22).
	$(RUN) --config ddi/configs/phase4_restart.yaml --seed $(SEED)

.PHONY: cache-store
cache-store:  ## [4.2] Per-document KV cache correctness (issue #23).
	$(RUN) --config ddi/configs/phase4_cache_store.yaml --seed $(SEED)

.PHONY: cache-order
cache-order:  ## [4.3] Retrieval-order invariance (issue #24).
	$(RUN) --config ddi/configs/phase4_order.yaml --seed $(SEED)

.PHONY: cache-throughput
cache-throughput:  ## [4.4] Throughput under document reuse (issue #25).
	$(RUN) --config ddi/configs/phase4_throughput.yaml --seed $(SEED)

# ===================================================================
# Phase 5 — failure envelope (C5)
# ===================================================================

.PHONY: split-hop
split-hop:  ## [5.1] Single-hop vs multi-hop degradation split (issue #26).
	$(RUN) --config ddi/configs/phase5_split_hop.yaml --seed $(SEED)

.PHONY: pcw-baseline
pcw-baseline:  ## [5.2] Parallel Context Windows baseline (issue #27).
	$(RUN) --config ddi/configs/phase5_pcw.yaml --seed $(SEED)

.PHONY: composition-probe
composition-probe:  ## [5.3] Synthetic cross-doc composition probe (issue #28).
	$(RUN) --config ddi/configs/phase5_composition.yaml --seed $(SEED)

# ===================================================================
# reporting
# ===================================================================

.PHONY: report
report:  ## Render one report. `make report NAME=T2` or `make report LIST=1`. Figures need OUT=<path>.
	@$(UV) run python -m ddi.report_cli \
		$(if $(LIST),--list,) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(OUT),--output $(OUT),)

# ===================================================================
# housekeeping
# ===================================================================

.PHONY: clean
clean:  ## Remove Python caches and build artifacts. Never touches results/.
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
