# DDI — Decoupled Document Inference
# Serverless entry point. Everything here is what a container needs to run,
# stage, evaluate, and publish experiments. Roadmap: paper/RESEARCH_PLAN.md
# and issue #1. Discipline: every run writes to results/runs.jsonl with full
# provenance (ground rule 1); reports regenerate from that store (ground rule 3).

UV       ?= uv
PYTEST   ?= $(UV) run pytest
RUN      ?= $(UV) run ddi-run
HFCLI    ?= $(UV) run huggingface-cli

SEED     ?= 0
SEEDS    ?= 0 1 2
CONFIG   ?= ddi/configs/example.yaml

# Upload targets — override on the command line.
HF_REPO  ?= AlexKay28/ddi
DATASET  ?= nq
CKPT     ?= checkpoints/phase2_finetune

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show every target grouped by section.
	@awk 'BEGIN {FS = ":.*?## "} \
		/^# ==/ {print ""; getline; sub(/^# */, "", $$0); printf "\033[1m%s\033[0m\n", $$0; next} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: UV=$(UV)  CONFIG=$(CONFIG)  SEED=$(SEED)  SEEDS=$(SEEDS)"
	@echo "Upload:    HF_REPO=$(HF_REPO)  DATASET=$(DATASET)  CKPT=$(CKPT)"

# ===================================================================
# install (first command on every fresh serverless container)
# ===================================================================

.PHONY: setup
setup:  ## Install CPU dev environment (uv pip install -e ".[dev]").
	$(UV) pip install --system -e ".[dev]"

.PHONY: setup-gpu
setup-gpu:  ## setup + FlashAttention-2 + report extras (for GPU boxes; expects CUDA torch pre-installed on the image).
	$(UV) pip install --system -e ".[dev,fa2,report]"
	@$(MAKE) --no-print-directory gpu-check

# ===================================================================
# tests (preflight; gates are the ground-rule-6 enforcement)
# ===================================================================

.PHONY: test
test:  ## Full test suite.
	$(PYTEST)

.PHONY: gates
gates:  ## Correctness gates only — cheap serverless preflight.
	$(PYTEST) -m gate

.PHONY: gpu-check
gpu-check:  ## Print GPU info and exit non-zero if CUDA is not available.
	@$(UV) run python -c "from ddi.gpu import configure, require_cuda; configure(); require_cuda()"

# ===================================================================
# datasets upload (freeze + publish so every subsequent run pulls identical splits)
# ===================================================================

.PHONY: upload-dataset
upload-dataset:  ## Push results/splits/$(DATASET).json to HF hub. Override DATASET, HF_REPO.
	$(HFCLI) upload $(HF_REPO) results/splits/$(DATASET).json splits/$(DATASET).json --repo-type dataset

# ===================================================================
# models upload (publish a checkpoint so eval containers can pull it)
# ===================================================================

.PHONY: upload-model
upload-model:  ## Push a local checkpoint to HF hub. Override CKPT, HF_REPO.
	$(HFCLI) upload $(HF_REPO) $(CKPT) $(notdir $(CKPT)) --repo-type model

# ===================================================================
# experiments — one make invocation = one seed of one cell
# ===================================================================

.PHONY: run
run:  ## Generic runner. Override CONFIG=<path> SEED=<n>.
	$(RUN) --config $(CONFIG) --seed $(SEED)

.PHONY: run-seeds
run-seeds:  ## Same CONFIG across SEEDS (default: 0 1 2). One big VM.
	@for seed in $(SEEDS); do $(RUN) --config $(CONFIG) --seed $$seed; done

# -- Phase 1 — datasets, metrics, baseline --

.PHONY: phase1-baseline
phase1-baseline:  ## [1.4] Full-causal baseline; feeds delta (issue #12).
	$(RUN) --config ddi/configs/phase1_baseline.yaml --seed $(SEED)

.PHONY: delta
delta:  ## [1.4] Compute Δ from stored baseline. Override MODEL DATASET METRIC [K].
	$(UV) run ddi-delta --model $(MODEL) --dataset $(DATASET) --metric $(METRIC) $(if $(K),--k $(K),) --record

# -- Phase 2 — mask family quality --

.PHONY: sweep-train-free
sweep-train-free:  ## [2.1] Train-free mask sweep — primary quality result (issue #13).
	$(RUN) --config ddi/configs/phase2_train_free.yaml --seed $(SEED)

.PHONY: finetune
finetune:  ## [2.2] Fine-tune with the mask present; writes a checkpoint (issue #14).
	$(RUN) --config ddi/configs/phase2_finetune.yaml --seed $(SEED)

.PHONY: sweep-bridge
sweep-bridge:  ## [2.3] Bridge-width curve — paper centerpiece (issue #15).
	$(RUN) --config ddi/configs/phase2_bridge.yaml --seed $(SEED)

.PHONY: probe-instruct
probe-instruct:  ## [2.4] Base vs instruct mechanism probe (issue #16).
	$(RUN) --config ddi/configs/phase2_base_vs_instruct.yaml --seed $(SEED)

# -- Phase 3 — efficiency --

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

# -- Phase 4 — caching (co-headline) --

.PHONY: cache-restart
cache-restart:  ## [4.1] Position-restart quality — cost of restarting? (issue #22).
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

# -- Phase 5 — failure envelope --

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
# results collection (regenerated from results/runs.jsonl only)
# ===================================================================

.PHONY: report
report:  ## Render one report. `make report NAME=T2` or `make report LIST=1`. Figures need OUT=<path>.
	@$(UV) run python -m ddi.report_cli \
		$(if $(LIST),--list,) \
		$(if $(NAME),--name $(NAME),) \
		$(if $(OUT),--output $(OUT),)

.PHONY: paper
paper:  ## Regenerate all tables/figures from the store, then compile paper/preprint.pdf if pdflatex is available.
	@bash paper/build.sh
