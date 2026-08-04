# DDI reproducibility Makefile — see experiments/SERVERLESS.md for the full flow.

.PHONY: exp-% finetune hf-fetch yt-fetch yt-fetch-records publish-summary test list-experiments gpu-check help

YT_PROXY ?= watt
YT_MODELS_ROOT ?= //home/gena/alexkay28/tickets/ALIGNMENT-2445/models
YT_RECORDS_ROOT ?= //home/gena/alexkay28/tickets/ALIGNMENT-2445/data/records_cache

# `finetune` defaults, baked in after a live OOM + offline-fetch run (issue #120):
#   FT_OFFLINE          this repo has no direct HF Hub access outside a networked
#                        dev box; set 0 to allow live Hub fetches instead.
#   FT_GRAD_CHECKPOINT   a 7B teacher+student pair sits right at an 80GB GPU's
#                        limit without it (see ddi/finetune.py's AdamW8bit comment);
#                        set 0 to trade the memory headroom back for speed.
#   FT_CUDA_ALLOC_CONF   works around allocator fragmentation on the same tight-fit
#                        GPUs; rarely needs overriding.
#   NUM_GPUS             single-host data parallelism (issue #123). Default 1 = the
#                        original single-device path. >1 launches training under
#                        `torchrun --nproc_per_node=$(NUM_GPUS)` and passes the
#                        value through as `--num-gpus` so the trainer knows to
#                        expect the DDP env.
FT_OFFLINE ?= 1
FT_GRAD_CHECKPOINT ?= 1
FT_CUDA_ALLOC_CONF ?= expandable_segments:True
NUM_GPUS ?= 1

# Python interpreter used by every target below except `test`. Default is the
# system `python3` — matches the offline serverless container's setup where
# torch/transformers/numpy are pre-installed at the OS level. Override on a
# dev machine to run inside the uv-managed venv:
#
#   PY="uv run --no-sync python" make exp-c1 MODEL=Qwen/Qwen2.5-7B-Instruct
#
# `test` intentionally does its own `uv sync --extra dev` + `uv run python -m
# pytest` — that target is meant for a networked dev box, not the container.
PY ?= python3

help:
	@echo "usage:"
	@echo "  make exp-<name> MODEL=<hf-id> [BACKENDS=sdpa,fa2]   # run experiments/<name>.yaml"
	@echo "  make finetune MODEL=<hf-id> [NUM_GPUS=N]  # self-distillation train (issue #120) — writes"
	@echo "                                     # results/checkpoints/... + finetune_summary.json"
	@echo "                                     # (offline + grad-checkpoint on by default; override"
	@echo "                                     # via FT_OFFLINE/FT_GRAD_CHECKPOINT/FT_CUDA_ALLOC_CONF)"
	@echo "                                     # NUM_GPUS=N>1 runs under torchrun for DDP (issue #123)"
	@echo "  make hf-fetch MODEL=<hf-id>        # cache a HF model into HF_HOME (from HF Hub)"
	@echo "  make yt-fetch MODEL=<hf-id>        # cache a HF model into HF_HOME (from YT, no HF access needed)"
	@echo "  make yt-fetch-records              # restore results/splits/*_records.jsonl from YT (dataset cache)"
	@echo "  make publish-summary SUMMARY=... ISSUE=<n>"
	@echo "  make test                          # run pytest"
	@echo "  make list-experiments              # list all available exp-* targets"
	@echo "  make gpu-check                     # confirm CUDA is visible"

# One-shot experiment: matches any experiments/<name>.yaml.
#   MODEL     optional at the make layer — ddi.run raises loudly if the YAML has no `model:`
#             and no --model was passed.
#   BACKENDS  comma-separated override of the YAML's `backends:` list (e.g. `sdpa` or
#             `sdpa,fa2`). Only kind: timing consumes it; ignored for other kinds. Handy for
#             skipping the fa2 sub-blocks that #70 leaves empty on hosts without flash-attn:
#             `make exp-c3 MODEL=... BACKENDS=sdpa`.
exp-%:
	$(PY) -m ddi.run experiments/$*.yaml $(if $(MODEL),--model $(MODEL)) $(if $(BACKENDS),--backends $(BACKENDS))
	$(PY) -m ddi.aggregate experiments/$*.yaml --out logs/$*_summary.json $(if $(MODEL),--model $(MODEL))

# Self-distillation trainer (issue #120): teacher = base + full mask, student = base + ddi_qf.
# Env vars set here are the FT_* defaults above — override any of them on the command line,
# e.g. `make finetune MODEL=... FT_GRAD_CHECKPOINT=0` to trade memory headroom back for
# speed. Unlike other kinds, `finetune` doesn't route through the `exp-%` pattern because
# the run step may need `torchrun` (issue #123) while the aggregate step stays single-
# process — swapping the launcher for one step only is what this rule owns. Summary file
# name/path still match what the pattern rule would have produced.
#
# NUM_GPUS>1 launches the run step under `torchrun --nproc_per_node=$(NUM_GPUS) --standalone`
# and passes `--num-gpus` so the trainer knows to expect the DDP env. NUM_GPUS=1 (default)
# invokes plain `$(PY) -m ddi.run` — behavior byte-identical to the pre-#123 target.
FT_LAUNCHER = $(if $(filter-out 1,$(NUM_GPUS)),torchrun --nproc_per_node=$(NUM_GPUS) --standalone,$(PY))

finetune:
	@test -n "$(MODEL)" || (echo "usage: make finetune MODEL=<hf-id> [NUM_GPUS=N]" && exit 2)
	HF_HUB_OFFLINE=$(FT_OFFLINE) TRANSFORMERS_OFFLINE=$(FT_OFFLINE) \
	DDI_GRAD_CHECKPOINT=$(FT_GRAD_CHECKPOINT) \
	PYTORCH_CUDA_ALLOC_CONF=$(FT_CUDA_ALLOC_CONF) \
	$(FT_LAUNCHER) -m ddi.run experiments/finetune.yaml --model $(MODEL) --num-gpus $(NUM_GPUS)
	$(PY) -m ddi.aggregate experiments/finetune.yaml --out logs/finetune_summary.json --model $(MODEL)

# Fetch a HF model once from the HF Hub (respects HF_HOME).
hf-fetch:
	@test -n "$(MODEL)" || (echo "usage: make hf-fetch MODEL=<hf-id>" && exit 2)
	$(PY) -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
	           AutoTokenizer.from_pretrained('$(MODEL)'); \
	           AutoModelForCausalLM.from_pretrained('$(MODEL)', torch_dtype='auto')"

# Fetch a HF model from our YT mirror instead of the HF Hub, for hosts with no
# direct HF access. Rebuilds a local HF_HOME cache (refs/main + a synthetic
# snapshot dir) so `from_pretrained(MODEL)` resolves offline afterwards — same
# contract as `hf-fetch`, different source. Requires the model to have already
# been mirrored to YT_MODELS_ROOT (see experiments/SERVERLESS.md).
yt-fetch:
	@test -n "$(MODEL)" || (echo "usage: make yt-fetch MODEL=<hf-id>" && exit 2)
	@test -n "$(HF_HOME)" || (echo "usage: HF_HOME=<dir> make yt-fetch MODEL=<hf-id>" && exit 2)
	@repo_name="$$(basename $(MODEL))"; \
	cache_name="models--$$(echo '$(MODEL)' | sed 's#/#--#g')"; \
	src="$(YT_MODELS_ROOT)/$$repo_name"; \
	dest="$(HF_HOME)/hub/$$cache_name"; \
	snap="$$dest/snapshots/yt-cache"; \
	yt --proxy=$(YT_PROXY) exists "$$src" | grep -q true || \
	    (echo "no such model on YT: $$src" && exit 1); \
	mkdir -p "$$snap" "$$dest/refs"; \
	printf yt-cache > "$$dest/refs/main"; \
	for f in $$(yt --proxy=$(YT_PROXY) list "$$src"); do \
	    echo "fetching $$f ..."; \
	    yt --proxy=$(YT_PROXY) read-file "$$src/$$f" > "$$snap/$$f"; \
	done; \
	echo "cached $(MODEL) -> $$dest"

# Restore the primed dataset record cache from our YT mirror, for hosts with
# no direct HF access. Unlike yt-fetch (models), this does NOT reconstruct an
# HF_HOME hub cache: mirroring the raw HF hub snapshot is not sufficient to
# make `datasets.load_dataset()` work offline (its offline path needs its own
# prebuilt arrow cache, not just the raw source files). Instead this restores
# results/splits/*_records.jsonl directly (see ddi/data/records_cache.py) -
# every ddi.data loader checks that cache first and never touches the HF Hub
# on a hit. Requires the cache to have already been mirrored to
# YT_RECORDS_ROOT (see upload_records_cache_to_yt.sh).
yt-fetch-records:
	@mkdir -p results/splits; \
	for f in $$(yt --proxy=$(YT_PROXY) list $(YT_RECORDS_ROOT)); do \
	    echo "fetching $$f ..."; \
	    yt --proxy=$(YT_PROXY) read-file "$(YT_RECORDS_ROOT)/$$f" > "results/splits/$$f"; \
	done; \
	echo "cached dataset records -> results/splits/"

# Publish a summary.md to a tracking GitHub issue.
publish-summary:
	@test -n "$(SUMMARY)" && test -n "$(ISSUE)" || \
	    (echo "usage: make publish-summary SUMMARY=<file> ISSUE=<n>" && exit 2)
	gh issue comment $(ISSUE) --body-file $(SUMMARY)

# Housekeeping.
# Note: `uv run pytest` picks up a system-installed pytest without torch — use
# `python -m pytest` via the venv interpreter instead. Also ensure the [dev]
# extra (pytest) is installed before invoking.
test:
	uv sync --extra dev
	uv run python -m pytest tests/ -q

list-experiments:
	@ls experiments/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml//' | sed 's/^/  exp-/'

gpu-check:
	$(PY) -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name())"
