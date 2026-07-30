# DDI reproducibility Makefile — see experiments/SERVERLESS.md for the full flow.

.PHONY: exp-% hf-fetch yt-fetch publish-summary test list-experiments gpu-check help

YT_PROXY ?= watt
YT_MODELS_ROOT ?= //home/gena/alexkay28/tickets/ALIGNMENT-2445/models

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
	@echo "  make hf-fetch MODEL=<hf-id>        # cache a HF model into HF_HOME (from HF Hub)"
	@echo "  make yt-fetch MODEL=<hf-id>        # cache a HF model into HF_HOME (from YT, no HF access needed)"
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
	$(PY) -m ddi.aggregate experiments/$*.yaml --out logs/$*_summary.md

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
