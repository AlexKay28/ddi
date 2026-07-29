# DDI reproducibility Makefile — see experiments/SERVERLESS.md for the full flow.

.PHONY: exp-% prefetch publish-summary test list-experiments gpu-check help

help:
	@echo "usage:"
	@echo "  make exp-<name>                    # run experiments/<name>.yaml end-to-end"
	@echo "  make prefetch MODEL=<hf-id>        # cache a HF model into HF_HOME"
	@echo "  make publish-summary SUMMARY=... ISSUE=<n>"
	@echo "  make test                          # run pytest"
	@echo "  make list-experiments              # list all available exp-* targets"
	@echo "  make gpu-check                     # confirm CUDA is visible"

# One-shot experiment: matches any experiments/<name>.yaml
exp-%:
	python -m ddi.run experiments/$*.yaml
	python -m ddi.aggregate experiments/$*.yaml --out logs/$*_summary.md

# Prefetch a HF model once (respects HF_HOME).
prefetch:
	@test -n "$(MODEL)" || (echo "usage: make prefetch MODEL=<hf-id>" && exit 2)
	python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
	           AutoTokenizer.from_pretrained('$(MODEL)'); \
	           AutoModelForCausalLM.from_pretrained('$(MODEL)', torch_dtype='auto')"

# Publish a summary.md to a tracking GitHub issue.
publish-summary:
	@test -n "$(SUMMARY)" && test -n "$(ISSUE)" || \
	    (echo "usage: make publish-summary SUMMARY=<file> ISSUE=<n>" && exit 2)
	gh issue comment $(ISSUE) --body-file $(SUMMARY)

# Housekeeping.
test:
	uv run pytest tests/ -q

list-experiments:
	@ls experiments/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml//' | sed 's/^/  exp-/'

gpu-check:
	python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name())"
