# DDI — see experiments/SERVERLESS.md for the full flow.
#
# This file is a dispatcher. The targets live in two files, split by what they administer:
#
#   Makefile.exp       running the science — exp-*, finetune, yt-fetch, lint/test.
#                      Knows nothing about Nirvana.
#   Makefile.nirvana   administering the GPU runs — publish a graph, check its pin,
#                      monitor what is running. Knows nothing about the experiments.
#
# They share no variables, so an edit to one cannot reach the other. Either can also be
# used on its own (`make -f Makefile.nirvana nv-list`).
#
# WHY THIS FILE STILL EXISTS, rather than the split being two standalone files: the Nirvana
# jobs shell out to bare `make` inside the exported Arcadia tree —
# `make yt-fetch-records`, `make yt-fetch MODEL=...`, `make finetune MODEL=...`,
# `make exp-quality MODEL=...` (see nirvana/graphs/scripts.py). Those run with no `-f`, so
# a tree whose root Makefile does not answer them fails at the first block of every
# published graph. Keep this dispatcher, and keep those target names reachable from it.

include Makefile.exp
include Makefile.nirvana

.PHONY: help
.DEFAULT_GOAL := help

help:
	@$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) help-exp
	@echo ""
	@$(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) help-nirvana
