# experiments/

Every experiment on `main` is one YAML file in this directory plus one line in the
top-level `Makefile`. Running an experiment is `make exp-<name>`; publishing the
resulting Markdown summary to a tracking GitHub issue is `make publish-summary`.

Docs describing the operator-facing flow (container base, HF prefetch, offline mode,
one-experiment-per-GPU rule) live in [`SERVERLESS.md`](SERVERLESS.md).

The design that produced this layout is in the parent refactor plan
([#81](https://github.com/AlexKay28/ddi/issues/81)); the research contract each YAML
serves is in [`../paper/RESEARCH_PLAN.md`](../paper/RESEARCH_PLAN.md).

---

## Paper claim map

The preprint is scoped to three claims (per [#81](https://github.com/AlexKay28/ddi/issues/81)):

- **C1 — quality parity.** DDI masks match `full` causal within the non-inferiority
  margin `Δ` on the five direct QA datasets.
- **C3 — speed and realization ratio.** Wall-clock speedup vs. the analytical
  sparsity ceiling across `(N, L, backend)` grids.
- **C4 — caching and restart layout.** Per-document KV blocks are reusable across
  requests and independent of retrieval order.

C5 (failure envelope), C6 (distractor robustness), fine-tuning, PCW, and composition
are out of scope for the preprint and their experiments are not indexed here.

---

## Index

| YAML | claim | one-line purpose | tracking issue |
|---|---|---|---|
| [`qwen_instruct_c1.yaml`](qwen_instruct_c1.yaml) | C1 | Qwen2.5-7B-Instruct quality parity across the five direct QA datasets (NQ, HotpotQA, MuSiQue, KILT-NQ, synthetic) under `full` / `ddi_qf` / `blind`. | [#74](https://github.com/AlexKay28/ddi/issues/74) |
| [`llama_instruct_c1.yaml`](llama_instruct_c1.yaml) | C1 | Llama-3.1-8B-Instruct quality parity across the same five datasets and mask set. | [#75](https://github.com/AlexKay28/ddi/issues/75) |
| [`qwen_instruct_c3.yaml`](qwen_instruct_c3.yaml) | C3 | Qwen timing grid over `(N, L, backend, mask)` — TTFT / TPOT / analytical / realized / realization ratio. | [#74](https://github.com/AlexKay28/ddi/issues/74) |
| [`llama_instruct_c3.yaml`](llama_instruct_c3.yaml) | C3 | Llama timing grid, same axes. | [#75](https://github.com/AlexKay28/ddi/issues/75) |
| [`cache_correctness_c4.yaml`](cache_correctness_c4.yaml) | C4 | Correctness gate for per-document KV cache reuse — cached logits match recomputed logits within tolerance. | *(pending — filed under C4 rollup)* |
| [`cache_throughput_c4.yaml`](cache_throughput_c4.yaml) | C4 | Cached vs. recomputed throughput on repeated documents; replaces the legacy F5 figure. | *(pending — filed under C4 rollup)* |
| [`restart_quality_c4.yaml`](restart_quality_c4.yaml) | C4 | Restart-position layout produces the same answers as sequential layout — parity precondition for cross-request cache reuse. | *(pending — filed under C4 rollup)* |

The YAML files themselves land in [#86](https://github.com/AlexKay28/ddi/issues/86);
this table is the source of truth for what will be there.

---

## Legacy bash drivers

The `exp_01_*.bash` scripts at the repo root are historical — they were the
Phase 1 / Phase 2 driver shape from before the YAML refactor. They stay in git for
provenance of past runs but are not part of the current flow and are not indexed
here. The one exception is `exp_3_grid.bash`, which is being ported to shell out to
`python -m ddi.run` over the two C3 YAMLs and will remain as a "run both models"
convenience wrapper.

---

## Adding a new experiment

Two steps, no Python:

1. **Write the YAML** in this directory. Copy the closest existing file, pick a
   `kind:` (`quality` / `timing` / `cache`), name it `<model>_<claim>.yaml` or
   `<task>_c<n>.yaml` to match the index scheme.
2. **Run it** with `make exp-<name>` (drop the `.yaml`). The `Makefile` has a
   pattern rule — no new `Makefile` line is needed.

The runner (`python -m ddi.run experiments/<name>.yaml`) writes rows to
`results/runs.jsonl` and emits `logs/<name>_summary.md` from the YAML's `report:`
block. Publish it to the tracking issue with:

```
make publish-summary SUMMARY=logs/<name>_summary.md ISSUE=<n>
```

For the container / prefetch / offline recipe, see [`SERVERLESS.md`](SERVERLESS.md).
