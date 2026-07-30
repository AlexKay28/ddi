# experiments/

Every experiment on `main` is one YAML file in this directory plus one line in the
top-level `Makefile`. Running an experiment is `make exp-<name>`; publishing the
resulting JSON summary to a tracking GitHub issue is `make publish-summary`.

Docs describing the operator-facing flow (container base, HF prefetch, offline mode,
one-experiment-per-GPU rule) live in [`SERVERLESS.md`](SERVERLESS.md).

This layout landed via the [#81 refactor](https://github.com/AlexKay28/ddi/issues/81);
the research contract each YAML serves is in
[`../paper/RESEARCH_PLAN.md`](../paper/RESEARCH_PLAN.md).

---

## Paper claim map

The preprint is scoped to three claims (per [#81](https://github.com/AlexKay28/ddi/issues/81)):

- **C1 — quality parity.** DDI masks match `full` causal within the non-inferiority
  margin `Δ` on the four direct QA datasets.
- **C3 — speed and realization ratio.** Wall-clock speedup vs. the analytical
  sparsity ceiling across `(N, L, backend)` grids.
- **C4 — caching and restart layout.** Per-document KV blocks are reusable across
  requests and independent of retrieval order.

C5 (failure envelope), C6 (distractor robustness), fine-tuning, PCW, and composition
are out of scope for the preprint and their experiments are not indexed here.

---

## Index

None of these YAMLs name a model — `make exp-<name> MODEL=<hf-id>` supplies it at
invocation time, so the same file drives every model in the sweep.

| YAML | claim | one-line purpose | tracking issue |
|---|---|---|---|
| [`quality.yaml`](quality.yaml) | C1 | Quality parity across the four direct QA datasets (HotpotQA, MuSiQue, KILT-NQ, synthetic) under `full` / `ddi_qf` / `blind`. Run with `MODEL=Qwen/Qwen2.5-7B-Instruct` → [#74](https://github.com/AlexKay28/ddi/issues/74), `MODEL=meta-llama/Llama-3.1-8B-Instruct` → [#75](https://github.com/AlexKay28/ddi/issues/75). | #74 / #75 |
| [`performance.yaml`](performance.yaml) | C3 | Timing grid over `(N, L, backend, mask)` — TTFT / TPOT / analytical / realized / realization ratio. Same model→issue mapping as `quality.yaml`. | #74 / #75 |
| [`cache_correctness.yaml`](cache_correctness.yaml) | C4 | Correctness gate for per-document KV cache reuse — cached logits match recomputed logits within tolerance. | *(pending — filed under C4 rollup)* |
| [`cache_throughput.yaml`](cache_throughput.yaml) | C4 | Cached vs. recomputed throughput on repeated documents; replaces the legacy F5 figure. | *(pending — filed under C4 rollup)* |
| [`cache_restart.yaml`](cache_restart.yaml) | C4 | Restart-position layout produces the same answers as sequential layout — parity precondition for cross-request cache reuse. | *(pending — filed under C4 rollup)* |

---

## Adding a new experiment

Two steps, no Python:

1. **Write the YAML** in this directory. Copy the closest existing file, pick a
   `kind:` (`quality` / `timing` / `cache`), name it `<claim>.yaml` or
   `<task>_c<n>.yaml` to match the index scheme. Do **not** hardcode a `model:`
   key — the model is supplied at invocation time (see below) so one YAML serves
   every model in the sweep.
2. **Run it** with `make exp-<name> MODEL=<hf-id>` (drop the `.yaml`). The
   `Makefile` has a pattern rule — no new `Makefile` line is needed.

The runner (`python -m ddi.run experiments/<name>.yaml --model <hf-id>`) writes
rows to `results/runs.jsonl` and emits `logs/<name>_summary.json` from the YAML's
`report:` block. `ddi.aggregate` groups by model automatically, so running the
same YAML with two different `MODEL=` values against the same store produces one
summary with a section per model. Publish it to the tracking issue with:

```
make publish-summary SUMMARY=logs/<name>_summary.json ISSUE=<n>
```

For the container / prefetch / offline recipe, see [`SERVERLESS.md`](SERVERLESS.md).
