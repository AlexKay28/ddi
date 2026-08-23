# experiments/

Every experiment is one YAML file in this directory. There is **no** matching `Makefile`
line: `make exp-<name>` is a pattern rule over `experiments/<name>.yaml`, so a new YAML is
runnable the moment it exists. `make list-experiments` prints the current set.

Docs describing the operator-facing flow (container base, model prefetch, offline mode,
one-experiment-per-GPU rule) live in [`SERVERLESS.md`](SERVERLESS.md). The research contract
each YAML serves is in [`../paper/RESEARCH_PLAN.md`](../paper/RESEARCH_PLAN.md); the layout
landed via the [#81 refactor](https://github.com/AlexKay28/ddi/issues/81).

---

## Paper claim map

The preprint is scoped to three claims (per [#81](https://github.com/AlexKay28/ddi/issues/81)):

- **C1 — quality parity.** DDI masks match `full` causal within the non-inferiority
  margin `Δ` on the retrieval-QA datasets.
- **C3 — speed and realization ratio.** Wall-clock speedup vs. the analytical
  sparsity ceiling across `(N, L, backend)` grids.
- **C4 — caching and restart layout.** Per-document KV blocks are reusable across
  requests and independent of retrieval order.

C5 (failure envelope), C6 (distractor robustness), PCW and composition are out of preprint
scope and have no YAML here.

**Fine-tuning does.** `finetune.yaml` is indexed below and is the bulk of what the Nirvana
fleet is currently running — see the note in [`../README.md`](../README.md#2-scope--models-datasets-paper-claims).
Older text in this repo describing it as out of scope predates that campaign.

---

## Index

Seven YAMLs, and **none of them names a model** — verified: no file here has a top-level
`model:` key. `make exp-<name> MODEL=<hf-id>` supplies it at invocation time, so one file
drives every model in the sweep. `finetune.yaml` carries a comment explaining why its
default was removed rather than corrected: a checked-in default is never used by a real run
(every caller passes `--model`) and is only ever read by a human trying to learn what the
config trains, so a stale one is pure misinformation.

| YAML | kind | one-line purpose | tracking |
|---|---|---|---|
| [`smoke.yaml`](smoke.yaml) | `quality` | **Not a measurement.** The CPU-only end-to-end check: `synthetic_qa` × 3 masks × 4 records × 8 new tokens, ~10s, no GPU/Hub/YT/network. Run it as `make smoke`, which builds the tiny model it needs. | VIBEQUEUE-16 |
| [`quality.yaml`](quality.yaml) | `quality` | C1 parity across `hotpot_qa` / `musique` / `synthetic_qa` under `full` / `ddi_qf` / `blind`; em, f1, presence, loss. `MODEL=Qwen/Qwen2.5-7B-Instruct` → [#74](https://github.com/AlexKay28/ddi/issues/74), `MODEL=meta-llama/Llama-3.1-8B-Instruct` → [#75](https://github.com/AlexKay28/ddi/issues/75). | #74 / #75 |
| [`performance.yaml`](performance.yaml) | `timing` | C3 timing grid over `(N, L, backend, mask)` — TTFT / TPOT / sparsity / analytical / realized / realization. Same model→issue mapping as `quality.yaml`. | #74 / #75 |
| [`cache_correctness.yaml`](cache_correctness.yaml) | `cache` (`task: correctness`) | C4 gate: per-document cached logits must match a fresh recompute within tolerance. If this fails the other C4 YAMLs are moot. | *(pending — C4 rollup)* |
| [`cache_throughput.yaml`](cache_throughput.yaml) | `cache` (`task: throughput`) | C4 QPS on repeated documents across `ddi_cache` / `ddi_recompute` / `full_causal`. | *(pending — C4 rollup)* |
| [`cache_restart.yaml`](cache_restart.yaml) | `cache` (`task: restart_quality`) | C4 precondition: restart-position layout must answer the same as sequential layout. | *(pending — C4 rollup)* |
| [`finetune.yaml`](finetune.yaml) | `finetune` | Self-distillation — teacher = `full`, student = `ddi_qf`, KL on answer-position logits. Run via `make finetune MODEL=… [NUM_GPUS=N]`, not `make exp-finetune`: the training step may need `torchrun` while the aggregate step stays single-process. | #120 |

Two YAMLs were deleted in VIBEQUEUE-16 and should not be recreated: `finetune_smoke.yaml`
(superseded by `smoke.yaml`, which is cheaper and actually runs without a GPU) and the
`run_finetune.sh` wrapper around it.

### Datasets

`hotpot_qa`, `musique`, `synthetic_qa`. That is the whole registry —
`ddi.data.list_datasets()` returns exactly those three, and `nirvana/common/constant.py`
single-sources the same list as `EVAL_DATASETS` for the graphs.

`kilt_nq` and `nq` are **gone**, loaders and all; naming either now raises
`KeyError: unknown dataset`. Reasons are in [`../README.md`](../README.md#datasets) — briefly,
`kilt_nq`'s documents were Wikipedia titles rather than passage bodies, and `nq` ships no
passages at all, so every mask variant collapsed to `full` on it. Neither absence weakens a
result; leaving them in a `datasets:` list used to abort the run at that cell.

`synthetic_qa` is the only loader that needs no network and no primed record cache, which is
why it is what `smoke.yaml` uses.

---

## Adding a new experiment

Two steps, no Python:

1. **Write the YAML** in this directory. Copy the closest existing file and pick a `kind:`
   (`quality` / `timing` / `cache` / `finetune`). Do **not** hardcode `model:` — it is
   supplied at invocation time.
2. **Run it** with `make exp-<name> MODEL=<hf-id>` (drop the `.yaml`). No `Makefile` edit.

Then:

```bash
python -m ddi.run experiments/<name>.yaml --model <hf-id>     # writes rows to the store
python -m ddi.aggregate experiments/<name>.yaml --out logs/<name>_summary.json --model <hf-id>
```

which is exactly what `make exp-<name>` runs. Rows land in `results/runs.jsonl`; the summary
is JSON built from the YAML's `report:` block. `ddi.aggregate` groups by model, so running
the same YAML under two `MODEL=` values against one store yields one summary with a section
per model.

### Keys are validated — both kinds of them

Neither half of a config is a free-form dict any more, and both checks exist because a
silently-ignored key cost real GPU time:

- **`--set` / `SETS=`** keys are checked against what the config's `kind:` actually reads
  (`_SET_KEYS` in `ddi/run.py`). An unknown key raises and names the valid ones. It used to be
  merged, folded into `cfg_hash` — so the run looked distinct — and then never read; that is
  how a 27B row passed `--set trainable_layers=` to a pinned revision predating the key,
  attempted a full finetune and OOMed an 8-GPU slot with no error anywhere.
  `--set-unchecked` is the deliberate bypass, for the one case the check cannot see: job-side
  code at a *different* revision that reads a key this one does not.
- **`report.tables[]`** keys are checked against `_TABLE_KEYS` in `ddi/aggregate.py`. For every
  kind except `finetune_loss` / `perplexity` (which dump every matching row) the only valid
  keys are `kind` and `metric`. `rows:`, `cols:` and `split_by:` used to appear on most
  tables; nothing read them — the JSON body shape is fixed per kind — and they now raise.

### `output_store:` and `--store` are not connected

If a YAML sets `output_store:`, `ddi.run` writes there — but `ddi.aggregate` does **not** read
that key. It takes `--store`, defaulting to `results/runs.jsonl`. `make exp-<name>` never
passes `--store`, so a YAML with a custom `output_store:` will aggregate the wrong file
through the pattern rule. `smoke.yaml` is the only config that sets one (to keep
random-weights rows out of the real store), and `make smoke` therefore does not go through
`exp-%` — it passes `--store` explicitly. Keep the two in sync if you change either.

For the container / prefetch / offline recipe, see [`SERVERLESS.md`](SERVERLESS.md).
