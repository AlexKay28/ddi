# ADR — DDI Experiment Architecture

**Status:** living document. Updated as the harness and experiment grid evolve.
**Scope:** the `ddi/` package, the `experiments/` YAMLs, the `nirvana/` pipeline,
and the `paper/` preprint scaffold. Everything an engineer needs to understand
*what is being measured, how, where the numbers live, and how to review them*.

---

## Table of contents

1. [DDI mask taxonomy](#1-ddi-mask-taxonomy)
2. [Claims C1 / C3 / C4](#2-claims-c1--c3--c4)
3. [Experiment grid — the 7 YAMLs](#3-experiment-grid--the-7-yamls)
4. [run_id schema — the store join key](#4-run_id-schema--the-store-join-key)
5. [Measurement paths](#5-measurement-paths)
6. [Parity / preservation framework](#6-parity--preservation-framework)
7. [Nirvana pipeline architecture](#7-nirvana-pipeline-architecture)
8. [Paper table / figure mapping](#8-paper-table--figure-mapping)
9. [Scope decisions](#9-scope-decisions)
10. [Open questions](#10-open-questions)
11. [Experiment result logging & review workflow](#11-experiment-result-logging--review-workflow)

---

## 1. DDI mask taxonomy

**Source of truth:** `ddi/masks.py::build_mask` — one function, five variants.

A RAG prompt is structured as:

```
[doc_1][doc_2] ... [doc_N][query][answer]
```

`build_mask(doc_lengths, query_len, answer_len, variant)` returns a boolean
`[T, T]` allow-matrix (True = attend) where `T = sum(doc_lengths) + query_len + answer_len`.
Conversion to HF's additive float format (`0 = keep, -inf = block`) is a separate
helper, `to_additive(allow, dtype)`.

| Variant | Semantics | Why it exists |
|---|---|---|
| `full` | Standard lower-triangular causal. Every token attends to all preceding tokens. | Baseline. C1 measures the gap between this and `ddi_qf`. |
| `ddi_qf` | Block-diagonal: each doc is causal within itself, blind to other docs. The query block sees all docs. The answer sees all docs + the query. | The headline DDI mask. "qf" = query-full: the query can read every document. |
| `ddi_strict` | Same as `ddi_qf` but only the *last* query row (the SEP token) bridges into the docs. | The ablation that makes `ddi_qf` meaningful. H2 found a 6x loss increase — query isolation is catastrophic. |
| `ddi_bridge_k` | Interpolates `ddi_strict` <-> `ddi_qf`: the last `bridge_k` query rows bridge into the docs. | Descoped from the preprint (#82 cut F2/F6). Still supported by the runner for ablation sweeps. |
| `blind` | Query block sees zero docs. Answer sees zero docs + the query. | The floor. `blind` measures what the model produces with no document access at all — the denominator of the parity framework's headroom `S`. |

**N=1 collapse:** when there is only one document, `ddi_qf` / `ddi_strict` /
`ddi_bridge_k` all degenerate to `full` — there is nothing to isolate. This is
the correctness gate checked at the mask level in `tests/test_gates.py`.

**Layouts** (`ddi/positions.py::build_positions`):

| Layout | Positions | Purpose |
|---|---|---|
| `sequential` | `[0, 1, 2, ..., T-1]` (HF default) | Baseline. |
| `restart` | Each doc's positions restart at 0 (`[0..L_i-1]`). Query+answer continues from `max(doc_lengths)`. | Makes each document's KV block a pure function of its content — the precondition for C4's per-document cache. |

---

## 2. Claims C1 / C3 / C4

**Source of truth:** `paper/RESEARCH_PLAN.md` \S4, `experiments/README.md`.

The preprint is scoped to three claims (per issue #82's scope cut):

### C1 — quality parity

**Claim:** Document-isolation (`ddi_qf`) costs little quality vs. `full` causal.

**Formal test:** `ddi_qf` gap to `full` < **\Delta** (the non-inferiority margin),
measured across >= 2 model families x >= 2 datasets.

**\Delta** (`ddi/delta.py`): `k * std` across per-seed means of the `full`
baseline, with `k=2` by default. Requires >= 3 seeds. Computed per
(model, dataset, metric) and written back to the store as `delta_{metric}` rows.
Ground rule 9: before any DDI number is called parity or non-parity, \Delta must
be fixed from the `full` baseline's seed variance alone.

**Metrics:** `loss` (masked cross-entropy on answer tokens), `em` (exact match),
`f1` (token F1), `presence` (answer string appears in generation).

**What would falsify:** `ddi_qf` gap > \Delta on any (model, dataset) pair where
the headroom `S = full - blind` is admissible (see \S6).

### C3 — speed and realization ratio

**Claim:** DDI reduces prefill cost, scaling with N (document count).

**Measurement:** Wall-clock TTFT (time-to-first-token) and TPOT (time-per-output-token)
across `(N, L, backend, mask)` grids, with `n_repeats=5` and `warmup=3`.

**The realization ratio** is the scientifically interesting part:
- `analytical_speedup` = theoretical sparsity-based speedup (e.g. ~10.6x at N=13, L=512)
- `realized_speedup` = `full_ttft / ddi_qf_ttft` (measured)
- `realization = realized / analytical`

H3 found a 9x gap between theory and measurement under HF-SDPA (0.094 realization
ratio). Whether this is structural (kernel launch overhead, no varlen path) or
fixable (FA2 varlen, Triton kernel, vLLM) is the load-bearing open question.

**What would falsify:** realized speedup <= 1.0 across the entire grid (DDI is
slower, not faster). The paper survives this — it becomes a negative result with
a mechanism explanation.

### C4 — caching and restart layout (co-headline)

**Claim:** Per-document KV blocks are reusable across requests and independent of
retrieval order.

**Three sub-experiments:**
1. **Correctness gate** (`cache_correctness.yaml`): cached logits match recomputed
   logits within `atol=1e-4`.
2. **Throughput** (`cache_throughput.yaml`): cached mode (`ddi_cache`) vs.
   recompute (`ddi_recompute`) vs. baseline (`full_causal`) — QPS under a
   document-reuse workload.
3. **Restart layout parity** (`cache_restart.yaml`): the `restart` position layout
   produces the same quality as `sequential`. This is the precondition — if restart
   breaks quality, the cache is useless.

**Why co-headline:** a per-document cache hit skips prefill entirely, which no
prefill-kernel improvement under `full` causal can match at any quality level.
C4 is the speedup that survives even if C3's realization gap is structural.

---

## 3. Experiment grid — the 7 YAMLs

**Source of truth:** `experiments/*.yaml`, `ddi/run.py` (dispatcher).

All YAMLs live in `experiments/`. None names a `model:` — `make exp-<name> MODEL=<hf-id>`
supplies it, so one file drives every model in the sweep. Dispatch is on `kind:`
via `_DISPATCH = {quality, timing, cache, finetune}` in `ddi/run.py`.

### YAML summary

| YAML | kind | claim | axes | cell count | body shape |
|---|---|---|---|---|---|
| `smoke.yaml` | quality | (CI check) | dataset=1, mask=3, records=4 | 12 | `{model: {mask: {dataset: value}}}` |
| `quality.yaml` | quality | C1 | dataset=3, mask=3, records=100 | 900 rows | `{model: {mask: {dataset: value}}}` |
| `performance.yaml` | timing | C3 | N=6, L=3, backend=2, mask=3, repeats=5 | 108 cells | `{model: {L: {backend: {mask: {N: value}}}}}` |
| `cache_correctness.yaml` | cache/correctness | C4 gate | mode=3, N=3, L=1, queries=5 | 45 cells | `{model: {mode: {L: {N: value}}}}` |
| `cache_throughput.yaml` | cache/throughput | C4 QPS | mode=3 (scalars, not lists) | 3 row-blocks | `{model: {mode: {arm: value}}}` |
| `cache_restart.yaml` | cache/restart_quality | C4 precondition | dataset=3, mask=1, layout=2, records=100 | 600 rows | `{model: {layout: {dataset: value}}}` |
| `finetune.yaml` | finetune | (#120) | dataset=3, epochs=1, lr=1e-5 | per-step rows | `{model: {tag: {step: value}}}` |

### Per-YAML detail

**`smoke.yaml`** — CPU-only end-to-end check. `datasets: [synthetic_qa]`,
`masks: [full, ddi_qf, blind]`, `max_records: 4`, `max_new_tokens: 8`,
`require_gpu: false`. Writes to `results/smoke_runs.jsonl` (not the real store).

**`quality.yaml`** — C1 measurement. `datasets: [hotpot_qa, musique, synthetic_qa]`,
`masks: [full, ddi_qf, blind]`, `max_records: 100`, `max_new_tokens: 64`.
Metrics: `em, f1, presence, loss`. Optional `checkpoint:` for local saved checkpoint.

**`performance.yaml`** — C3 timing grid. `Ns: [1, 3, 5, 9, 13, 25]`,
`Ls: [128, 512, 2048]`, `backends: [sdpa, fa2]`, `masks: [full, ddi_qf, blind]`,
`n_repeats: 5`, `warmup: 3`. Grid = 6 x 3 x 2 x 3 = 108 cells. Writes:
`ttft_median_ms`, `tpot_median_ms`, `mask_sparsity`, `analytical_speedup`,
`backend_used`. Q defaults to 64, n_new_tokens to 16.

**`cache_correctness.yaml`** — C4 correctness gate. `modes: [ddi_cache, ddi_recompute, full_causal]`,
`Ns: [3, 9, 25]`, `Ls: [512]`, `n_queries_per_doc: 5`, `atol_check: 1e-4` (advisory).
`full_causal` must be present as baseline.

**`cache_throughput.yaml`** — C4 throughput. `modes: [ddi_cache, ddi_recompute, full_causal]`.
**Known limitation:** reads scalars (`top_docs=32`, `docs_per_request=5`,
`doc_token_len=500`), not lists — a real sweep needs the run_id extended (see \S10).

**`cache_restart.yaml`** — C4 precondition. `layouts: [sequential, restart]`,
`datasets: [hotpot_qa, musique, synthetic_qa]`, `masks: [ddi_qf]`, `max_records: 100`.

**`finetune.yaml`** — Self-distillation finetuner (#120). Teacher = `full` mask
(frozen), Student = `ddi_qf` mask, loss = temperature-scaled KL on answer-position
logits. `epochs: 1`, `lr: 1e-5`, `kl_temperature: 1.0`, `grad_accum_steps: 4`,
`num_gpus: 1` (DDP via `torchrun`). `trainable_layers` for frozen-trunk (27B).

### `--set` overrides

Validated against `_SET_KEYS[kind]` — the exhaustive enumeration of what each
dispatcher actually reads. Unknown key raises. `--set-unchecked` is the deliberate
bypass for job-side code at a different revision.

### Per-cell isolation

One bad cell logs to stderr and the loop continues. `main()` exits non-zero if
any cell failed (`_CellTracker.failed_total`). `_exit(code)` uses `os._exit` to
skip NCCL/UCX teardown segfault.

---

## 4. run_id schema — the store join key

**Source of truth:** `ddi/runids.py` (one dataclass per prefix, positional parsing).

Every number in the store is keyed by a `run_id` string. The run_id is the join
key — `ddi.aggregate`, `ddi.delta`, and `ddi.parity` all parse it to group rows.
The shapes are **frozen** (old rows are real research data; a change that makes a
historical row unreadable is a regression).

| Prefix | Grammar | Parser | Seed |
|---|---|---|---|
| `phase2_train_free:` | `{model}:{dataset}:{variant}:seed={seed}:{sha8}:{record_id}` | `QualityRunId` | int (required) |
| `phase3_timing:` | `{model}:N={N}:L={L}:Q={Q}:{variant}:{backend}:seed={seed}:{sha8}` | `TimingRunId` | int \| None |
| `phase4_cache:` | `{model}:{mode}:N={N}:L={L}:seed={seed}:{sha8}:q{query_i}` | `CorrectnessRunId` | int \| None |
| `phase4_throughput:` | `{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha8}` | `ThroughputRunId` | int \| None |
| `phase4_restart:` | `{model}:{dataset}:{tag}:{layout}:seed={seed}:{sha8}:{record_id}` | `RestartRunId` | int (required) |
| `finetune:` | `{model}:{tag}:seed={seed}:{sha8}:step={n}` (or `:epoch={n}`) | `FinetuneRunId` | int (required) |
| `phase1_delta:` | `{model}:{dataset}:metric={metric}:k={k}[:seed={seed}]` | *(no parser — no consumer parses it back)* | — |

**Design rules:**
- Never raise — "this row is not mine" is the common case. Every `parse` returns `None`.
- Segment counts are minimums — record IDs and reuse tags are free-form tail segments.
- `seed` is `int | None` on timing/correctness/throughput (those parsers never looked at it historically); required on quality/restart/finetune.

**Known gap:** `ThroughputRunId` does not carry `top_docs` / `docs_per_request` /
`doc_token_len`, so two runs differing only in those merge into one aggregator cell.
Fixing this requires extending the run_id grammar + the producer in `run.py`.

---

## 5. Measurement paths

**Source of truth:** `ddi/run.py` (the dispatcher), `ddi/inference.py`,
`ddi/timing.py`, `ddi/cache.py`, `ddi/finetune.py`, `ddi/perplexity.py`.

### Quality cell (`_quality_cell`)

Grid: **(dataset x mask)**, one row per (record, metric).

1. **Loss arm:** encode docs + query + answer via `ddi.prompt.encode_segments` (the
   single shared encoder). Build mask, build positions, forward with
   `attention_mask=additive`, compute `shifted_cross_entropy` on answer tokens.
2. **Generation arm:** encode docs + query (no answer) via `inference.encode`,
   call `inference.generate` (greedy, mask threaded through every decode step).
   Score with `em`, `f1`, `presence`.

Run_id: `phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha8}:{record_id}`

### Timing cell (`_timing_cell`)

Grid: **(mask x N x L x backend)**. Fresh model load per backend (HF picks attn
kernel at `from_pretrained` time). Uses synthetic random ids (`torch.randint`).

Two independent measurements:
- **Prefill** (`measure_prefill`): single forward pass, `use_cache=False`.
- **Decode** (`measure_decode`): prefill with `use_cache=True`, then decode loop.

Writes: `ttft_median_ms`, `tpot_median_ms`, `mask_sparsity`, `analytical_speedup`,
`backend_used`. `n_repeats=5`, `warmup=3`.

Run_id: `phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant}:{backend}:seed={seed}:{sha8}`

### Cache correctness (`_cache_correctness`)

Grid: **(mode x N x L x query_i)**. `full_causal` must be in `modes` (baseline).
Compares each mode's logits against `full_causal` via `max_abs_logit_diff`.

Uses `DocCache` / `compute_doc_kv` / `compute_key` from `ddi.cache`. Cache entries
keyed by `sha256(model_id, dtype, doc_text)[:16]`.

Run_id: `phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha8}:q{query_i}`

### Cache throughput (`_cache_throughput`)

One row-block per `mode`. Reads scalars (not lists) — `top_docs`, `docs_per_request`,
`doc_token_len`. Helpers: `_sample_doc_indices` (zipf/uniform/fixed distribution),
`_time_doc_prefill`, `_time_query_prefill`, `_time_full_causal`, `_one_request`.

Writes: `qps`, `n_requests`, `total_elapsed_s`, `hit_rate`, per-request latency.

Run_id: `phase4_throughput:{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha8}`

### Cache restart quality (`_cache_restart_quality`)

Grid: **(dataset x mask x layout)**. Same loss/generation pattern as `_quality_cell`
but with `build_positions(doc_lengths, query_len, answer_len, layout)`.

Run_id: `phase4_restart:{model}:{dataset}:{tag}:{layout}:seed={seed}:{sha8}:{record_id}`

### Finetune (`run_finetune`)

Wraps `ddi.finetune.finetune` in a single cell. Teacher = `full` (frozen), Student
= `ddi_qf`, loss = temperature-scaled KL on answer-position logits. DDP via
`torchrun --nproc_per_node=N`.

Run_id: `finetune:{model}:{tag}:seed={seed}:{sha8}:step={n}` (or `:epoch={n}`)

---

## 6. Parity / preservation framework

**Source of truth:** `ddi/parity.py`, `REVIEW-VIBEQUEUE-2.md` \S1.

The fine-tune parity claim ("can a model be trained so `ddi_qf == full`?") is two
conditions, not one. Reporting only the first is how a broken model passes:

| | Condition | Catches |
|---|---|---|
| **Parity** | tuned model: `ddi_qf` ~= `full` | the mask costing quality |
| **Preservation** | tuned model under `full` ~= **base** model under `full` | a model that "achieved parity" by getting worse at everything |

### Formulas

All measured as a fraction of the headroom the documents are worth at all:

```
S = full_base - blind_base          # headroom (denominator)
P = (full_tuned  - ddi_qf_tuned) / S    # parity gap        -> 0
R = (full_base   - full_tuned)     / S    # preservation gap  -> 0
E = (full_base   - ddi_qf_tuned)  / S    # end-to-end gap    -> 0
L = (blind_tuned - blind_base)    / S    # leakage           -> 0
```

`E == P + R` identically, so a failure is always attributable to one half.
As ratios: `closure = 1 - P`, `preservation = 1 - R`, both targeting 1.0.

Metric orientation: `em`/`f1`/`presence` are higher-is-better; `loss` is negated.

### Admissibility gate

**More important than the ratios.** When `S` (headroom) is at the noise floor,
every ratio above is noise/noise and swings wildly while looking like signal.

The gate: `S > max(min_span, k * Delta)` where `k=3.0` and `min_span=0.05`.
A cell that fails the gate is reported as `INADMISSIBLE` — no parity claim can
be made from it in either direction.

`Delta` comes from `ddi.delta.compute_delta` (>= 3 seeds on the base model's
`full` rows). Without it, the CLI refuses to print verdicts unless
`--allow-missing-delta` is passed.

### Verdict strings

| Verdict | Meaning |
|---|---|
| `SUCCESS: parity gap within tolerance, model intact` | Both gates pass. |
| `SUCCESS: parity gap within tolerance, full improved too` | Both gates pass; finetune improved `full` as well. |
| `PARITY BY REGRESSION: full degraded — not bankable` | Parity gap OK but `full` got worse. |
| `PARITY NOT REACHED: model intact, mask still costs quality` | Model intact but `ddi_qf` still differs from `full`. |
| `HARMFUL: full degraded and the mask still costs quality` | Both halves failed. |
| `INVALID (leakage ...)` | `blind` moved — closure is parametric, not mask-related. |
| `INADMISSIBLE: headroom <= floor` | Documents carry no measurable signal; parity undefined. |

---

## 7. Nirvana pipeline architecture

**Source of truth:** `nirvana/graphs/ddi_rows.py` + `nirvana/graphs/ddi_pipeline.py`.

The Nirvana pipeline runs the GPU fleet campaign. It produces 35 publishable entry
points across 3 base models (4B Qwen3.5-4B-Instruct, 7B Qwen2.5-7B-Instruct, 27B
Qwen3.6-27B-Instruct), plus 3 diagnostic graphs. All publish comments to ticket
`VIBEQUEUE-2`.

### Two pipeline shapes

`build_ddi_pipeline(...)` is the one builder. Shape is picked on one argument:

- **eval-only** — pass `eval_models` -> one 1-GPU eval per listed model. Most of
  the grid. Two+ models rendered side-by-side with `ddi_qf - full` gap subtracted.
- **finetune** — pass `seeds` + reserved `ft_suffix` -> one 8-GPU finetune + one
  1-GPU eval of that checkpoint. 8 of the 35 are this shape (the GPU cost).

### Chain

```
arc_export + zeloboba_arc_checkout_deterministic
  -> merge_tar_archives
  -> (one finetune+eval cell per seed, OR one eval per model)
  -> _collect
  -> Make DataFrame report (Python + pandas, runs _REPORT_TEMPLATE)
  -> add_comment_to_ticket
```

### Checkpoint collision guard

Two halves: `reserve_checkpoint(suffix, *models, owner=)` runs at module import;
`_claim_checkpoint` runs in the builder on the derived `(model, suffix)`.
Forgetting to reserve -> failed `nv-check`; reserving another row's held path ->
import-time raise; protected pre-review suffixes refused on the 4B.

### Current limitation

`_REPORT_TEMPLATE` is hardcoded to `kind == "quality"` and the four quality
metrics (`em/f1/presence/loss`). It cannot render timing, cache, throughput, or
restart tables. Extending it is issue #73 (template-shaped renderers).

---

## 8. Paper table / figure mapping

**Source of truth:** `paper/sections/*.tex`, `paper/generated/`.

| Artifact | Description | Populated by | Status |
|---|---|---|---|
| T1 | Mask taxonomy | `ddi/masks.py` (static) | `paper/generated/T1_mask_taxonomy.md` exists |
| T2 | Quality parity table (C1) | `quality.yaml` -> `ddi.aggregate` -> `kind: quality` | Stub (`\todoreport`) |
| T3 | Timing grid (C3) | `performance.yaml` -> `ddi.aggregate` -> `kind: timing` | Stub |
| F3 | Realization ratio figure (C3) | `performance.yaml` -> derived from T3 data | Stub |
| F5 | Throughput vs. reuse rate (C4) | `cache_throughput.yaml` -> `kind: throughput` | Stub |

**Render flow:** `make paper` runs `ddi.aggregate` on the store, produces JSON
summaries in `paper/generated/`, which the LaTeX `\todoreport` macros consume.
Currently all T2/T3/F3/F5 are stubs because `results/runs.jsonl` lives only on
GPU job hosts (gitignored).

### Sections

| Section | File | Content |
|---|---|---|
| 00 | `00_intro.tex` | Contribution list, scope statement |
| 01 | `01_related_work.tex` | FiD, PCW, CEPE, Longformer, PagedAttention |
| 02 | `02_method.tex` | Mask taxonomy, position layouts, \Delta |
| 03 | `03_experimental_design.tex` | Datasets, models, axes, ground rules |
| 04 | `04_results_quality.tex` | T2 (train-free quality sweep) |
| 05 | `05_results_efficiency.tex` | T3, F3 (timing, realization ratio) — has FA2 `\todo` |
| 06 | `06_results_caching.tex` | F5 (throughput under reuse) |
| 07 | *(deleted by #82)* | Was: failure envelope (C5/C6/PCW) |
| 08 | `08_limitations_conclusion.tex` | Limitations, preprint scope, conclusion |

---

## 9. Scope decisions

### What #82 cut from the preprint

- **C5 (failure envelope)** — single-hop vs multi-hop split. Section 07 deleted.
- **C6 (distractor robustness)** — Lost-in-the-Middle-style evaluation.
- **PCW (Parallel Context Windows)** — now cited-only, not a baseline.
- **Composition dataset** — MoNaCo / 2WikiMultiHop / MuSiQue for cross-document stress.
- **`ddi_bridge_k`** — bridge-width sweep (F2/F6) dropped from the paper text.
- **Fine-tuning** — cut from the *preprint text* (T4 mechanism probe, fine-tuned
  columns in T2). **However**, the code/finetune.yaml/Nirvana finetune rows still
  exist and run — the README says "Fine-tuning does" and `finetune.yaml` is "the
  bulk of what the Nirvana fleet is currently running". Whether fine-tuning
  re-enters the preprint is an open scoping decision.

### What was re-included

- **Fine-tuning** (at the experiment layer, not yet the paper layer) — the
  self-distillation finetuner (#120) and 8 Nirvana finetune rows are active.
- **C4 as co-headline** — originally planned as an appendix; promoted because a
  per-document cache hit skips prefill entirely, which no kernel improvement
  under `full` causal can match.

### Models

| Model | Role | Preprint? |
|---|---|---|
| Qwen2.5-7B-Instruct | Primary | Yes |
| Llama-3.1-8B-Instruct | Primary (independent family) | Yes |
| Qwen3.5-4B-Instruct | Fleet (Nirvana) | Yes (if numbers land) |
| Qwen3.6-27B-Instruct | Scale trend (frozen trunk) | Appendix |

---

## 10. Open questions

### FA2 realization gap (load-bearing)

`05_results_efficiency.tex` has a `\todo`: the FA2 half of the timing grid is
deferred. If FA2 varlen also shows near-unity realization across N, the
prefill-only compute claim is null on tested backends and the caching path (C4)
is the sole efficiency win.

**Resolution:** wire FlashAttention-2 varlen through DDI's mask family and
re-run `performance.yaml` with `backends: [sdpa, fa2]`. The `fa2` path exists in
`ddi/attention.py` but the varlen block-diagonal path is not yet wired.

### vLLM feasibility (issues #65-69)

vLLM does not natively support arbitrary 4D attention masks. Its attention path
is optimized around causal-with-optional-prefix-caching. Getting DDI's mask into
vLLM's forward is real integration work, not a config change.

**Phased plan:** A (design + prototype) -> B (correctness gate) -> C (C3 re-eval)
-> D (C4 re-eval). Decision gate after B: if green, proceed; if not, accept
HF-only caveat in paper.

**Depends on #122** (inference surface isolation) — vLLM's shim needs a single
injection point, not 13 scattered `model(...)` calls.

### Fine-tuning in the preprint

The README says: "Fine-tuning does [run]. Whether it belongs in the preprint is
a scoping decision that has not been revisited." The parity framework
(`ddi/parity.py`) exists precisely to answer this — but it needs the Nirvana
fleet results to populate it.

### `cache_throughput.yaml` sweep gap

`docs_per_request` / `doc_token_len` / `top_docs` are not in the `ThroughputRunId`,
so two runs differing only in those merge into one cell. A real C4 sweep needs
the run_id grammar extended and the YAML keys made list-typed.

---

## 11. Experiment result logging & review workflow

This section describes how results flow from GPU job to paper table, and how to
review them at each stage.

### 11.1 The data pipeline

```
GPU job (Nirvana or local)
  |
  |  ddi.run writes one JSONL row per (run_id, metric, value)
  |  via ddi.store.append(run_id, metric, value, provenance, store)
  |
  v
results/runs.jsonl  (append-only JSONL, gitignored, lives on job hosts)
  |
  |  ddi.aggregate reads the store, groups by run_id, produces JSON summaries
  |  python -m ddi.aggregate <exp.yaml> --out <summary.json> [--model REPO_ID]
  |
  v
logs/<name>_summary.json  (JSON, one object per report.tables[] entry)
  |
  |  make paper  ->  paper/generated/T*.md, F*.md  (rendered from summaries)
  |
  v
paper/sections/*.tex  (LaTeX consumes paper/generated/ via \todoreport macros)
```

### 11.2 The store format

**File:** `results/runs.jsonl` (append-only, `fcntl.flock`-guarded for concurrent
appends by parallel sweep workers).

**One row per `(run_id, metric, value)`:**

```json
{
  "run_id": "phase2_train_free:Qwen/Qwen2.5-7B-Instruct:hotpot_qa:ddi_qf:seed=0:abcd1234:rec_001",
  "metric": "em",
  "value": 0.52,
  "provenance": {
    "git_sha": "157e3bf...",
    "dirty": false,
    "config_hash": "a1b2c3d4",
    "seed": 0,
    "model_revision": "Qwen/Qwen2.5-7B-Instruct",
    "dataset_revision": "hotpot_qa",
    "gpu_model": "NVIDIA L40S",
    "driver_version": "550.54.14",
    "cuda_version": "12.4",
    "torch_version": "2.4.0",
    "timestamp_utc": "2026-08-04T12:00:00+00:00"
  }
}
```

**Ground rules** (from issue #1):
1. Every metric row carries full provenance (git SHA + dirty flag, config hash,
   seed, model revision, dataset revision, GPU model, driver/CUDA/torch versions,
   UTC timestamp).
2. The store is append-only and committed (ground rule 2).
3. No number appears in a generated table or figure that is not in this store
   (ground rule 3).
4. No DDI number enters the store before the correctness gates are green (ground
   rule 6).

### 11.3 Provenance capture

**Source:** `ddi/provenance.py::capture(config_hash, seed, model_revision, dataset_revision)`

Captures 11 fields automatically on every `store.append` call:
- `git_sha` + `dirty` — falls back to `"not-a-git-repo"` on serverless containers
  with no `.git` dir
- `config_hash` — SHA256 of the YAML bytes (not the path; two copies join, an
  edit doesn't)
- `seed`, `model_revision`, `dataset_revision` — passed by the caller
- `gpu_model` — `torch.cuda.get_device_name(0)` or `"cpu"`
- `driver_version` — from `nvidia-smi` or `"unknown"`
- `cuda_version`, `torch_version` — from torch
- `timestamp_utc` — ISO 8601

### 11.4 Aggregation

**CLI:** `python -m ddi.aggregate <exp.yaml> --out <summary.json> [--store results/runs.jsonl] [--model REPO_ID]`

Reads the `report.tables:` block from the experiment YAML and emits one JSON
object per table entry, wrapped in an envelope carrying the store hash, git SHA,
and model filter.

Dispatch on `kind`:

| kind | Rows | Body shape | Reduction |
|---|---|---|---|
| `quality` | `phase2_train_free:` | `{model: {mask: {dataset: value}}}` | mean |
| `timing` | `phase3_timing:` | `{model: {L: {backend: {mask: {N: value}}}}}` | median |
| `throughput` | `phase4_throughput:` | `{model: {mode: {arm: value}}}` | mean |
| `restart_quality` | `phase4_restart:` | `{model: {layout: {dataset: value}}}` | mean |
| `correctness` | `phase4_cache:` | `{model: {mode: {L: {N: value}}}}` | max (worst case) |

`timing` also derives `realized` and `realization` by pairing variant rows with
matching `full` rows at `(model, N, L, backend, seed)`.

### 11.5 Reviewing results post-run

After a GPU job completes, review results at three levels:

#### Level 1 — Nirvana ticket comments (immediate, per-row)

Each Nirvana entry point publishes a comment to ticket `VIBEQUEUE-2` with a
markdown table rendered by `_REPORT_TEMPLATE`. This is the first thing to check:
did the job produce quality numbers, or did it crash? The comment includes
`{model: {mask: {dataset: value}}}` tables for quality runs, and the
`ddi_qf - full` gap for side-by-side model comparisons.

#### Level 2 — Aggregate JSON (local, per-experiment)

Copy `results/runs.jsonl` from the job host, then:

```bash
python -m ddi.aggregate experiments/quality.yaml --out logs/quality_summary.json
python -m ddi.aggregate experiments/performance.yaml --out logs/performance_summary.json
```

Inspect the JSON: are there missing cells? Are `ddi_qf` values in the expected
range relative to `full` and `blind`?

#### Level 3 — Parity / delta CLI (analytical, per-claim)

**Compute \Delta** (the noise floor):
```bash
python -m ddi.delta_cli --store results/runs.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --dataset hotpot_qa --metric em
```

**Run the parity gate** (requires base + tuned model rows + \Delta):
```bash
python -m ddi.parity --store results/runs.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --tuned-model Qwen/Qwen2.5-7B-Instruct__ddi_qf__all4 \
  --delta-store results/runs.jsonl
```

This prints a markdown table with `S`, `closure`, `preservation`, `E`, `leak`,
and the verdict for each (dataset, metric) cell. Inadmissible cells are listed
at the bottom.

### 11.6 Data retention

| Directory | Contents | Gitignored? | Retention |
|---|---|---|---|
| `results/runs.jsonl` | The store (all metric rows) | Yes | Lives on job hosts; copy locally for review |
| `results/checkpoints/` | Fine-tuned model checkpoints | Yes | Per-run; referenced by `finetune.yaml` `output_dir` |
| `results/splits/` | Dataset record caches (~2.5GB) | Yes | Reusable across runs; speeds up dataset loading |
| `logs/` | Run logs, summary JSONs, `finetune_summary.json` | No (committed) | Historical record; `logs/qwen_instruct_c1_summary.md` is stale (pre-review) |
| `paper/generated/` | Rendered T*/F* artifacts | No (committed) | Regenerated by `make paper` |

### 11.7 Review checklist: raw rows -> paper table

Follow this checklist every time a new batch of results lands:

1. **Copy the store:** `scp` or `rsync` `results/runs.jsonl` from the job host.

2. **Validate provenance:** spot-check a few rows. Confirm `git_sha` matches the
   expected commit, `dirty` is `false`, `gpu_model` is the expected GPU, and
   `config_hash` matches the YAML you intended to run.

3. **Check the correctness gates:** run `pytest tests/test_gates.py` — these are
   the Phase 0 correctness backstop (SDPA cross-check, N=1 equivalence). If any
   gate fails, the numbers are void.

4. **Compute \Delta:** for each (model, dataset, metric) in the quality grid, run
   `ddi.delta_cli`. Confirm >= 3 seeds were used. If \Delta can't be measured,
   the parity claim is undefined for that cell.

5. **Run the parity gate:** if fine-tuned results are present, run `ddi.parity`
   with `--base-model` and `--tuned-model`. Check for `INADMISSIBLE` cells and
   `INVALID (leakage)` verdicts before reading any `SUCCESS`.

6. **Aggregate:** run `ddi.aggregate` for each YAML that has results. Inspect the
   JSON for missing cells, unexpected values, or aggregation artifacts.

7. **Render:** run `make paper` to populate `paper/generated/`. Check that the
   `\todoreport` stubs in the LaTeX are replaced by actual numbers.

8. **Paste into LaTeX:** copy the rendered markdown/JSON values into the
   corresponding `paper/sections/*.tex` tables. The paper's tables are
   hand-maintained — the generated artifacts are the source of truth, the `.tex`
   is the presentation.

9. **Sanity-check against hypotheses:** compare against H1/H2/H3 from
   `RESEARCH_PLAN.md \S3`. If `ddi_qf` loss is far from `full`, or `ddi_strict`
   is not catastrophic, or the realization ratio is not ~0.094, investigate
   before trusting the numbers.

10. **Update CHANGELOG:** add an entry under `## Unreleased` (or the appropriate
    dated section) describing what was run, on which model, and the headline
    finding.

### 11.8 The store as single source of truth

**Ground rule 3:** no number appears in a generated table or figure that is not
in the store. The store (`results/runs.jsonl`) is the single source of truth for
all paper numbers. If a number is in the paper but not in the store, it's a bug.
If a number is in the store but not in the paper, it's either not yet rendered
or descoped.

The store is append-only — rows are never deleted or modified. If a run is
invalid (e.g., a bug was found post-run), the invalid rows stay in the store
and the aggregator is configured to filter them out (by `git_sha`, `config_hash`,
or `run_id` prefix). This preserves the audit trail.
