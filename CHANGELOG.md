# Changelog

Chronological log of experiments and project decisions. **Newest first.**

Numbers here are pointers, not sources of truth. Every experiment traces to a row in
`results/runs.jsonl` (ground rule 2 in [issue #1](https://github.com/AlexKay28/ddi/issues/1));
tables and figures regenerate from that store (ground rule 3). This file is the narrative layer
on top — what was run, why, and what it told us.

---

## Entry format

**Experiment entries** — one per run or sweep:

- **Date + title** — `## YYYY-MM-DD — [phase/sub-phase] short title`
- **Sub-issue** — link (e.g., `#13` for Phase 2.1)
- **SHA** — `abcdef1` (mark dirty if applicable)
- **Config** — `configs/<file>.yaml`
- **Runs** — `run_id` list, or count + query template
- **Finding** — one or two sentences; link the generated figure / table
- **`Δ` check** — if a quality claim: gap to `full` vs. `Δ` (green / red / n/a)
- **Follow-up** — issues opened, next sweep, or `closes #N`

**Non-experiment entries** — infrastructure, decisions, rebuilds — use the same date header
without the sub-issue / SHA / runs fields.

---

## Unreleased

Phase 0 in progress. No experiment entries until the three correctness gates
([issue #4](https://github.com/AlexKay28/ddi/issues/4)) are green — per ground rule 6, no DDI
number enters the store before that.

---

## 2026-07-27 — **Phase 1 complete**

All four Phase 1 sub-issues closed: #9 datasets, #11 metrics, #10 hop labels, #12 baseline
runner + Δ. The harness can now: load a frozen dataset with verifiable split hashes, tokenize
and score model output against SQuAD-style metrics, and run a `phase1_full_baseline` that writes
per-record loss + EM/F1/presence to `results/runs.jsonl` under a `full` mask routed through the
same `build_mask → to_additive → forward_with_mask` path every DDI variant will use (no
baseline-vs-variant implementation drift). `compute_delta` reads those rows and returns
`Δ = k × std_across_seeds(per_seed_mean(metric))` — this is what makes ground rule 9 real. Ready
for Phase 2 (issues #13 train-free primary, #14 conditional finetuned, #15 bridge sweep
centerpiece, #16 base-vs-instruct probe).

## 2026-07-27 — [1.4] baseline runner + Δ

`ddi.runners.phase1_baseline`: registers `task=phase1_full_baseline`. For each record: builds a
`variant="full"` mask via `build_mask` + `to_additive`, runs it through `forward_with_mask` for
answer-only cross-entropy (next-token-shifted), then `model.generate` (greedy) for EM/F1/presence.
Every metric writes a row to `results/runs.jsonl` with full provenance. Skips records with empty
`possible_answers` (loud print). The baseline shares the mask-injection path with future DDI
variants, so any drift between "baseline" and "DDI" numbers is a code bug, not an implementation
difference. `ddi.delta`: `compute_delta` filters by `phase1_baseline:{model}:{dataset}:` +
metric, groups by `provenance.seed`, computes sample std (ddof=1), returns `Δ = k × std`
(default `k=2`, requires ≥3 seeds). `record_delta` writes per-seed-mean rows AND a `delta_{metric}`
row so the whole computation is auditable per ground rule 3. `ddi-delta` CLI wraps it. Makefile
gains `phase1-baseline` and `delta` targets. Closes #12.

## 2026-07-27 — [1.2] hop labels + ddi-audit CLI

`ddi.hop_labels`: `hop_type(record)` reads the native field; `register_heuristic(dataset)` lets
later loaders (HotpotQA, Open RAG Bench) plug in a labeler for `hop_type == "unknown"` records
without touching the loaders themselves; `apply_heuristic` yields shallow-copied dicts so
callers can iterate loaders directly. `sample_for_audit` + `load_audit` + `agreement_rate` wire
the hand-audit workflow. `ddi-audit sample` writes a JSONL for review; `ddi-audit score`
re-samples, applies the heuristic, loads human labels, prints the rate + disagreements +
skipped count. Closes #10.

## 2026-07-27 — [1.1] dataset framework + NQ + synthetic

`ddi.data`: registry-based loaders yielding a common record shape (`info_contexts`, `dialog`,
`possible_answers`, `hop_type`, `gold_position`, `meta`). `splits.py` provides deterministic
`split_ids` + order-invariant `hash_split`; `manifest.py` reads/writes/validates a JSON manifest
at `results/splits/<name>.json` per ground rule 1 — mismatch raises `RuntimeError` with a diff.
Two initial loaders: `synthetic_qa` (deterministic, offline, 100 records / 60·20·20 split) for
tests and smoke runs, and `nq` (via `google-research-datasets/nq_open`) as the first real
dataset. `pyproject.toml` gains `datasets>=2.14`. Closes #9.

## 2026-07-27 — [1.3] metrics

`ddi.metrics`: SQuAD-style `normalize` (lowercase → punctuation strip → whole-token article
removal to avoid eating `"the"` inside `"theatre"` → whitespace collapse), then `exact_match`,
`token_f1` (Counter-multiset overlap, canonical SQuAD formula, max across candidates), and
`answer_presence` (refuses trivial empty-candidate hits). `make_answer_labels` returns a
`[batch, seq]` long tensor that is `input_ids` inside `[answer_start:answer_end)` and `-100`
elsewhere — feeds HF's own answer-only cross-entropy path. Closes #11.

## 2026-07-27 — **Phase 0 complete**

All seven Phase 0 sub-issues closed: #2 skeleton, #3 masks, #6 attention backends, #5 HF model
wrapper, #4 correctness gates, #7 provenance + store, #8 report generation. The DDI harness now
has a single mask definition (ground rule 8), three attention backends verified equivalent by the
Phase 0.5 gates (ground rule 6), mask injection into HF's own forward (no hand-rolled decoder),
run identity + append-only store (ground rules 1, 2), and idempotent report rendering from the
store (ground rule 3). Ready for Phase 1 (datasets + `full` baseline + fix `Δ`).

## 2026-07-27 — [0.5] correctness gates

`tests/test_gates.py`: three parametrized gates, all `@pytest.mark.gate`. Gate 1 — N=1 mask
equivalence + output equivalence over available backends. Gate 2 — SDPA-referenced backend
cross-check on doc-only fixed input; skips (never passes vacuously) when only SDPA is available.
Gate 3 — mask shape sanity plus the `len()`-vs-`.shape[-1]` regression encoded permanently. Meta
test verifies every `test_gate*` carries the marker. `gate` marker registered in `pyproject.toml`.
`make gates` runs just the gate suite. Closes #4.

## 2026-07-27 — Makefile as experiment shortcuts

Reframed root `Makefile` around experiment shortcuts. Setup collapsed to two targets (`setup`,
`setup-gpu`); every phase gets its own labeled targets (`sweep-train-free`, `sweep-bridge`,
`timing-N`, `realization`, `cache-throughput`, `pcw-baseline`, ...), each pointing at a config
under `ddi/configs/phase*.yaml`. `help` output is sectioned. `run-seeds` added for one-VM seed
loops. `report` target wraps `ddi.report_cli` (new module). Two config stubs seeded
(`phase2_train_free.yaml`, `phase2_bridge.yaml`) to demonstrate the shape.

## 2026-07-27 — [0.4] HF model wrapper with mask-preservation patch

`ddi.model.load_model` returns `(model, tokenizer)` and applies `patch_model` which installs an
instance-level `_update_causal_mask` (via `types.MethodType` on `model` or `model.model`) that
returns a supplied **4D float** mask unchanged and delegates every other shape/dtype to the
original. `unpatch_model` deletes the instance attribute to unshadow the class descriptor cleanly.
`forward_with_mask` validates 4D + floating-point and calls HF's own forward with `use_cache=False`.
No hand-rolled decoder loop. Integration test gated on `DDI_INTEGRATION=1`; unit tests use a mock.
Closes #5.

## 2026-07-27 — Makefile as serverless entry point + CI

`Makefile` reworked to use `uv` throughout (`uv pip install --system`, `uv run <cmd>`) and
reframed as the single interface for serverless experiment runs: fresh container → `make setup`
→ `make <phase-target>`. GitHub Actions workflow at `.github/workflows/test.yml` installs via
`make setup`, runs `make lint test` on Python 3.11 and 3.12, matches the serverless first-run
sequence so CI green ≡ the container will boot cleanly.

## 2026-07-27 — [0.3] attention backends

`ddi.attention.get_backend` dispatches to three implementations of one mathematical object: SDPA
with explicit additive mask (**reference** — defines correct), FlashAttention-2 varlen scoped to
doc-only prefill (uses `cu_seqlens`, CUDA-only, lazy-imported), and Triton deferred as
`NotImplementedError`. Cross-backend equivalence is verified by the Phase 0.5 gate. Closes #6.

## 2026-07-27 — [0.7] report generation

`ddi.report` provides the `@register` decorator (mirrors `ddi.cli.register_runner`), sorted
`list_reports`, and `render(name, store, output=None)` dispatching tables (markdown) vs figures
(write to output) by `T*` / `F*` naming. Store reader `iter_rows` in file order plus
`filter_rows` (metric / run_id / provenance subset). Helpers: `median_with_iqr` with linear
interpolation, `group_by` with dotted-path lookup. Idempotent by construction. Renderers for
T2/T3/F2–F6 register from later-phase modules as their data lands. Closes #8.

## 2026-07-27 — [0.2] masks.py

`ddi.masks.build_mask` returns a boolean `[T, T]` allow-matrix for all five variants (`full`,
`ddi_qf`, `ddi_strict`, `ddi_bridge_k`, `blind`) via vectorised boolean-tensor ops over row/col
index grids. `to_additive` converts to `-inf` form preserving bf16 representability. N=1 collapse
to `full` for the qf/strict/bridge_k family is explicit and matches the correctness-gate semantics
in issue #1 §0.5. Independent triple-loop reference in `tests/test_masks.py`. Closes #3.

## 2026-07-27 — [0.6] provenance and append-only store

`ddi.provenance.capture` returns a `Provenance` (git SHA/dirty, config hash, seed, model/dataset
revisions, GPU model, driver/CUDA/torch versions, UTC timestamp). `ddi.store.append` writes one
`fcntl.flock`-protected JSONL row per metric. Two-process concurrent-write test verifies parallel
sweeps do not interleave. Closes #7.

## 2026-07-27 — Makefile added

Root `Makefile` collects install / test / lint / format / run / clean targets and pins overridable
variables (`PYTHON`, `CONFIG`, `SEED`). Placeholder comments mark where phase-specific sweep
targets will land as their sub-issues close.

## 2026-07-27 — [0.1] package skeleton

`pyproject.toml`, `ddi/` package with stub modules (`masks`, `attention`, `model`, `provenance`,
`store`, `report`), working `ddi.cli` with runner registry, `ddi/configs/example.yaml`, `tests/`
with CLI smoke tests, `results/` directory. Closes #2.

## 2026-07-27 — CHANGELOG created

Repo scaffolded for the clean-room rebuild. Awaiting Phase 0.
