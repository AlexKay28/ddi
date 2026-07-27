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
