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

## 2026-07-27 — CHANGELOG created

Repo scaffolded for the clean-room rebuild. Awaiting Phase 0.
