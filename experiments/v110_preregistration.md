# V110 Preregistration — D-Arm Decomposition on the Trained k5 Checkpoint

**Frozen:** 2026-09-15, before any evaluation run. **Ticket:** VIBEQUEUE-110.
**Motivation:** VIBEQUEUE-90 — V70 arm D differs from B/C in mask (full-causal),
position treatment (restart retained), and link layout (k5 markers retained)
simultaneously; the D degradation cannot be attributed to isolation training
without separating these effects.

## Question

Which component of the D arm's execution — the full-causal mask itself, the
restart/link layout retention, or the trained weights — drives its collapse
(pinned D EM 0.183333 / F1 0.281447 on confirmation300)?

## Arms (all on the trained k5 checkpoint `results/v70_arms_valid_seed47/checkpoint_k5`)

- **k5-d-replica:** the pinned V70 D construction — harness
  `generate_monolithic_ids(full_causal=True)`: full-causal mask swapped in,
  restart positions and k5 marker layout retained. Replication anchor; must
  reproduce the pinned D quality (any drift is a deviation, not a finding).
- **k5-std:** standard execution of the trained weights — plain token stream
  (documents + chat-templated query, no markers, no BOS), sequential
  positions, full-causal mask (`variant='full'`, prefill and every decode
  step). Same construction as the V109 R arm, different weights.
- **k5-iso-seq:** isolated mask without the restart/link machinery — plain
  token stream, sequential positions, `variant='ddi_qf'` (documents
  self-attending, query sees documents, answer rows see all). Isolates the
  mask effect from positions/links.

## Data, metrics, analysis (frozen)

Identical to V109: frozen confirmation300 IDs in manifest order, gold =
`possible_answers`, EM/F1/presence via `ddi.metrics`, greedy 64, EOS stop,
bf16/eager/batch-1. Paired per-record deltas against the pinned C and B arms
of `results/v70_confirmation300_k5.json` (10,000 resamples, seed 47).
Diagnostic; no gates.

## Interpretation contract (frozen)

- If k5-std ≈ pinned C/B while k5-d-replica ≈ pinned D: the collapse is driven
  by the layout retention (restart+links under a full-causal mask), not by
  full-causal execution or the mask alone.
- If k5-iso-seq ≈ pinned C/B: the isolated mask alone is harmless on trained
  weights.
- If k5-std ≈ pinned D: full-causal execution itself is out-of-distribution
  for the trained weights regardless of layout.
- These are characterizations, not gates; the paper may only restate the D-arm
  paragraph in terms of the measured decomposition.

## Implementation pins

- Runner: `scripts/run_v109_v110_references.py` with arms `k5-d-replica`,
  `k5-std`, `k5-iso-seq`.
- Checkpoint aggregate SHA-256 (sorted per-file hashes) recorded in the result.
- Smoke on 2 train-split records precedes the real run; smoke results are not
  part of the analysis.

## Deviations

Any deviation must be amended here with a UTC timestamp before interpretation.

## Amendment 1 (2026-09-15T22:00Z, after the run)

The k5-d-replica replication anchor drifted from the pinned D arm: EM 0.1967
vs pinned 0.183333, F1 0.2865 vs pinned 0.281447 (4 records differ by EM).
Same checkpoint directory, same harness path, same frozen records; the runtime
changed since the V70 run (torch 2.13.0+cu126 on 2026-09-15 vs the V70 stack),
so eager bf16 kernels are not bitwise-identical across stacks. Recorded as a
deviation per this prereg; the anchor still reproduces the collapse (direction
and magnitude), and no conclusion depends on the pinned-vs-replica delta.
