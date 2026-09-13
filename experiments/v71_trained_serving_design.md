# V71 Design Draft — Trained-Contract Serving Timing

**Status:** design draft. This is NOT the preregistration. The frozen protocol
(`experiments/v71_preregistration.{md,json}`, with SHA pins and amendment log)
must exist before any timed measurement, per house rule.

**Ticket:** VIBEQUEUE-71. **Predecessors:** V65 (audited warm-path win),
V66 (serving boundary), V70 (trained contract, exact-zero quality gate).

## Motivation

The paper currently assembles its joint claim from two different contracts:

- Timing (2.05× TTFT all-hit, V66 crossovers) was measured under the
  **untrained** canonical contract (V65/V66 checkpoint).
- Quality (cache byte-identical to fresh, C−B CI [0,0], n=300) was measured
  under the **trained** k=5 link-token contract (V70 checkpoint_k5).

No measurement shows the trained contract's warm path still beats the
monolithic baseline. The V70 contract also changes serving costs in two
known directions: +5N link tokens of prefill, and trained (vs canonical)
document encoding. V71 closes that cell or records an honest regression.

## Analytical expectation (to be superseded by measurement)

Link tokens add `5N` prefill tokens against a total of `Σ L_i + Q`. At the
favorable V66 shapes this is negligible: L=512, N=10 → 50/5168 ≈ 1.0% prefill
growth. At the unfavorable short shapes it is not: L=32, N=10 → 50/384 ≈ 13%,
in a regime where reuse already never wins. Prediction: crossovers hold or
shift negligibly right at L=512, and the L=32 verdicts do not change (they
are already losses). This is a hypothesis, not a gate.

## Arms

| Arm | Checkpoint | Execution |
|---|---|---|
| A | original monolithic full-causal baseline (unchanged from V65/V66) | fresh prefill+decode per request |
| B′ | V70 `checkpoint_k5` | trained-contract fresh: independent bf16 doc encoding + k=5 fixed markers, no persistence |
| C′ | V70 `checkpoint_k5` | trained-contract cached: true hit-only reads from a reopened persisted store; fail closed on encode attempt |

A stays the untrained baseline for crossover-shift comparability with V66.
B′ exists to price the contract's own compute overhead (descriptive B′ vs
untrained canonical fresh) and as the byte-identity reference for C′.

## Grid and timing scopes

Reuse the frozen V66 serving grid unchanged: frozen V62 development split,
L ∈ {32,128,512} × N × nominal hit fractions × decode budgets, seed 47,
15 timed repeat pairs per cell, CUDA-event prefill timing plus synchronized
wall scopes as in V66 (TTFT, first-generation-token wall, full 64-token wall).
No cell definition may change between the preregistration freeze and the run.

## Gates

- **Primary (per cell):** C′ vs A verdicts under the identical frozen
  V66 materiality rule (CI-supported win). Deliverable is the crossover
  table and its diff against V66's — "the boundary moves from h≥x to h≥y at
  shape Z" is the headline either way.
- **Secondary sanity (per cell subset):** C′ vs B′ generated IDs byte-identical.
  Expected exact zero per V70. A failure here means the trained contract
  breaks under serving-harness assembly (different code path than the V70
  eval harness) and is itself a reportable finding that blocks primary-gate
  interpretation.
- **Descriptive:** B′ vs untrained canonical fresh timing (contract overhead);
  C′ cold-encode cost under the trained contract, reported separately as in
  V65.

## Stop rules and budget

Single A100. Declare a wall-clock grid cap in the preregistration (V70 used a
hard 6-hour cap; the V66-grid re-run with 3 arms should size its cap from a
5-cell pilot). Stop on: NaN/nonfinite outputs, checkpoint restore mismatch,
any C′ encode attempt succeeding (hit-only violation), or the cap. No
mid-run protocol edits; amendments only between stages, timestamped.

## Cost estimate

No training. One V66-grid sweep ≈ hours on one A100 (V66 was a single-seed
sweep of the same shape). Pilot first, cap in preregistration.

## Open decisions to resolve before freezing (D1–D4)

- **D1 — persisted-store warm-up:** C′ store must be built by trained-contract
  encoding of the universe documents before timing, and reopened fresh for
  the timed phase (V70's reopened-store rule). Freeze the build pass as a
  separate untimed stage.
- **D2 — mixed-hit traffic under trained contract:** V66's mixed-hit arms
  mix hits and misses; misses under C′ require trained-contract cold encode.
  Decide: measure mixed-hit cells with trained cold misses (more honest,
  slower) or scope V71 to all-hit/mixed with pre-built misses. Default
  proposal: trained cold misses, reported as a separate cost line.
- **D3 — B′ presence in all cells vs a subset:** full-grid B′ triples nothing
  (it is one more arm, not one more sweep) but adds decode variance. Default:
  B′ in every cell; it is the byte-identity reference.
- **D4 — repetition count:** keep 15 pairs per cell (V66 parity) unless the
  pilot shows the 3-arm cell exceeding the cap; any reduction must be frozen
  before the first timed cell.

## Rollback

Limited to this cycle's V71 files: delete `experiments/v71_*` and
`results/v71_*`; restore touched serving-harness modules from pre-cycle
copies. Do not alter V65/V66/V70 artifacts or their manifests.
