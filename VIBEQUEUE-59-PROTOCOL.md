# VIBEQUEUE-59 Cost, TTFT, and Crossover Protocol

## Frozen before implementation

- Hardware/runtime/checkpoint: the V58 A100, Python 3.12, torch 2.5.1+cu121,
  transformers 4.52.4, and
  `results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart`.
- Backend/layout/precision: SDPA, restart positions, bf16.
- Controlled grid: `N={2,5,10}`, `L={32,128,512}`, `Q=64`, seeds `{0,1,2}`.
- Timing: two untimed warmups and ten retained repeats per cell. Every CUDA-producing stage
  is synchronized before its boundary is recorded.
- TTFT ends after the first-token argmax is materialized on the host. Decode is excluded.
- Fresh arm: combined restart-layout DDI document and query prefill.
- Cold arm: persisted lookup miss, independent document projection and atomic disk write,
  persisted reload/assembly, then query prefill.
- Warm arm: persisted disk lookup/load, CPU-to-GPU transfer and K/V concatenation, query
  preparation, then query prefill.
- Warm stage breakdown: cache lookup/load, H2D plus assembly, query mask/position preparation,
  query forward plus first-token decision. Stage sums must match measured end-to-end TTFT apart
  from timer precision.
- Storage: retained serialized bytes per document and per document token.

## Estimands

1. Warm speedup is paired `fresh_ttft / warm_ttft` per cell and seed.
2. Under the explicitly labeled linear all-cold/all-warm mixture model, hit-rate crossover is
   `h* = (cold - fresh) / (cold - warm)`. Values outside `[0,1]` mean no feasible crossover.
3. For repeated identical requests after one cache build, reuse-count crossover is
   `k* = build_cost / (fresh - warm)`. If `warm >= fresh`, no finite reuse count wins.
4. Report medians, raw repeats, per-seed values, and paired-bootstrap 95% intervals. Do not
   substitute throughput or 16-token request latency for TTFT.

## Decision and scope

- This ticket explains cost and crossover; it does not reopen V58's failed exact/QA gates.
- A performance opportunity may be reported only where the warm TTFT interval is favorable.
- The hit-rate estimate is not presented as measured partial-hit latency. A partial-hit sweep is
  secondary and runs only if the primary grid leaves budget.
- Source is dirty, so results remain diagnostic and cannot become paper evidence.
- Rollback removes only V59 opt-in scripts/tests/report and immutable V59 attempt directories.
