# V58 Matched Document-KV Reuse Protocol

## Frozen before execution

- Hardware: one NVIDIA A100 80GB PCIe, driver 580.65.06.
- Runtime: Python 3.12.13, torch 2.5.1+cu121, CUDA 12.1, transformers 4.52.4.
- Primary checkpoint: `results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart`.
  This checkpoint is explicitly restart-trained. R5b checkpoints are not substituted.
- Backend/layout: bf16 SDPA and restart positions in both fresh and cached arms.
- Dataset: frozen HotpotQA seed-0 100-record manifest. MuSiQue is secondary if budget remains.
- Exact-equivalence gate: finite same-prefix logits, `atol=rtol=1e-4`, and exact greedy IDs.
- QA non-inferiority gate: for both EM and F1, the paired bootstrap 95% lower bound of
  `cached - fresh` must be at least `-0.02`. This does not override exact-equivalence failure.
- Material timing gate: paired cached/fresh ratio 95% lower bound greater than 1.05.
- Timing: one warmup and at least five retained repeats per request/regime. Natural stopping
  is separate from 16-token fixed work with 15 decode forwards.
- Cache regimes: cold, all-hit, partial-hit, reordered, changed-neighbor and organic stream.
- Initial budget: four GPU hours or loss of the attached GPU. Start with N=2,L=32,Q=8
  localization and five QA records before the 100-record campaign.
- Source is dirty. Every attempt retains exact executing sources and patch and remains
  diagnostic. No threshold may change after observing reuse outcomes.

## Decision rules

1. Exact equivalence and QA non-inferiority are reported independently.
2. A semantic/indexing defect may be fixed only after localization and replay of the original
   failure. Precision sensitivity is not silently relabeled as a semantic fix.
3. A quality-preserving performance result requires QA non-inferiority, matched work, material
   timing benefit, artifact integrity and explicit cache/storage conditions.
4. Failed and unavailable cells remain in the report. Flex is out of scope.

## Rollback

All V58 runners are opt-in. Rollback removes the V58 scripts/tests and immutable attempt
directories; production defaults and V53/V57 artifacts remain untouched.
