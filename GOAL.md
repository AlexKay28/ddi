# GOAL

Resolve the DDI paper's two-blocker chain: quality closure on the 7B AND restart layout quality on the cache path.

## The chain

-inf fix → 7B finetune → tuned quality → tuned restart → cache claim

## Current state (2026-08-24)

- C1 quality: proved on 4B (VIBEQUEUE-11), NOT tested on 7B (gradient bug blocks finetune)
- C3 compute: null on sdpa (~1.0x realization at 95% sparsity), fa2 untested
- C4 cache correctness: PASSES (ddi_cache ≡ ddi_recompute)
- C4 cache throughput: 1.72x QPS (2.53 vs 1.47)
- C4 restart parity: FAILS (EM 0.31→0.07 hotpot, 0.05→0.00 musique on untrained 7B)

## The blocker

Cache throughput requires restart layout (position-independent KV blocks). Restart layout degrades quality on the untrained 7B because pretrained RoPE never saw documents at position 0. We cannot test whether training fixes this without a working finetune, and the finetune is blocked by the `-inf` gradient bug.

## Steps

- [x] Step 0: Quality + timing + cache experiments on untrained 7B (results posted)
- [ ] Step 1: Fix `-inf` → `torch.finfo(dtype).min` in `ddi/masks.py`
- [ ] Step 2: Run 7B finetune (self-distillation, ~8 GPU hrs, background spawn + monitor)
- [ ] Step 3: Validate quality closure on tuned 7B (quality grid: expect `ddi_qf ≈ full`)
- [ ] Step 4: Restart layout quality on tuned 7B (`cache_restart.yaml` vs tuned checkpoint)
- [ ] Step 5: (contingency) Mixed-layout finetune if Step 4 still degrades

## Success criteria

- ddi_qf ≈ full on hotpot_qa/musique (within Δ or 2σ) on the tuned 7B
- restart ≈ sequential on hotpot_qa/musique on the tuned 7B (within Δ or 2σ)
- Both on the same checkpoint, same run

## Constraints

- Spawn finetune in background, monitor progress periodically. Never block on a 8-hour run.
- HF_HUB_OFFLINE=1 everywhere.
- All results logged to results/runs.jsonl.
- Branch: feat/122-inference-isolation
