# GOAL

Resolve the DDI paper's two-blocker chain: quality closure on the 7B AND restart layout quality on the cache path.

## The chain

-inf fix → 7B finetune → tuned quality → tuned restart → cache claim

## Current state (2026-08-27) — RESOLUTION COMPLETE

- C1 quality: **PARTIALLY CLOSED** on 7B. Sequential finetune achieves ddi_qf ≈ full on hotpot (0.46=0.46). Full quality regressed from base 0.57→0.46. Mixed finetune maintains parity (ddi_qf=0.39, full=0.40) at slightly lower level.
- C3 compute: null on sdpa (~1.0x realization at 95% sparsity), fa2 untested
- C4 cache correctness: PASSES (ddi_cache ≡ ddi_recompute)
- C4 cache throughput: 1.72x QPS (2.53 vs 1.47)
- C4 restart parity: **HARD WALL** — mixed-layout finetune closed perplexity gap (1.628 vs 1.635) but generation gap persists (restart EM=0.07 vs sequential=0.39 on hotpot)

## Results summary

### Quality (EM, 100 records per cell)

| mask | dataset | base | seq-tuned | mixed-tuned |
|:---|:---|:---|:---|:---|
| full | hotpot | 0.57 | 0.46 | 0.40 |
| full | musique | 0.29 | 0.23 | 0.22 |
| ddi_qf | hotpot | 0.31 | 0.46 | 0.39 |
| ddi_qf | musique | 0.05 | 0.13 | 0.12 |

**Quality parity (ddi_qf ≈ full): ACHIEVED** on hotpot at both checkpoints.

### Restart quality (EM, 100 records per cell)

| checkpoint | layout | hotpot | musique |
|:---|:---|:---|:---|
| base 7B | sequential | 0.31 | 0.05 |
| base 7B | restart | 0.07 | 0.00 |
| seq-tuned | sequential | 0.46 | 0.13 |
| seq-tuned | restart | 0.06 | 0.00 |
| mixed-tuned | sequential | 0.39 | 0.12 |
| mixed-tuned | restart | 0.07 | 0.00 |

**Restart parity: NOT ACHIEVED.** Mixed training closed perplexity (1.628 vs 1.635) but not generation (0.07 vs 0.39).

## The blocker → hard wall

Cache throughput requires restart layout (position-independent KV blocks). Restart layout degrades GENERATION quality (greedy decode) even after mixed-layout finetune, despite matching PERPLEXITY (teacher-forced next-token prediction). The model can predict tokens given correct context under both layouts, but greedy decoding under restart positions compounds small errors and diverges.

Root cause: 50/50 coin flip per record means each record sees only ONE layout per epoch. The model never learns to produce the same OUTPUT for the same record under both layouts. Perplexity (teacher-forced) only measures single-token prediction given the correct prefix; generation requires the model to produce the entire answer from scratch, where early errors cascade.

### What would be needed (not attempted)
- 2-epoch mixed: each record sees both layouts across epochs
- Restart-only finetune: train exclusively on restart positions
- Architecture change: position-agnostic attention or learned position embeddings

## Steps

- [x] Step 0: Quality + timing + cache experiments on untrained 7B (results posted)
- [x] Step 1: Fix `-inf` → `torch.finfo(dtype).min` in `ddi/masks.py`
- [x] Step 2: Run 7B sequential finetune (13.5hr, 26,231 steps, student pplx 1.648)
- [x] Step 3: Quality eval on sequential checkpoint — hotpot parity (0.46=0.46), full regressed 0.57→0.46
- [x] Step 4: Restart quality on sequential checkpoint — FAILS (restart=0.06 vs seq=0.46)
- [x] Step 5: Mixed-layout finetune (13.8hr, student pplx seq=1.628, restart=1.635)
- [x] Step 5e: Quality eval on mixed checkpoint — parity maintained (0.39≈0.40)
- [x] Step 5f: Restart quality on mixed checkpoint — STILL FAILS (restart=0.07 vs seq=0.39)
- [x] Step 6: Document results

## Success criteria — partial

- [x] ddi_qf ≈ full on hotpot_qa on the tuned 7B (0.39 ≈ 0.40 on mixed checkpoint)
- [ ] restart ≈ sequential on hotpot_qa on the tuned 7B (0.07 vs 0.39 — HARD WALL)
- Quality claim holds; restart cache claim does not hold for generation

## Constraints

- Spawn finetune in background, monitor progress periodically. Never block on a 8-hour run.
- HF_HUB_OFFLINE=1 everywhere.
- All results logged to results/runs.jsonl.
- Branch: feat/122-inference-isolation
- No bitsandbytes installed — trainable_layers=8 (top 8/28), bitsandbytes fallback to torch.optim.AdamW
- HF cache for Qwen/Qwen2.5-7B-Instruct was lost (arc FUSE mount drop); mixed finetune uses sequential checkpoint as base model
- Periodic checkpointing every 5000 steps, keep only last (~15GB)
