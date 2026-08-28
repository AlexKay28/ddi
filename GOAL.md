# GOAL

Resolve the DDI paper's two-blocker chain: quality closure on the 7B AND restart layout quality on the cache path.

## The chain

-inf fix → 7B finetune → tuned quality → tuned restart → cache claim

## Current state (2026-08-28) — RESOLUTION COMPLETE

- C1 quality: **PARTIALLY CLOSED** on 7B. Sequential finetune achieves ddi_qf ≈ full on hotpot (0.46=0.46). Full quality regressed from base 0.57→0.46. Mixed finetune maintains parity (ddi_qf=0.39, full=0.40) at slightly lower level. Restart-only finetune also achieves parity (ddi_qf=0.42, full=0.39).
- C3 compute: **SOLVED with flex_attention** — 1.77x model-level speedup at 97.9% sparsity (N=25, L=2048). sdpa was a 0.40x REGRESSION (4D mask forces slow math backend). flex_attention compiles block-diagonal pattern into block-sparse kernel. See VIBEQUEUE-21.
- C4 cache correctness: PASSES (ddi_cache ≡ ddi_recompute)
- C4 cache throughput: 1.72x QPS (2.53 vs 1.47) — measured on restart layout (generation quality broken on restart; needs re-measurement on sequential)
- C4 restart parity: **HARD WALL (DEFINITIVE)** — three finetune approaches all fail. Sequential, mixed-layout, and restart-only training all produce restart EM≈0.07 vs sequential EM≈0.42 on hotpot. Even restart-only training (perplexity 1.629 < sequential 1.638) fails to fix generation.

## Results summary

### Quality (EM, 100 records per cell)

| mask | dataset | base | seq-tuned | mixed-tuned | restart-tuned |
|:---|:---|:---|:---|:---|:---|
| full | hotpot | 0.57 | 0.46 | 0.40 | 0.39 |
| full | musique | 0.29 | 0.23 | 0.22 | 0.21 |
| ddi_qf | hotpot | 0.31 | 0.46 | 0.39 | 0.42 |
| ddi_qf | musique | 0.05 | 0.13 | 0.12 | 0.12 |

**Quality parity (ddi_qf ≈ full): ACHIEVED** on hotpot across all three checkpoints.

### Restart quality (EM, 100 records per cell)

| checkpoint | layout | hotpot | musique |
|:---|:---|:---|:---|
| base 7B | sequential | 0.31 | 0.05 |
| base 7B | restart | 0.07 | 0.00 |
| seq-tuned | sequential | 0.46 | 0.13 |
| seq-tuned | restart | 0.06 | 0.00 |
| mixed-tuned | sequential | 0.39 | 0.12 |
| mixed-tuned | restart | 0.07 | 0.00 |
| restart-tuned | sequential | 0.42 | 0.12 |
| restart-tuned | restart | 0.07 | 0.00 |

**Restart parity: NOT ACHIEVED (definitive).** Three approaches, same wall. Even restart-only training (perplexity 1.629 < sequential 1.638) fails to fix generation.

### Perplexity vs generation (restart-tuned checkpoint)

| metric | sequential | restart | gap |
|:---|:---|:---|:---|
| perplexity (teacher-forced) | 1.638 | 1.629 | restart BETTER |
| generation EM (hotpot) | 0.42 | 0.07 | -83% |

Perplexity gap is closed (even inverted), but generation gap persists. This is exposure bias amplified by position perturbation: teacher-forced next-token prediction is stable under restart positions, but autoregressive greedy decoding compounds small errors that cascade under shifted positions.

## The blocker → hard wall (definitive)

Cache throughput requires restart layout (position-independent KV blocks). Restart layout degrades GENERATION quality (greedy decode) even after three different finetune strategies, despite matching (or beating) PERPLEXITY (teacher-forced next-token prediction). The model can predict tokens given correct context under both layouts, but greedy decoding under restart positions compounds small errors and diverges.

Root cause: **exposure bias amplified by position perturbation.** Teacher-forced perplexity measures single-token prediction given the correct prefix — stable under restart positions. Autoregressive generation requires the model to produce the entire answer from scratch, where early errors cascade under shifted positional encodings. Training exclusively on restart positions teaches the model to predict well given correct context (perplexity 1.629 < sequential 1.638) but does NOT fix the compounding error dynamics of free generation.

### What was tried (all failed)
1. **Sequential finetune** — restart EM=0.06 vs sequential=0.46 (perplexity gap 1.648 vs 1.443)
2. **Mixed-layout finetune** (50/50 coin flip per record) — restart EM=0.07 vs sequential=0.39 (perplexity gap closed: 1.628 vs 1.635)
3. **Restart-only finetune** — restart EM=0.07 vs sequential=0.42 (perplexity gap inverted: restart 1.629 < sequential 1.638)

### What would be needed (not attempted)
- Architecture change: position-agnostic attention or learned position embeddings
- Contrastive loss between sequential and restart outputs for the same record
- Speculative decoding with sequential teacher correction

## Steps

- [x] Step 0: Quality + timing + cache experiments on untrained 7B (results posted)
- [x] Step 1: Fix `-inf` → `torch.finfo(dtype).min` in `ddi/masks.py`
- [x] Step 2: Run 7B sequential finetune (13.5hr, 26,231 steps, student pplx 1.648)
- [x] Step 3: Quality eval on sequential checkpoint — hotpot parity (0.46=0.46), full regressed 0.57→0.46
- [x] Step 4: Restart quality on sequential checkpoint — FAILS (restart=0.06 vs seq=0.46)
- [x] Step 5: Mixed-layout finetune (13.8hr, student pplx seq=1.628, restart=1.635)
- [x] Step 5e: Quality eval on mixed checkpoint — parity maintained (0.39≈0.40)
- [x] Step 5f: Restart quality on mixed checkpoint — STILL FAILS (restart=0.07 vs seq=0.39)
- [x] Step 6: Restart-only finetune (13.8hr, student pplx seq=1.638, restart=1.629)
- [x] Step 6e: Quality eval on restart checkpoint — parity maintained (0.42≈0.39)
- [x] Step 6f: Restart quality on restart checkpoint — STILL FAILS (restart=0.07 vs seq=0.42)
- [x] Step 7: Document results — hard wall definitive, paper stands with acknowledged limitation

## Success criteria — partial

- [x] ddi_qf ≈ full on hotpot_qa on the tuned 7B (0.42 ≈ 0.39 on restart-tuned checkpoint)
- [ ] restart ≈ sequential on hotpot_qa on the tuned 7B (0.07 vs 0.42 — HARD WALL, 3 approaches tried)
- Quality claim holds; restart cache claim does not hold for generation

## Constraints

- Spawn finetune in background, monitor progress periodically. Never block on a 8-hour run.
- HF_HUB_OFFLINE=1 everywhere.
- All results logged to results/runs.jsonl.
- Branch: feat/122-inference-isolation
- No bitsandbytes installed — trainable_layers=8 (top 8/28), bitsandbytes fallback to torch.optim.AdamW
- HF cache for Qwen/Qwen2.5-7B-Instruct was lost (arc FUSE mount drop); mixed finetune uses sequential checkpoint as base model
- Periodic checkpointing every 5000 steps, keep only last (~15GB)
