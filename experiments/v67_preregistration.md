# V67 Preregistration — Quality Recovery Experiment

**Ticket:** VIBEQUEUE-67  
**Created:** 2026-09-10T14:00:00 UTC  
**Owner approval:** GRANTED (logged in Tracker 2026-09-10)

## Hypotheses

- **H1 (primary):** Request-time link tokens (k∈{1,5}) act as a stabilizer, re-anchoring greedy near-tie decisions. Document KVs stay context-independent (V58/V60/V65 byte-identity).
- **H2 (control recipe):** Revised perturbation-consistent training: train WITH production cache-reuse noise (per-record bf16 re-encode perturbation, cache-dropout, multi-hop supervision with natural distractors, random document order).

## Design

Revised recipe (H2) applied to 3 EQUAL-budget arms: **k0 (control, recipe changed → re-run), k1, k5**.  
k0-new vs V62-k0-old isolates H2; k1/k5 vs k0-new isolates H1.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Base checkpoint | results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart |
| Steps per arm | 500 (re-estimated by 20-step smoke) |
| Steps formula | V62 pilot: 100 steps = 463s → 4.63s/step. 500 × 4.63s ≈ 38.6 min/arm. 3 arms ≈ 116 min. |
| Effective batch size | 4 (grad_accum_steps=4, 1 record/backward) |
| Learning rate | 1.0e-5 |
| Warmup | 20 steps linear, then constant |
| Seeds | 1 (seed=0; budget allows only 1 seed per arm within 6h cap) |
| Optimizer | AdamW8bit (bitsandbytes), weight_decay=0.0 |
| Trainable layers | 8 (top 8/28, 1.86B trainable, 5.75B frozen) |
| KL temperature | 1.0 |
| Anchors | teacher_first_token_weight=1.0, full_anchor_weight=1.0, blind_anchor_weight=1.0 |
| Max grad norm | 1.0 |
| Max skipped steps | 10 |
| Layout | restart |
| Min docs | 2 |
| Arm order | k0, k1, k5 (fixed) |

## Perturbation Recipe (H2)

| Component | Value | Mechanism |
|---|---|---|
| bf16 re-encode noise rate | 0.15 | Per record, with prob 0.15, re-encode document tokens through bf16 round-trip |
| Cache dropout rate | 0.10 | Per record, with prob 0.10, drop one random doc from cache simulation |
| Multi-hop distractors | 0 (natural) | hotpot_qa already contains natural distractors |
| Document order randomization | true | Shuffle doc order per record per epoch (seed+step) |

## Link Mechanism (H1)

- k values: 0, 1, 5
- Link token IDs: [90, 91, 92, 93, 94] (unused Qwen2.5 placeholder tokens)
- Computation: per-request, NEVER stored in document KVs
- Contract: ddi.link_token_contract.LinkTokenContract
- Position layout: restart-documents-request-tail-v1
- Mask tests: tests/test_v67_link_masks.py (2-document hand-checkable + causal-leakage; MUST pass before training)

## Splits

| Split | Source | Count | sha256 |
|---|---|---|---|
| Train | V62 manifest train arm | 800 | a3b1f5b6...bcbb |
| Development | V67 fresh dev split | 100 | 9f597560...44ac (NEW, not used in V58/V60/V62/V65 dev selection) |
| Confirmation | V62 manifest confirmation arm | 100 | 529f0109...04d7 (FROZEN, untouched until confirmation) |

## Gates

- **Cache-vs-fresh:** paired non-inferiority, −0.02 floor, 10k bootstrap (seed 47), for EM and F1
- **Full-reference:** paired non-inferiority, −0.02 floor, 10k bootstrap, for EM and F1
- **Full-path preservation:** tracked (arm D), reported, no gate
- **Dev selection:** select best arm on dev by mean EM (tie-break F1), THEN untouched confirmation
- **Divergent-population recovery:** count of 22 V68 divergent ids recovered

## Stop Rules (HARD)

1. No dev improvement vs V62 k0 baseline (EM=0.40, F1=0.524) → STOP, report negative, no confirmation
2. Quality collapse (loss→NaN or max_skipped_steps exceeded) → stop that arm, continue others
3. 6h A100 hard cap → stop at nearest clean checkpoint, report partial

## Budget

- Hardware: 1× A100 80GB (CUDA_VISIBLE_DEVICES=0)
- Estimated: 2.7h (3 arms × 38.6 min + 45 min eval)
- Hard cap: 6h
- 20-step smoke re-estimates before full budget commits

## V62 Baseline (dev)

| Metric | V62 pilot (k0) | V62 base |
|---|---|---|
| EM | 0.40 | 0.60 |
| F1 | 0.524 | 0.689 |

## V65 Latency Baseline

| Metric | Value |
|---|---|
| All-hit TTFT median | 58.78 ms |
| Monolithic TTFT median | 127.62 ms |
| All-hit vs monolithic ratio | 2.17x |
| All-hit schedule wall | 5.216 s |
| Monolithic schedule wall | 7.479 s |
