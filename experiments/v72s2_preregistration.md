# V72 Stage 2 Preregistration — Llama-3.1-8B Transfer

**Frozen:** 2026-09-13, before any Llama training. **Ticket:** VIBEQUEUE-72 (stage 2).
**Inherits:** V70 protocol verbatim (data, schedule, optimizer, gates, bootstrap, selection).
**Sealed plan:** tahoe program `ddi-v72-v73`, seal `606ef5e8…`.

## Deltas from V70

1. **Base model:** Llama-3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`), DDI-QF 8L restart checkpoint at `results/checkpoints/meta-llama__Llama-3.1-8B-Instruct__ddi_qf__8L`.
2. **Trainable scope:** top 8 of 32 decoder layers (Llama has 32; Qwen had 28).
3. **Seed:** 47 (matching V70's original, not V72 stage 1's seed 1 — this is a cross-family test of the contract, not a seed replication).
4. **Checkpoint dirs:** `results/v72s2_llama_seed47/checkpoint_{k0,k5}`.
5. **Everything else:** identical to V70 (1000 steps, AdamW8bit, lr 1e-5, 15% independent presentations, 6h cap, paired bootstrap 10k seed 47, margin -0.02, lexicographic selection).

## Out of scope

Serving-timing (V71); dataset expansion; V73 prefix-cache baseline (separate).

## Incident note

Two mount crashes during this session (disk exhaustion during checkpoint saves) are documented in V72 stage 1 Amendment 2. The same disk-guard discipline applies here: 500M free guard before each stage; unselected-arm checkpoints from prior experiments deleted to maintain headroom.
