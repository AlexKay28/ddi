# V112 Preregistration — Genuine Prefix-Cache Redo

**Frozen:** 2026-09-15, before any timed measurement. **Ticket:** VIBEQUEUE-114.
**Context:** the V73 prefix-cache baseline was withdrawn (VIBEQUEUE-89) because
its cache-hit branch stored a boolean flag and always re-ran full generation.
This experiment replaces it with a genuine prefix cache and supplies the
fixed-order positive control the review required.

## Question

Does an exact-prefix KV cache — implemented for real — win in fixed-order
traffic and degrade under document reordering, while DDI's order-independent
per-document cache keeps its win under both?

## Setup (frozen)

- Model: the untrained restart checkpoint (V66 pin
  `results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart`,
  directory SHA `00df4c47…`), bf16, eager attention, batch 1, greedy decode,
  `max_new_tokens=16`, EOS stop. All five arms run these exact settings.
- Data: the frozen V62 development split only (V66 `_load_dev_content` rules;
  confirmation IDs forbidden). Documents: `pack_documents(pool, 500, 5)` —
  five 500-token documents per request, packed once and shared by the whole
  request sequence. Queries: 20 distinct dev queries via `build_query_text`
  (tokenized, `add_special_tokens=False`), one per request.
- Sequence: 20 requests back-to-back. **Fixed order**: every request uses the
  same five documents in the same order. **Reordered**: every request uses the
  same five documents in a per-request shuffled order (seed 47).
- 3 timed repeats of the whole sequence per arm; caches are reset between
  repeats; medians reported per arm.

## Arms

1. **monolithic** — standard sequential full-causal prefill over docs+query,
   then greedy decode. Every request, no caching.
2. **prefix-fixed** — exact-prefix KV cache keyed by the *ordered* document
   token tuple. Miss: full prefill (identical stream to monolithic); the
   document-prefix K/V tensors (DynamicCache) are stored (GPU-resident, like
   an LRU serving tier). Hit: stored K/Vs are loaded into a fresh DynamicCache
   and only the query tail is prefilled with positions continuing after the
   prefix; decode proceeds over the extended cache.
3. **prefix-reordered** — identical cache, but the per-request shuffle changes
   the ordered key, so every request misses.
4. **ddi-fixed** — the DDI canonical cache (`CanonicalCache` + `DocCache`,
   order-independent per-document content keys, hit-only pre-state built
   before timing), fixed order.
5. **ddi-reordered** — same, reordered sequence.

## Gates (frozen)

- **G1 (miss identity):** the first prefix request (a miss, same forward shape
  as monolithic) produces byte-identical `generated_ids` to monolithic.
- **G2 (hit mechanism — the positive control):** on every hit, the timed
  prefill forwards exactly `query_len` tokens; a per-arm counter asserts that
  document tokens are never re-forwarded on a hit.
- **G4 (fixed-order win):** median prefill ms of prefix-fixed hits is below
  half of monolithic's median prefill ms.
- **G5 (order dependence):** median prefill ms of prefix-reordered is within
  10% of monolithic (no win), while ddi-reordered's median *total* ms (its
  timing boundary is the opaque `CanonicalCache.request` call) is below
  monolithic's median total ms.

## Measured, not gated

- **Hit fidelity:** exact-match fraction of prefix-hit `generated_ids` vs
  monolithic. bf16 reduction order differs between the standalone prefix
  forward and the full-stream forward, so divergence is possible and is
  reported verbatim; DDI's canonical contract is the exact-identity contrast.
- Per-request prefill/ttft/total walls per arm; hit/miss counts.

## Deviations

Any deviation is amended here with a UTC timestamp before interpretation.
