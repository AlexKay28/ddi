<!-- T1 — mask taxonomy. Static artifact per RESEARCH_PLAN §10 (source of truth: ddi/masks.py). -->

Attention allowance under each variant. `T = sum(doc_lengths) + query_len + answer_len`.
For every variant, positions inside a region are causal within that region.

| variant | documents (block-diagonal?) | query / dialog → docs | answer → all |
|---|---|---|---|
| `full` | causal over the whole prompt (no isolation) | causal (sees everything preceding) | causal |
| `ddi_qf` | mutually blind (each doc causal within itself) | full attention over all documents | causal, sees all |
| `ddi_strict` | mutually blind | only the last query row ("SEP trick") attends to docs | causal, sees all |
| `ddi_bridge_k` | mutually blind | last `k` query rows attend to docs (`k=1` ≡ `ddi_strict`, `k=query_len` ≡ `ddi_qf`) | causal, sees all |
| `blind` | mutually blind, still self-causal | query blind to all docs (control) | causal over query+answer only |

Notes:

- **N=1 collapse.** With a single document, `ddi_qf` / `ddi_strict` / `ddi_bridge_k` all reduce
  to `full` (nothing to isolate). This is the Phase 0.5 correctness gate #4 checks at the mask
  level and the model level. `blind` deliberately does **not** collapse — it is the control
  that measures "what if the model saw no documents at all".
- **RoPE positions are orthogonal.** The `layout` axis (`sequential` vs `restart`) determines
  position IDs; each combination `(variant, layout)` is a distinct experimental cell.
- **Additive form.** `ddi.masks.to_additive(allow_bool)` converts True → 0.0 and False → `-inf`,
  representable in bf16 without NaN.
