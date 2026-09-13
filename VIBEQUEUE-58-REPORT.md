# VIBEQUEUE-58: Persisted Document-KV Reuse Validation

## Decision

**No-go for bf16 persisted document-KV reuse as an exact or quality-preserving path.**

All-hit reuse is materially faster, but it fails both the frozen exact-equivalence gate and the
paired QA non-inferiority gate. No production default or paper evidence was changed.

## Frozen primary result

- Attempt: `results/v58_cache_qa_100_20260908T122202Z/`
- Dataset: frozen HotpotQA seed-0 manifest, 100 identical requests per arm.
- Model: `Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart`, bf16 SDPA, restart positions.
- Cache behavior: 1,000/1,000 cold document accesses missed and 1,000/1,000 subsequent
  accesses hit. Every persisted payload and full decision-logit tensor has a retained SHA-256.
- Independent audit: pass, 100 records, no integrity failures.

| Gate | Result | Decision |
|---|---:|---|
| Finite logits | 100/100 | pass |
| Logits `atol=rtol=1e-4` | 0/100 | fail |
| Exact greedy IDs | 92/100 | fail |
| EM cached - fresh | -0.030, 95% CI [-0.070, 0.000] | fail, lower bound < -0.02 |
| F1 cached - fresh | -0.0126, 95% CI [-0.0420, 0.0139] | fail, lower bound < -0.02 |
| Fixed-work cached speedup | 1.093x, 95% CI [1.082, 1.105] | pass |

Fresh EM/F1 were 0.410/0.5538; cached EM/F1 were 0.380/0.5412. The largest retained
same-prefix logit difference was 2.375.

## Localization

`results/v58_diag_bf16_20260908.json` localizes the first mismatch to layer 0 document K/V,
while cache assembly itself is exact. On the identical random input:

- bf16: layer-0 K/V differ; query max-absolute logit difference is 0.1511 and the `1e-4`
  gate fails, although the first argmax agrees;
- fp32: small GEMM-shape differences remain in document K/V, but query max-absolute logit
  difference falls to `3.43e-5`, the `1e-4` gate passes, and the argmax agrees;
- assembled persisted K/V exactly equals the independently computed source entries.

This supports numerical sensitivity from projecting documents independently rather than a
cache indexing, position, mask, serialization, or assembly defect. Precision sensitivity is
not relabeled as a semantic fix.

## Scope and stopping

The five-record gate first showed identical QA/greedy outputs, exact-logit failure, and an
inconclusive material-speed lower bound. The preregistered 100-record all-hit expansion then
failed exact and QA gates despite passing speed. Because all-hit is the strongest reuse regime,
partial-hit, reordered, changed-neighbor, organic-stream, and MuSiQue expansions were stopped;
they cannot rescue the rejected quality-preserving claim. Cold and all-hit cache-state behavior
was nevertheless verified on every primary request.

One first 100-record attempt stopped after 32 records when the home quota was exhausted:
`results/v58_cache_qa_100_20260908T113149Z/`. It remains a failed immutable attempt. The Arc
mount was recovered after clearing 46.2 GiB of disposable `uv` cache, and the campaign was
restarted from a new output directory rather than mixing provenance.

## Verification

- `ruff check scripts/diagnose_cache_kv.py scripts/validate_cache_qa.py scripts/audit_cache_qa.py tests/test_cache_workload.py`
- `pytest -p no:cacheprovider -q`: 583 passed, 28 skipped
- `pytest -p no:cacheprovider -q tests/test_cache.py tests/test_cache_workload.py`: 16 passed
- `git diff --check`
- `scripts/audit_cache_qa.py results/v58_cache_qa_100_20260908T122202Z`: pass

All V58 artifacts are dirty-source local diagnostics and are not admissible paper evidence.
