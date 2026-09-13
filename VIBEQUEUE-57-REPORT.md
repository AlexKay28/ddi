# V57 DDI GPU Validation Report

## Verdict

The validation protocol completed with a bounded negative result. The attached A100 was used
for model-scale Flex parity, persisted document-KV reuse, and a 100-request matched
HotpotQA quality/performance campaign. No acceptance threshold was changed.

- Compiled Flex is not admitted as a validated model backend: all six required proxy cells
  fail the 100% token-argmax gate.
- Persisted document-KV reuse is not admitted as correctness preserving: every bounded smoke
  cell fails the `atol=rtol=1e-4` numerical gate and five of eight trajectories differ.
- Dense SDPA does not turn DDI-QF analytical sparsity into material speed: the matched
  100-request prefill interval is tightly centered near 1.0 and the request interval favors
  full attention slightly.
- The matched HotpotQA artifact confirms a material quality loss for base Qwen2.5-7B under
  DDI-QF on these records. This is a seed-0 dirty-source diagnostic, not a new paper claim.

The correctness failures activate the preregistered stop rule. Target-shape Flex timing,
kernel/module promotion, the expensive cache anchor, geometry/hit/concurrency expansion, and
paper figure promotion were not run.

## Frozen environment

| Field | Value |
| --- | --- |
| Base revision | `16dd67d7e53a8168cf6580e6d0197c219e6ba198` |
| Source state | Dirty; exact patches and executing sources retained per attempt |
| GPU | NVIDIA A100 80GB PCIe |
| Driver | 580.65.06 |
| Validation Python | 3.12.13 |
| Torch / CUDA | 2.5.1+cu121 / 12.1 |
| Transformers | 4.52.4 |
| Model | Local offline Qwen2.5-7B-Instruct `yt-cache` snapshot |
| Precision | bf16 |

The checkout `.venv` uses Python 3.13, where Torch 2.5.1 refuses Dynamo. A temporary Python
3.12 runtime with matching development headers was therefore frozen for Flex execution.

## Flex parity

Artifact: `results/v57_20260908T1208Z/`.

All cells pass finite-logit, cache-length, cosine and one-step cached-decode argmax checks, but
fail the required 100% prefill token-argmax agreement.

| N | Variant | Cosine | Token argmax | Max abs | Decode argmax |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | full | 0.999837 | 96.875% | 0.40625 | equal |
| 1 | ddi_qf | 0.999837 | 96.875% | 0.40625 | equal |
| 1 | blind | 0.999858 | 98.438% | 0.37500 | equal |
| 2 | full | 0.999834 | 96.875% | 0.65625 | equal |
| 2 | ddi_qf | 0.999870 | 96.875% | 1.12500 | equal |
| 2 | blind | 0.999867 | 98.125% | 1.12500 | equal |

The first failure artifact localizes divergence to attention layer 0. Explicit scale is equal
and projected Q/K/V are bit-identical, narrowing the mismatch to the Flex versus SDPA attention
calculation rather than model weights, positions, projections or GQA expansion.

## Persisted KV cache

Artifact: `results/v57_cache_smoke_20260908T1220Z/`.

The bounded real-model smoke used seed 0, `N=2,L=32,Q=8,A=4`, one warmup and two measured
requests. It covered cold, warm, reordered, partial reuse, duplicate documents and a fresh
actual stream with independently predicted hit vectors.

| Check | Result |
| --- | ---: |
| Validation comparisons | 8 |
| Finite comparisons | 8 / 8 |
| Numerical failures (`1e-4`) | 8 / 8 |
| Fixed greedy-ID failures | 5 / 8 |
| Natural greedy-ID failures | 5 / 8 |
| Hit-state failures | 0 / 8 |
| Maximum same-prefix absolute error | 0.8125 |
| Cache / fresh DDI QPS ratio | 0.7504 diagnostic only |
| Cache / full QPS ratio | 0.7322 diagnostic only |

The persisted format now stores and verifies schema version, manifest identity, exact token-ID
digest and tensor dtype, and rejects payload/path key mismatches. The independent audit reloads
full logits and every cache payload; artifact integrity passes while correctness explicitly
remains false.

## Matched QA and performance

Artifact: `results/v57_matched_qa_hotpot100_seed0_20260908T1238Z/`.

The campaign joins exact input IDs, document/query geometry, actual causal-pair sparsity,
natural output IDs, EM/F1/presence, and five-repeat fixed-work prefill/request timings for the
same 100 frozen HotpotQA seed-0 request IDs.

| Arm | EM | F1 | Presence | Mean median prefill | Mean median request |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.5600 | 0.7034 | 0.6100 | 136.38 ms | 499.66 ms |
| ddi_qf | 0.3000 | 0.4153 | 0.3100 | 136.21 ms | 502.21 ms |

Paired bootstrap contrasts:

| Contrast | Estimate | 95% interval | Gate |
| --- | ---: | --- | --- |
| full - DDI-QF EM | 0.2600 | [0.1600, 0.3600] | DDI quality loss |
| full - DDI-QF F1 | 0.2881 | [0.1916, 0.3867] | DDI quality loss |
| full - DDI-QF presence | 0.3000 | [0.1900, 0.4000] | diagnostic |
| full / DDI-QF prefill latency | 1.0013 | [1.0008, 1.0018] | fails 1.05 speed gate |
| full / DDI-QF request latency | 0.9955 | [0.9924, 0.9986] | fails 1.05 speed gate |

`post_audit.json` recomputes every quality metric, exact input hash, repeat count, work count,
median and paired interval. `tradeoffs.svg` plots quality against actual sparsity and observed
request trade-offs. Cache-hit joins are unavailable because the cache correctness gate failed;
kernel and attention-module timing are unavailable because Flex model parity failed.

## Implementation

- `ddi/flex_attention.py`: explicit common scale and opt-in projected Q/K/V plus attention
  output capture.
- `ddi/flex_parity.py`: one-model matched backend comparison, explicit positions, independent
  prefill caches, matched decode, hard gates, JSON output and first-divergence artifacts.
- `ddi/cache.py`: self-validating persisted identity and token metadata.
- `scripts/validate_cache_workload.py`: configurable bounded/full campaigns, duplicate and
  fresh-stream checks, exact expected hits and explicit correctness verdict.
- `scripts/audit_cache_workload.py`: dynamic campaign audit with manifest, logits and cache
  payload recomputation.
- `scripts/validate_matched_qa.py`: immutable matched natural-QA/fixed-work campaign.
- `scripts/audit_matched_qa.py`: independent paired audit and SVG trade-off rendering.

## Verification

- Full unit suite: `582 passed, 28 skipped`.
- Focused new/changed tests: `32 passed`.
- Scoped Ruff checks for all changed validation files: pass.
- Repository-wide Ruff still reports 16 unrelated pre-existing findings in untouched files.
- `git diff --check`: pass.

## Evidence boundary

These runs are reproducible local dirty-source diagnostics with retained snapshots, not clean
revision deciding evidence. They support a completed negative finding and prevent invalid
speed claims. No commit, push or paper evidence promotion was performed.
