# V57 GPU Validation Protocol

## Frozen before execution

- Base revision: `16dd67d7e53a8168cf6580e6d0197c219e6ba198`.
- Source state: dirty; every attempt must retain the exact tracked patch and executing sources.
- Hardware: one NVIDIA A100 80GB PCIe, driver 580.65.06.
- Validation runtime: Python 3.12.12, torch 2.5.1+cu121, CUDA 12.1,
  transformers 4.52.4. The checkout Python 3.13 environment is excluded because torch
  2.5.1 Dynamo cannot compile Flex on Python 3.13.
- Model/tokenizer: local offline `Qwen/Qwen2.5-7B-Instruct` snapshot named `yt-cache`.
- Precision: bf16 for model-scale gates; tiny CPU tests use fp32 and are diagnostic only.
- Seeds: Flex 47; cache 0, 1, 2; matched QA starts with frozen seed 0 records.
- Flex proxy: `N={1,2}`, `L=128`, `Q=64`, variants `full`, `ddi_qf`, `blind`.
- Flex gates: finite logits, cache length equal to input length, cosine >= 0.999, 100%
  token-argmax agreement, finite cached decode, and identical next-token argmax.
- Cache anchor: `N=5,L=500,Q=32,A=16`, 32 documents, Zipf 1.5, seeds 0/1/2,
  10 warmups and 50 measured requests when the bounded smoke passes.
- Cache gates: finite logits, `atol=rtol=1e-4`, exact fixed-work and natural greedy IDs,
  and exact expected cold/warm/reordered/partial/duplicate/fresh-stream hit traces.
- Timing: at least five repeats after warmup. Compilation/first use is separate from steady
  state. A material speedup requires the paired ratio's 95% lower bound to exceed 1.05.
- Initial GPU budget: four hours or loss of the attached GPU, whichever occurs first. Run
  correctness smoke cells before expanding geometry. OOM and timed-out cells are retained.
- No acceptance threshold may be changed after observing results. No failed correctness cell
  may be promoted to validated performance evidence.

## Attempt layout

Each attempt uses a fresh `results/v57_*` directory containing a manifest, source snapshot,
working patch, exact inputs, raw validation and timing rows, tensor diagnostics, summary, and
audit. Attempts are never overwritten. Existing V51/V53 artifacts are read-only.

## Staged execution

1. Run CPU unit gates and reproduce the existing model-scale failure.
2. Run shared-model Flex prefill/cache/decode parity with explicit positions and scale. On the
   first failure, retain projected Q/K/V and attention outputs at the first divergent layer.
3. Run self-validating persisted-cache scenarios and a bounded 7B smoke. Expand to the V53
   anchor only if correctness passes; otherwise retain a completed negative finding.
4. Run natural QA and fixed-work timing on the same frozen requests. Join by immutable request
   identity and report quality, actual sparsity, cache hits, latency and uncertainty together.
5. Audit all artifacts and publish a bounded verdict to VIBEQUEUE-57 and VIBEQUEUE-56.

## Rollback

Validation code is opt-in. Rollback removes the new V57 validator/tests and the new immutable
attempt directory; production generation defaults and historical artifacts remain unchanged.
