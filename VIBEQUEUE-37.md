# VIBEQUEUE-37 Environment Handoff

Sprint I now has a per-sample padded `BlockMask` builder,
`ddi.flex_attention.make_batched_block_mask`. It represents each sample's
document/query layout in the batch dimension and keeps padded rows finite with
self-attention. `tests/test_flex_attention.py` checks `full`, `ddi_qf`, and
`blind` token semantics for both single and batched masks.

## Validated Kernel Results

The following CUDA checks used locally cached artifacts only; no package was
downloaded or installed.

| Check | Result |
| --- | --- |
| Single flex-vs-SDPA, `full` | max abs error `3.05e-5` |
| Single flex-vs-SDPA, `ddi_qf` | max abs error `0` |
| Single flex-vs-SDPA, `blind` | max abs error `4.88e-4` |
| Batched variable-layout flex-vs-SDPA, `full` | max abs error `3.05e-5` |
| Batched variable-layout flex-vs-SDPA, `ddi_qf` | max abs error `2.44e-4` |
| Batched variable-layout flex-vs-SDPA, `blind` | max abs error `6.10e-5` |
| Raw flex microbenchmark, B=1, H=8, D=64, T=512 | `0.142 ms`, `7,057 samples/s` |
| Raw flex microbenchmark, B=8, H=8, D=64, T=512 | `0.205 ms`, `38,968 samples/s` |

The microbenchmark is the attention kernel only. It is not an end-to-end model
throughput result and must not be used as the ticket's QPS success claim.

## Environment Evidence

The active reproducible project runtime is `.venv`: Python 3.13.12 and
`torch 2.5.1+cu121`. CUDA is available, but
`torch._dynamo.is_dynamo_supported()` is false, so flex cannot compile.

Checked local alternatives:

- `/usr/bin/python3.12`, `/usr/bin/python3.11`, and the existing
  `miniforge3/envs/aider` Python 3.12 environment contain no torch.
- `/usr/bin/python3.10` has `torch 2.13.0+cu126` with working Dynamo and can
  run raw flex, but is outside `pyproject.toml`'s `>=3.11` requirement. Its
  installed transformers/torchvision pair also prevents project test
  collection.
- The uv cache has a CPython-3.12 `torch 2.5.1+cu124` artifact and dependencies.
  With cache paths, cached CUDA libraries, and existing local Python 3.12
  headers it can run the raw checks above, but it is neither the lock-selected
  `+cu121` artifact nor an installed project environment.
- The lock-selected cached `torch 2.5.1+cu121` artifact is CPython 3.13 only.

## Remaining Blocker

No existing Python <=3.12 environment combines a CUDA torch compatible with
the project lock and the project dependencies. Provision a reproducible Python
<=3.12 CUDA environment from approved local artifacts or an approved package
source, then run the actual patched model's batch=8 versus batch=1 prefill QPS
benchmark. Until then the ticket is an environment-blocked negative result,
not an end-to-end throughput delivery.

## VIBEQUEUE-40 Follow-up

The local Python 3.10 CUDA/Dynamo runtime can now load the cached Qwen2.5-7B
checkpoint after a process-local torchvision import shim. A full-model proxy
at N=2, L=128, Q=64 uncovered and fixed direct-mask shape cropping and the
Transformers 4.57 `past_key_values` spelling for cached flex prefill. The
remaining dense-reference parity gate does not pass: cosine similarity is
0.9989066 and token-argmax agreement is 92.5%. Consequently no N=25, L=2048
prefill/generation, memory, or speed result is admitted; raw kernel results
remain distinct from model validation.
