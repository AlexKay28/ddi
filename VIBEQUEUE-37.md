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

## Historical Environment Blocker

At this stage no existing Python <=3.12 environment combined a CUDA torch
compatible with the project lock and dependencies. This was superseded by the
later V47 diagnostic below; it is not the current instruction to run a batch
benchmark.

## VIBEQUEUE-40 Follow-up

The local Python 3.10 CUDA/Dynamo runtime can now load the cached Qwen2.5-7B
checkpoint after a process-local torchvision import shim. A full-model proxy
at N=2, L=128, Q=64 uncovered and fixed direct-mask shape cropping and the
Transformers 4.57 `past_key_values` spelling for cached flex prefill. The
remaining dense-reference parity gate does not pass: cosine similarity is
0.9989066 and token-argmax agreement is 92.5%. Consequently no N=25, L=2048
prefill/generation, memory, or speed result is admitted; raw kernel results
remain distinct from model validation.

## VIBEQUEUE-44 Resolution

No separate batched end-to-end flex throughput result is admitted. The
full-model gate fails at the smaller proxy shape `N=2, L=128, Q=64` before a
batch-8 shape can be considered: maximum logit error is `2.953125`, mean error
is `0.0753116`, cosine similarity is `0.9989066` (below the `0.999` gate), and
token-argmax agreement is `92.5%`. The batched `BlockMask` unit and raw-kernel
checks establish mask and kernel behavior only; they do not establish model
parity. Consequently no batch-throughput number is reported, including for a
smaller shape, until full-model parity passes for the exact batched path.

## VIBEQUEUE-47 Superseding Diagnostic

The aggregate VIBEQUEUE-40/44 proxy values above are historical. The current
per-mask diagnostic on torch 2.13.0+cu126 reports cosine/argmax agreement of
`0.993342`/`87.19%` for `full`, `0.998553`/`95.00%` for `ddi_qf`, and
`0.998701`/`94.38%` for `blind`. All three fail the required 0.999 cosine and
100% argmax gates, preserving the mechanistic boundary and no-throughput
disposition.

## VIBEQUEUE-49 Scope Disposition

Compiled Flex timing is deferred outside the bounded paper, not completed.
The raw single/batched kernel checks above remain mechanistic diagnostics only.
No Flex model-throughput, batching, or serving result is admitted, and timing
must not begin unless the unchanged exact gates in `EVIDENCE_TO_CLAIM.md` first
pass for every applicable mask and path.
