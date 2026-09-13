"""The decode/prefill attention masks must own aligned storage.

Regression test for the 2026-08-06 outage: all 15 VIBEQUEUE-2 runs died at the first
decode step with `CUBLAS_STATUS_EXECUTION_FAILED`, then a sticky `CUDA error: misaligned
address` on every call after it.

`ddi.inference.generate` builds one allow-matrix covering prefill + the whole decode
budget and then slices it. Those slices are VIEWS into the big matrix:

* the prefill block is non-contiguous (its row stride is the full matrix width);
* each decode row is contiguous by stride but carries a storage offset of `row * width`.

A bf16 tensor-core GEMM wants its operands 16-byte (8-element) aligned, so a mask whose
data pointer sits at an arbitrary element offset is rejected by cuBLAS. `.clone()` forces
a fresh allocation and therefore a base-aligned pointer.

This file deliberately imports only `torch` — no `transformers`, no GPU — because the
suites that would otherwise cover `generate` (`test_inference`, `test_model`) cannot be
collected on a host without `transformers`, which is exactly why this shipped.
"""

from __future__ import annotations

import torch

WIDTH = 64
PREFIX = 40
BUDGET = 24


def _allow_matrix() -> torch.Tensor:
    return torch.zeros(WIDTH, WIDTH, dtype=torch.bfloat16)


def test_contiguous_does_not_rescue_an_offset_row() -> None:
    """The trap: the decode row is already contiguous, so `.contiguous()` is a no-op.

    Pinned because it is the obvious wrong fix — it looks like it addresses the problem
    and leaves the unaligned base pointer exactly where it was.
    """
    full = _allow_matrix()
    row = full[PREFIX, : PREFIX + 1]

    assert row.is_contiguous(), "a row slice is contiguous by stride"
    assert row.storage_offset() != 0, "...but it points into the middle of the matrix"
    assert row.contiguous().storage_offset() == row.storage_offset()


def test_cloned_decode_row_owns_aligned_storage() -> None:
    full = _allow_matrix()
    for i in range(BUDGET):
        kv_len = PREFIX + i + 1
        step_mask = full[PREFIX + i, :kv_len].clone().reshape(1, 1, 1, kv_len)

        assert step_mask.storage_offset() == 0, f"decode step {i} is not base-aligned"
        assert step_mask.is_contiguous()
        assert step_mask.shape == (1, 1, 1, kv_len)


def test_cloned_prefill_block_is_contiguous() -> None:
    full = _allow_matrix()

    raw = full[:PREFIX, :PREFIX]
    assert not raw.is_contiguous(), "the prefill block is a strided view of the matrix"

    fixed = raw.clone().unsqueeze(0).unsqueeze(0)
    assert fixed.is_contiguous()
    assert fixed.storage_offset() == 0
    assert fixed.shape == (1, 1, PREFIX, PREFIX)


def test_clone_preserves_mask_values() -> None:
    """Alignment must not come at the cost of changing what the mask says."""
    full = _allow_matrix()
    full.uniform_(-1.0, 1.0)
    full[3, 5] = float("-inf")

    for i in range(BUDGET):
        kv_len = PREFIX + i + 1
        view = full[PREFIX + i, :kv_len]
        assert torch.equal(view.clone().reshape(1, 1, 1, kv_len).flatten(), view)

    block = full[:PREFIX, :PREFIX]
    assert torch.equal(block.clone(), block)
