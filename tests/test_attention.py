"""Tests for the attention backends (issue #6, Phase 0.3).

SDPA is the reference: `test_sdpa_matches_manual_softmax` pins its output to a naive
softmax(qk^T + mask)v computed independently. The FA2 tests here exercise validation only;
numerical cross-backend equivalence between SDPA and FA2 belongs to the Phase 0.5 gate
(issue #4), which needs CUDA. All tests in this file run green on CPU.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from ddi.attention import get_backend
from ddi.masks import build_mask, to_additive

_HAS_FA2 = importlib.util.find_spec("flash_attn") is not None
_HAS_CUDA = torch.cuda.is_available()


def _random_qkv(
    batch: int = 1,
    heads: int = 2,
    seq: int = 8,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_get_backend_dispatches():
    assert callable(get_backend("sdpa"))
    assert callable(get_backend("fa2"))
    assert callable(get_backend("triton"))
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SDPA — the reference
# ---------------------------------------------------------------------------


def test_sdpa_matches_manual_softmax():
    """SDPA output must equal a naive softmax(qk^T/sqrt(d) + mask)v to machine precision (fp32)."""
    q, k, v = _random_qkv()
    seq = q.shape[-2]
    causal_allow = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    additive = to_additive(causal_allow, dtype=torch.float32)

    sdpa = get_backend("sdpa")
    out = sdpa(q, k, v, additive, None)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores + additive  # broadcasts [T, T] → [B, H, T, T]
    ref = torch.softmax(scores, dim=-1) @ v

    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_with_block_diagonal_mask():
    """Feed a real DDI mask through SDPA end-to-end; verify shape and dtype preservation."""
    doc_lengths = [2, 2]
    query_len, answer_len = 2, 1
    total = sum(doc_lengths) + query_len + answer_len
    allow = build_mask(doc_lengths, query_len, answer_len, "ddi_qf")
    additive = to_additive(allow, dtype=torch.float32)

    q, k, v = _random_qkv(seq=total)
    out = get_backend("sdpa")(q, k, v, additive, None)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert out.device == q.device


def test_sdpa_broadcasts_2d_mask():
    """A [T, T] mask must broadcast — callers should not have to tile per (batch, head)."""
    q, k, v = _random_qkv()
    seq = q.shape[-2]
    additive_2d = to_additive(torch.tril(torch.ones(seq, seq, dtype=torch.bool)), dtype=torch.float32)
    assert additive_2d.dim() == 2

    out = get_backend("sdpa")(q, k, v, additive_2d, None)
    assert out.shape == q.shape


def test_sdpa_missing_mask_raises():
    q, k, v = _random_qkv()
    with pytest.raises(ValueError, match="additive_mask"):
        get_backend("sdpa")(q, k, v, None, None)


# ---------------------------------------------------------------------------
# FA2 — validation-only on CPU
# ---------------------------------------------------------------------------


def test_fa2_requires_cuda():
    """The CUDA check must fire before the flash_attn import so this test runs without fa2."""
    q, k, v = _random_qkv()
    with pytest.raises(RuntimeError, match="CUDA"):
        get_backend("fa2")(q, k, v, None, [2, 2, 2, 2])


@pytest.mark.skipif(not (_HAS_FA2 and _HAS_CUDA), reason="requires flash_attn and CUDA")
def test_fa2_rejects_non_doc_only_input():
    """doc_lengths must sum to seq — anything else means someone tried to pass mixed regions."""
    q, k, v = _random_qkv(seq=6)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    with pytest.raises(ValueError, match="doc-only prefill"):
        get_backend("fa2")(q, k, v, None, [2, 3])  # sum=5, seq=6


# ---------------------------------------------------------------------------
# Triton — deferred
# ---------------------------------------------------------------------------


def test_triton_raises_not_implemented():
    q, k, v = _random_qkv()
    with pytest.raises(NotImplementedError, match="Triton"):
        get_backend("triton")(q, k, v, None, None)
