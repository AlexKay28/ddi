"""Attention backends behind one interface (issue #6, Phase 0.3).

Backends: dense SDPA + explicit additive mask (**reference semantics** — the definition of
correct); FlashAttention-2 varlen; optional Triton block-diagonal. All three satisfy the same
signature so they double as each other's tests.
"""

from __future__ import annotations

from typing import Literal, Protocol

import torch

Backend = Literal["sdpa", "fa2", "triton"]


class AttentionBackend(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: torch.Tensor | None,
        doc_lengths: list[int] | None,
    ) -> torch.Tensor: ...


def _sdpa_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: torch.Tensor | None,
    doc_lengths: list[int] | None,  # noqa: ARG001 -- unused; kept for interface parity
) -> torch.Tensor:
    if additive_mask is None:
        raise ValueError("SDPA backend requires an explicit additive_mask")

    # 2D masks are broadcast up so callers can pass a single [T, T] and skip the per-batch tile.
    if additive_mask.dim() == 2:
        additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)

    # is_causal=False is deliberate: causality lives in the mask, not in a kernel flag. Passing
    # both would double-mask and, on some backends, silently override the additive term.
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=additive_mask, is_causal=False
    )


def _fa2_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: torch.Tensor | None,  # noqa: ARG001 -- varlen encodes structure geometrically
    doc_lengths: list[int] | None,
) -> torch.Tensor:
    # CUDA check happens before importing flash_attn so the CPU-only error path does not require
    # the fa2 package to be installed at all.
    if not q.is_cuda:
        raise RuntimeError("FA2 backend requires CUDA")
    if doc_lengths is None:
        raise ValueError("FA2 varlen backend requires doc_lengths")

    batch, heads, seq, head_dim = q.shape
    if batch != 1:
        raise ValueError("FA2 varlen backend supports batch=1 only in Phase 0")
    if sum(doc_lengths) != seq:
        raise ValueError("FA2 varlen backend is scoped to doc-only prefill")

    # Lazy import: the package is an optional extra, and CPU-only test runs must not require it.
    from flash_attn import flash_attn_varlen_func

    cu_seqlens = torch.zeros(len(doc_lengths) + 1, dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = torch.tensor(doc_lengths, dtype=torch.int32, device=q.device).cumsum(0)
    max_seqlen = max(doc_lengths)

    # fa2 varlen expects [total_tokens, heads, head_dim]; drop the batch dim and put heads second.
    q_flat = q.squeeze(0).transpose(0, 1).contiguous()
    k_flat = k.squeeze(0).transpose(0, 1).contiguous()
    v_flat = v.squeeze(0).transpose(0, 1).contiguous()

    out_flat = flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    return out_flat.transpose(0, 1).unsqueeze(0).contiguous()


def _triton_backend(
    q: torch.Tensor,  # noqa: ARG001
    k: torch.Tensor,  # noqa: ARG001
    v: torch.Tensor,  # noqa: ARG001
    additive_mask: torch.Tensor | None,  # noqa: ARG001
    doc_lengths: list[int] | None,  # noqa: ARG001
) -> torch.Tensor:
    raise NotImplementedError("Triton backend deferred; use sdpa or fa2 (Phase 3 revisits)")


_BACKENDS: dict[str, AttentionBackend] = {
    "sdpa": _sdpa_backend,
    "fa2": _fa2_backend,
    "triton": _triton_backend,
}


def get_backend(name: Backend) -> AttentionBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {tuple(_BACKENDS)}")
    return _BACKENDS[name]
