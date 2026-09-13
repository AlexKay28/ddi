"""Deferred positional encoding for position-agnostic KV cache reuse.

Inspired by LazyAttention (ICML 2026, arXiv 2606.04302): store un-rotated keys
in the cache and apply RoPE rotation at attention time using correct global
positions.  This eliminates DDI's restart-position quality wall (0.46 -> 0.07
EM on HotpotQA) because the cache is position-agnostic — the same document KV
serves any position — and correct global positions are applied at runtime.

Key property (RoPE relative rotation)::

    (R_m q)^T (R_n k) = q^T R_{n-m} k

So the attention score depends only on the relative offset n - m.  If we
cache un-rotated keys and apply the correct global rotation at attention time,
the math is identical to having baked the rotation in — but the cache is
position-agnostic.

LazyAttention's key optimization: during **prefill** (compute-bound), rotate
**K only** (K tiles are smaller than Q tiles, so rotating K is cheaper).  During
**decode** (bandwidth-bound), rotate **Q only** (Q is a single token, cheaper
than rotating the full K cache).  Never rotate both — that doubles the cost
for no benefit (the relative offset is what matters, not the absolute rotation).

The rotation uses **absolute token indices** m*theta_i to compute cos/sin,
which produces the same result as relative-offset rotation because of the
R_{n-m} factorization.  This avoids error accumulation across layers.

Usage (patching a model's attention layers)::

    from ddi.deferred_rope import enable_deferred_rope, disable_deferred_rope

    # Before encoding a document (cache write):
    enable_deferred_rope(model)
    out = model(input_ids=doc_ids, position_ids=zeros, use_cache=True)
    disable_deferred_rope(model)
    # out.past_key_values contains UN-rotated keys

    # Before attending with cached keys (cache read):
    enable_deferred_rope(model, global_positions=seq_positions)
    out = model(input_ids=query_ids, past_key_values=cache, ...)
    disable_deferred_rope(model)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


_DEFER_FLAG = "_ddi_defer_rope"
_GLOBAL_POS = "_ddi_defer_global_positions"


def enable_deferred_rope(
    model: PreTrainedModel,
    *,
    global_positions: torch.Tensor | None = None,
) -> None:
    """Enable deferred-RoPE mode on ``model``.

    When enabled, the attention patch (segmented_fa2 or flex) will:
    1. Skip ``apply_rotary_pos_emb`` on keys before writing to cache.
    2. Apply the correct global rotation to cached keys at attention time.

    Per LazyAttention: during prefill, rotate **K only** (not Q).  During
    decode, rotate **Q only** (not K).  The attention patch determines which
    mode it's in by checking the query sequence length.

    Args:
        model: the model to patch.
        global_positions: ``[seq_len]`` global position IDs for the cached keys.
            Required at attention time (cache read).  Not needed at encode time
            (cache write) — pass ``None`` or zeros.
    """
    setattr(model, _DEFER_FLAG, True)
    if global_positions is not None:
        setattr(model, _GLOBAL_POS, global_positions)
    else:
        setattr(model, _GLOBAL_POS, None)


def disable_deferred_rope(model: PreTrainedModel) -> None:
    """Disable deferred-RoPE mode, restoring standard behavior."""
    setattr(model, _DEFER_FLAG, False)
    if hasattr(model, _GLOBAL_POS):
        delattr(model, _GLOBAL_POS)


def is_deferred_rope(model: PreTrainedModel) -> bool:
    """Check whether deferred-RoPE mode is active."""
    return bool(getattr(model, _DEFER_FLAG, False))


def get_global_positions(model: PreTrainedModel) -> torch.Tensor | None:
    """Get the global position tensor for deferred rotation, or None."""
    return getattr(model, _GLOBAL_POS, None)


def rotate_keys(
    keys: torch.Tensor,
    global_positions: torch.Tensor,
    rope_theta: float,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply RoPE rotation to un-rotated keys using global positions.

    Used during **prefill**: rotate K only (per LazyAttention, K tiles are
    smaller, so this is cheaper than rotating Q).

    The rotation uses absolute token indices m*theta_i to compute cos/sin
    (per LazyAttention's stateless design), which produces the same result
    as relative-offset rotation because of the R_{n-m} factorization.

    Args:
        keys: ``[batch, heads, seq_len, head_dim]`` un-rotated keys.
        global_positions: ``[seq_len]`` global position IDs for each key.
        rope_theta: RoPE base frequency (from model config).
        dtype: compute dtype for cos/sin (defaults to keys.dtype).

    Returns:
        ``[batch, heads, seq_len, head_dim]`` rotated keys.
    """
    if dtype is None:
        dtype = keys.dtype

    head_dim = keys.shape[-1]
    seq_len = keys.shape[2]

    if global_positions.shape[0] != seq_len:
        raise ValueError(
            f"global_positions length {global_positions.shape[0]} != "
            f"keys seq_len {seq_len}"
        )

    positions = global_positions.to(keys.device).float()
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=keys.device).float() / head_dim)
    )
    freqs = positions[:, None] * inv_freq[None, :]  # [seq_len, head_dim/2]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    # Duplicate for full head_dim: cos = [cos, cos], sin = [sin, sin]
    cos_full = torch.cat([cos, cos], dim=-1)  # [seq_len, head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)  # [seq_len, head_dim]

    # Apply rotation: k_rotated = k * cos + rotate_half(k) * sin
    cos_t = cos_full.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_t = sin_full.unsqueeze(0).unsqueeze(0)
    return keys * cos_t + _rotate_half(keys) * sin_t


def rotate_query(
    query: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply RoPE rotation to a single query token using its global position.

    Used during **decode**: rotate Q only (per LazyAttention, Q is a single
    token during decode, so this is cheaper than rotating the full K cache).

    Args:
        query: ``[batch, heads, 1, head_dim]`` single query token.
        positions: ``[1]`` global position ID for the query token.
        rope_theta: RoPE base frequency.
        dtype: compute dtype for cos/sin.

    Returns:
        ``[batch, heads, 1, head_dim]`` rotated query.
    """
    return rotate_keys(query, positions, rope_theta, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotate_half: swap halves and negate the second half.

    ``[x_0..x_{d/2-1}, x_{d/2}..x_{d-1}]`` -> ``[-x_{d/2}..-x_{d-1}, x_0..x_{d/2-1}]``
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def compute_global_positions(
    doc_lengths: list[int],
    query_len: int,
    layout: str = "sequential",
) -> torch.Tensor:
    """Compute global position IDs for all tokens under the given layout.

    For deferred encoding, use ``layout="sequential"`` to get correct global
    positions (the fix for the restart wall).  The positions are:

    - Document i: [sum(doc_lengths[:i]), sum(doc_lengths[:i]) + doc_lengths[i])
    - Query: [doc_total, doc_total + query_len)

    Args:
        doc_lengths: per-document token counts.
        query_len: number of query tokens.
        layout: "sequential" (correct globals) or "restart" (per-doc from 0).

    Returns:
        ``[doc_total + query_len]`` long tensor of position IDs.
    """
    total = sum(doc_lengths) + query_len
    positions = torch.zeros(total, dtype=torch.long)

    if layout == "restart":
        offset = 0
        for length in doc_lengths:
            positions[offset : offset + length] = torch.arange(length)
            offset += length
        positions[offset:] = torch.arange(
            max(doc_lengths), max(doc_lengths) + query_len
        )
    elif layout == "sequential":
        positions = torch.arange(total)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    return positions
