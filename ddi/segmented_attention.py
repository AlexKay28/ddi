"""CPU reference for DDI's segmented cold-attention semantics.

This is intentionally a mathematical reference, not a FlashAttention, Flex, or
model integration.  Q/K/V are explicit token-major tensors.  Document K/V may
live in an arbitrary packed-cache order; ``DocumentSegment.cache_slots`` maps
logical document rows to that storage without expanding GQA K/V heads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ddi.masks import Variant, build_mask


@dataclass(frozen=True)
class DocumentSegment:
    """One logical document's rows in a physically packed document K/V cache."""

    cache_slots: torch.Tensor
    logical_positions: torch.Tensor
    logical_document_id: object | None = None


@dataclass(frozen=True)
class SegmentedAttentionOutput:
    """Outputs for document prefill, query prefill, and optional decode rows."""

    document: torch.Tensor
    query: torch.Tensor
    decode: torch.Tensor | None


def merge_lse(partials: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Merge attention partials ``(output, logsumexp)`` without averaging.

    Each LSE has shape ``[rows, q_heads]`` and normalizes its matching output.
    The calculation is stable even when a segment dominates the other scores.
    """
    if not partials:
        raise ValueError("merge_lse requires at least one partial")
    outputs, lses = zip(*partials, strict=True)
    shape = outputs[0].shape
    if len(shape) != 3 or lses[0].shape != shape[:2]:
        raise ValueError("partials must be [rows, heads, dim] outputs with [rows, heads] LSEs")
    if any(output.shape != shape or lse.shape != shape[:2] for output, lse in partials):
        raise ValueError("all partials must have identical shapes")
    if any(
        not torch.isfinite(output).all() or not torch.isfinite(lse).all()
        for output, lse in partials
    ):
        raise ValueError("partials must be finite")
    stacked_lse = torch.stack(lses)
    total_lse = torch.logsumexp(stacked_lse, dim=0)
    weights = torch.exp(stacked_lse - total_lse).unsqueeze(-1)
    return (torch.stack(outputs) * weights).sum(dim=0)


def segmented_attention(
    document_q: torch.Tensor,
    document_k: torch.Tensor,
    document_v: torch.Tensor,
    query_q: torch.Tensor,
    query_k: torch.Tensor,
    query_v: torch.Tensor,
    documents: Sequence[DocumentSegment],
    query_positions: torch.Tensor,
    *,
    variant: Variant,
    decode_q: torch.Tensor | None = None,
    decode_k: torch.Tensor | None = None,
    decode_v: torch.Tensor | None = None,
    decode_positions: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> SegmentedAttentionOutput:
    """Apply a ``build_mask`` variant to explicit document, query, and decode Q/K/V.

    Cache slots are gathered into logical document order before applying the
    repository's single source of truth. Padding and caller-supplied masks are
    deliberately unsupported: this reference is for exact, unpadded layouts.
    """
    if attention_mask is not None:
        raise ValueError("segmented_attention does not accept masks; pass exact unpadded segments")
    _validate_inputs(
        document_q,
        document_k,
        document_v,
        query_q,
        query_k,
        query_v,
        documents,
        query_positions,
        decode_q,
        decode_k,
        decode_v,
        decode_positions,
    )

    document_q_parts: list[torch.Tensor] = []
    document_k_parts: list[torch.Tensor] = []
    document_v_parts: list[torch.Tensor] = []
    doc_lengths: list[int] = []
    offset = 0
    for segment in documents:
        length = segment.cache_slots.numel()
        document_q_parts.append(document_q[offset : offset + length])
        document_k_parts.append(document_k.index_select(0, segment.cache_slots))
        document_v_parts.append(document_v.index_select(0, segment.cache_slots))
        doc_lengths.append(length)
        offset += length
    logical_document_q = torch.cat(document_q_parts) if document_q_parts else document_q.clone()
    logical_document_k = torch.cat(document_k_parts) if document_k_parts else document_k.clone()
    logical_document_v = torch.cat(document_v_parts) if document_v_parts else document_v.clone()
    q_parts = [logical_document_q, query_q]
    k_parts = [logical_document_k, query_k]
    v_parts = [logical_document_v, query_v]
    if decode_q is not None:
        q_parts.append(decode_q)
        k_parts.append(decode_k)
        v_parts.append(decode_v)
    q = torch.cat(q_parts)
    k = torch.cat(k_parts)
    v = torch.cat(v_parts)
    allow = build_mask(
        doc_lengths, query_q.shape[0], 0 if decode_q is None else decode_q.shape[0], variant
    ).to(q.device)
    output = _masked_attention(q, k, v, allow)
    document_end = logical_document_q.shape[0]
    query_end = document_end + query_q.shape[0]
    document_out = output[:document_end]
    query_out = output[document_end:query_end]
    decode_out = output[query_end:] if decode_q is not None else None
    return SegmentedAttentionOutput(document_out, query_out, decode_out)


def segmented_attention_decomposed(
    document_q: torch.Tensor,
    document_k: torch.Tensor,
    document_v: torch.Tensor,
    query_q: torch.Tensor,
    query_k: torch.Tensor,
    query_v: torch.Tensor,
    documents: Sequence[DocumentSegment],
    query_positions: torch.Tensor,
    *,
    variant: Variant,
    decode_q: torch.Tensor | None = None,
    decode_k: torch.Tensor | None = None,
    decode_v: torch.Tensor | None = None,
    decode_positions: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> SegmentedAttentionOutput:
    """Decompose DDI-QF or blind attention into stable CPU-reference partials.

    Each document prefill is computed locally. Query and decode rows merge their
    visible document partials with their causal query/decode-tail partials using
    :func:`merge_lse`. ``full`` is intentionally unsupported: its document rows
    attend across document boundaries, so independent document prefill is invalid.
    """
    if variant == "full":
        raise ValueError(
            "segmented_attention_decomposed does not support variant='full'; "
            "full causal document rows attend across documents"
        )
    if variant not in ("ddi_qf", "blind"):
        raise ValueError(
            "segmented_attention_decomposed supports only variant='ddi_qf' or 'blind'"
        )
    if attention_mask is not None:
        raise ValueError(
            "segmented_attention_decomposed does not accept masks; pass exact unpadded segments"
        )
    _validate_inputs(
        document_q,
        document_k,
        document_v,
        query_q,
        query_k,
        query_v,
        documents,
        query_positions,
        decode_q,
        decode_k,
        decode_v,
        decode_positions,
    )

    document_k_parts: list[torch.Tensor] = []
    document_v_parts: list[torch.Tensor] = []
    document_outputs: list[torch.Tensor] = []
    offset = 0
    for segment in documents:
        length = segment.cache_slots.numel()
        local_q = document_q[offset : offset + length]
        local_k = document_k.index_select(0, segment.cache_slots)
        local_v = document_v.index_select(0, segment.cache_slots)
        document_k_parts.append(local_k)
        document_v_parts.append(local_v)
        document_outputs.append(_causal_attention(local_q, local_k, local_v))
        offset += length

    document_out = torch.cat(document_outputs) if document_outputs else document_q.clone()
    query_out = _merge_tail_attention(
        query_q,
        document_k_parts if variant == "ddi_qf" else (),
        document_v_parts if variant == "ddi_qf" else (),
        query_k,
        query_v,
        causal_prefix=0,
    )
    if decode_q is None:
        return SegmentedAttentionOutput(document_out, query_out, None)

    decode_out = _merge_tail_attention(
        decode_q,
        document_k_parts if variant == "ddi_qf" else (),
        document_v_parts if variant == "ddi_qf" else (),
        torch.cat((query_k, decode_k)),
        torch.cat((query_v, decode_v)),
        causal_prefix=query_q.shape[0],
    )
    return SegmentedAttentionOutput(document_out, query_out, decode_out)


def _masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allow: torch.Tensor,
) -> torch.Tensor:
    """Apply an explicit boolean mask without expanding GQA K/V heads."""
    return _attention_partial(q, k, v, allow)[0]


def _causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute one causal component without constructing a global segmented mask."""
    allow = torch.tril(torch.ones(q.shape[0], k.shape[0], dtype=torch.bool, device=q.device))
    return _attention_partial(q, k, v, allow)[0]


def _merge_tail_attention(
    q: torch.Tensor,
    document_k_parts: Sequence[torch.Tensor],
    document_v_parts: Sequence[torch.Tensor],
    tail_k: torch.Tensor,
    tail_v: torch.Tensor,
    *,
    causal_prefix: int,
) -> torch.Tensor:
    """Merge document components with a causal query/decode-tail component."""
    if q.shape[0] == 0:
        return q.clone()
    partials = []
    for document_k, document_v in zip(document_k_parts, document_v_parts, strict=True):
        document_allow = torch.ones(
            q.shape[0], document_k.shape[0], dtype=torch.bool, device=q.device
        )
        partials.append(_attention_partial(q, document_k, document_v, document_allow))
    tail_columns = torch.arange(tail_k.shape[0], device=q.device)
    tail_rows = torch.arange(q.shape[0], device=q.device).unsqueeze(1)
    tail_allow = tail_columns <= causal_prefix + tail_rows
    partials.append(_attention_partial(q, tail_k, tail_v, tail_allow))
    return merge_lse(partials)


def _attention_partial(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independently normalized GQA attention output and its per-row LSE."""
    rows, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    groups = q_heads // kv_heads
    q_grouped = q.reshape(rows, kv_heads, groups, head_dim)
    scores = torch.einsum("rhgd,shd->rhgs", q_grouped, k) * (head_dim**-0.5)
    scores = scores.masked_fill(~allow.unsqueeze(1).unsqueeze(1), float("-inf"))
    lse_grouped = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse_grouped.unsqueeze(-1))
    out = torch.einsum("rhgs,shd->rhgd", probs, v).reshape(rows, q_heads, head_dim)
    return out, lse_grouped.reshape(rows, q_heads)


def _validate_inputs(
    document_q: torch.Tensor,
    document_k: torch.Tensor,
    document_v: torch.Tensor,
    query_q: torch.Tensor,
    query_k: torch.Tensor,
    query_v: torch.Tensor,
    documents: Sequence[DocumentSegment],
    query_positions: torch.Tensor,
    decode_q: torch.Tensor | None,
    decode_k: torch.Tensor | None,
    decode_v: torch.Tensor | None,
    decode_positions: torch.Tensor | None,
) -> None:
    tensors = (document_q, document_k, document_v, query_q, query_k, query_v)
    if any(tensor.dim() != 3 for tensor in tensors):
        raise ValueError("Q, K, and V must be unpadded token-major [rows, heads, dim] tensors")
    if document_k.shape != document_v.shape or query_k.shape != query_v.shape:
        raise ValueError("K and V shapes must match")
    if document_q.shape[2] != document_k.shape[2] or query_q.shape[2] != query_k.shape[2]:
        raise ValueError("Q, K, and V head dimensions must match")
    if document_k.shape[1] != query_k.shape[1] or document_q.shape[1] != query_q.shape[1]:
        raise ValueError("document and query head counts must match")
    if document_q.shape[1] % document_k.shape[1]:
        raise ValueError("Q head count must be a positive multiple of KV head count (GQA)")
    if any(
        not tensor.is_floating_point() or not torch.isfinite(tensor).all() for tensor in tensors
    ):
        raise ValueError("Q, K, and V must be finite floating-point tensors")
    if (
        len({tensor.device for tensor in tensors}) != 1
        or len({tensor.dtype for tensor in tensors}) != 1
    ):
        raise ValueError("Q, K, and V must share one device and dtype")
    _validate_positions(query_positions, query_q.shape[0], "query_positions")
    if decode_q is None and any(
        item is not None for item in (decode_k, decode_v, decode_positions)
    ):
        raise ValueError(
            "decode_q, decode_k, decode_v, and decode_positions must be provided together"
        )
    if decode_q is not None:
        if decode_k is None or decode_v is None or decode_positions is None:
            raise ValueError(
                "decode_q, decode_k, decode_v, and decode_positions must be provided together"
            )
        decode_tensors = (decode_q, decode_k, decode_v)
        if (
            decode_q.dim() != 3
            or decode_q.shape[1:] != query_q.shape[1:]
            or decode_k.dim() != 3
            or decode_v.dim() != 3
            or decode_k.shape[1:] != query_k.shape[1:]
            or decode_v.shape[1:] != query_k.shape[1:]
        ):
            raise ValueError("decode Q, K, and V must match query head geometry")
        if any(
            tensor.device != query_q.device or tensor.dtype != query_q.dtype
            for tensor in decode_tensors
        ):
            raise ValueError("decode Q, K, and V must share query device and dtype")
        if any(not torch.isfinite(tensor).all() for tensor in decode_tensors):
            raise ValueError("decode Q, K, and V must be finite")
        _validate_positions(decode_positions, decode_q.shape[0], "decode_positions")
        if (
            query_positions.numel()
            and decode_positions.numel()
            and decode_positions[0] <= query_positions[-1]
        ):
            raise ValueError("decode_positions must be strictly after query_positions")
    total_doc = sum(segment.cache_slots.numel() for segment in documents)
    if document_q.shape[0] != total_doc or document_k.shape[0] != total_doc:
        raise ValueError(
            "segments must exactly cover document Q rows and packed document K/V slots"
        )
    slots: list[torch.Tensor] = []
    for segment in documents:
        if segment.cache_slots.dtype != torch.long or segment.cache_slots.dim() != 1:
            raise ValueError("cache_slots must be one-dimensional torch.long tensors")
        if segment.cache_slots.device != document_k.device:
            raise ValueError("cache_slots must be on the Q/K/V device")
        _validate_positions(
            segment.logical_positions, segment.cache_slots.numel(), "logical_positions"
        )
        if segment.logical_positions.device != document_k.device:
            raise ValueError("logical_positions must be on the Q/K/V device")
        slots.append(segment.cache_slots)
    all_slots = (
        torch.cat(slots) if slots else torch.empty(0, dtype=torch.long, device=document_k.device)
    )
    if not torch.equal(
        torch.sort(all_slots).values, torch.arange(total_doc, device=document_k.device)
    ):
        raise ValueError(
            "cache_slots must be an exact, duplicate-free cover of packed document K/V"
        )


def _validate_positions(positions: torch.Tensor, length: int, name: str) -> None:
    if positions.dtype != torch.long or positions.dim() != 1 or positions.numel() != length:
        raise ValueError(
            f"{name} must be a one-dimensional torch.long tensor with one value per row"
        )
    if positions.numel() > 1 and not torch.all(positions[1:] > positions[:-1]):
        raise ValueError(
            f"{name} must be strictly increasing; padding or duplicate logical rows are invalid"
        )
