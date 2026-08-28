"""Mask builder — single source of truth for attention masks (issue #3, Phase 0.2).

Variants: `full`, `ddi_qf`, `ddi_strict`, `ddi_bridge_k`, `blind`. Returns a boolean [T, T]
allow-matrix (True = attend). Additive `-inf` conversion is a separate, separately-tested helper.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch

Variant = Literal["full", "ddi_qf", "ddi_strict", "ddi_bridge_k", "blind"]

_VARIANTS: tuple[str, ...] = get_args(Variant)


def build_mask(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    variant: Variant,
    bridge_k: int | None = None,
) -> torch.Tensor:
    """Return a boolean [T, T] allow-matrix, where T = sum(doc_lengths) + query_len + answer_len."""
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    if query_len < 0 or answer_len < 0:
        raise ValueError(f"query_len and answer_len must be >= 0, got {query_len}, {answer_len}")
    for i, length in enumerate(doc_lengths):
        if length < 1:
            raise ValueError(f"doc_lengths[{i}] = {length} < 1; each document must be non-empty")
    if variant == "ddi_bridge_k":
        if bridge_k is None:
            raise ValueError("variant='ddi_bridge_k' requires bridge_k")
        if bridge_k < 1 or bridge_k > query_len:
            raise ValueError(
                f"bridge_k must satisfy 1 <= bridge_k <= query_len={query_len}, got {bridge_k}"
            )
    elif bridge_k is not None:
        raise ValueError(f"bridge_k is only valid for variant='ddi_bridge_k', not {variant!r}")

    doc_total = sum(doc_lengths)
    total = doc_total + query_len + answer_len

    # Row / column index grids used by every branch below.
    idx = torch.arange(total)
    rows = idx.unsqueeze(1)  # [T, 1]
    cols = idx.unsqueeze(0)  # [1, T]
    causal = cols <= rows  # lower-triangular incl. diagonal

    if variant == "full":
        return causal

    # Region masks — cheap to compute once and reuse across variants.
    is_doc_row = rows < doc_total
    is_doc_col = cols < doc_total
    is_query_row = (rows >= doc_total) & (rows < doc_total + query_len)
    is_query_col = (cols >= doc_total) & (cols < doc_total + query_len)
    is_answer_row = rows >= doc_total + query_len

    # Per-document block id; non-doc positions get a sentinel that we mask away below rather
    # than a "matches nothing" id, because torch's `==` treats sentinel-to-sentinel as equal.
    block_id = torch.full((total,), -1, dtype=torch.long)
    offset = 0
    for b, length in enumerate(doc_lengths):
        block_id[offset : offset + length] = b
        offset += length
    same_block = block_id.unsqueeze(1) == block_id.unsqueeze(0)

    # Restricting to (doc_row, doc_col) both keeps the sentinel matches out and encodes the
    # "block-diagonal only within the doc region" semantics in one AND.
    doc_to_doc = is_doc_row & is_doc_col & same_block & causal

    # N=1 collapse: DDI's isolation constraints (cross-doc blocking, SEP-bridged query→doc
    # gating) exist to prevent leakage *between* documents. With a single document there is
    # nothing to isolate, so the qf/strict/bridge_k family degenerates to `full` — this is the
    # correctness gate (§7: 1.00x at N=1) checked at the mask level in tests.
    if variant in ("ddi_qf", "ddi_strict", "ddi_bridge_k") and len(doc_lengths) <= 1:
        return causal

    # Query→doc allowance depends on the variant; other query and answer patterns are shared.
    if variant == "ddi_qf":
        query_sees_doc = is_query_row & is_doc_col
    elif variant == "ddi_strict":
        # Only the last query row bridges into the docs (the "SEP trick").
        last_query_row_idx = doc_total + query_len - 1
        query_sees_doc = (rows == last_query_row_idx) & is_query_row & is_doc_col
    elif variant == "ddi_bridge_k":
        # Last `bridge_k` rows of the query bridge into the docs; interpolates strict ↔ qf.
        assert bridge_k is not None  # narrowed by earlier validation
        first_bridge_row = doc_total + query_len - bridge_k
        query_sees_doc = (rows >= first_bridge_row) & is_query_row & is_doc_col
    else:  # variant == "blind"
        query_sees_doc = torch.zeros_like(causal)

    query_to_query = is_query_row & is_query_col & causal

    # Answer sees the same docs the query-block sees as a whole (union across query rows),
    # sees the whole query, and is causal within itself. For `blind` that union is empty.
    if variant == "blind":
        answer_sees_doc = torch.zeros_like(causal)
    else:
        answer_sees_doc = is_answer_row & is_doc_col
    answer_to_query = is_answer_row & is_query_col
    answer_to_answer = is_answer_row & causal & ~is_doc_col & ~is_query_col

    return (
        doc_to_doc
        | query_sees_doc
        | query_to_query
        | answer_sees_doc
        | answer_to_query
        | answer_to_answer
    )


def to_additive(allow: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert a boolean allow-matrix to an additive mask (0 = keep, -inf = block)."""
    if allow.dtype != torch.bool:
        raise ValueError(f"expected bool allow-matrix, got {allow.dtype}")
    # Build additive form as -inf then zero-out the allowed entries. Using masked_fill on a
    # pre-filled -inf tensor keeps the -inf representable in bf16 (finfo(bf16).min → -inf).
    out = torch.full(allow.shape, float("-inf"), dtype=dtype, device=allow.device)
    return out.masked_fill(allow, 0.0)
