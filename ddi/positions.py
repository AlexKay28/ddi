"""Position-ID builder — axis P of the DDI grid (issue #22, Phase 4.1).

Two layouts:

- `sequential` — HF's default. Included so a call site can pass explicit position IDs without
  changing behaviour (useful for the runner, which builds both layouts and forwards the same way).
- `restart` — each document's positions start at 0. This is what makes a document's KV block a
  pure function of its content: identical bytes give identical RoPE-encoded keys/values regardless
  of where in the retrieval order the document lands. Unlocks C4's per-document cache (Phase 4.2).

See RESEARCH_PLAN §6 (axis P) and §8 Phase 4.1.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch

Layout = Literal["sequential", "restart"]

_LAYOUTS: tuple[str, ...] = get_args(Layout)


def build_positions(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    layout: Layout,
) -> torch.Tensor:
    """Return a [T] long tensor of position IDs, T = sum(doc_lengths) + query_len + answer_len.

    - sequential: [0, 1, 2, ..., T-1] (matches HF's default).
    - restart: each document's positions restart at 0 ([0, 1, ..., L_i - 1]); the query+answer
      block continues from `max(doc_lengths)` (or 0 if there are no docs).
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {_LAYOUTS}")
    if query_len < 0 or answer_len < 0:
        raise ValueError(
            f"query_len and answer_len must be >= 0, got {query_len}, {answer_len}"
        )
    for i, length in enumerate(doc_lengths):
        if length < 1:
            raise ValueError(f"doc_lengths[{i}] = {length} < 1; each document must be non-empty")

    doc_total = sum(doc_lengths)
    total = doc_total + query_len + answer_len

    if layout == "sequential":
        return torch.arange(total, dtype=torch.long)

    # layout == "restart"
    positions = torch.empty(total, dtype=torch.long)
    offset = 0
    for length in doc_lengths:
        positions[offset : offset + length] = torch.arange(length, dtype=torch.long)
        offset += length

    # WHY max(doc_lengths): the query+answer block must sit at absolute positions strictly beyond
    # every document token so query→doc RoPE relative distances are well-defined and stable across
    # requests that share the same *set* of docs. Using max(L_i) is the smallest such offset —
    # anything less would make some query rows RoPE-collide with doc rows they need to attend to.
    query_start = max(doc_lengths) if doc_lengths else 0
    tail_len = query_len + answer_len
    positions[doc_total : doc_total + tail_len] = torch.arange(
        query_start, query_start + tail_len, dtype=torch.long
    )
    return positions
