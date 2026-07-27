"""Mask builder — single source of truth for attention masks (issue #3, Phase 0.2).

Variants: `full`, `ddi_qf`, `ddi_strict`, `ddi_bridge_k`, `blind`. Returns a boolean [T, T]
allow-matrix (True = attend). Additive `-inf` conversion is a separate, separately-tested helper.
"""

from __future__ import annotations

from typing import Literal

import torch

Variant = Literal["full", "ddi_qf", "ddi_strict", "ddi_bridge_k", "blind"]


def build_mask(
    doc_lengths: list[int],
    query_len: int,
    answer_len: int,
    variant: Variant,
    bridge_k: int | None = None,
) -> torch.Tensor:
    """Return a boolean [T, T] allow-matrix, where T = sum(doc_lengths) + query_len + answer_len."""
    raise NotImplementedError("issue #3")


def to_additive(allow: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert a boolean allow-matrix to an additive mask (0 = keep, -inf = block)."""
    raise NotImplementedError("issue #3")
