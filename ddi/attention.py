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


def get_backend(name: Backend) -> AttentionBackend:
    raise NotImplementedError("issue #6")
