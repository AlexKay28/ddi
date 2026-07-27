"""Shared runner helpers.

Formulas that must stay bit-identical across runners live here — Phase 1.4 baseline
and Phase 2.1 train-free sweep both compute the *same* answer-only cross-entropy, so
that `Δ` (delta) between them means "mask family effect" and nothing else.
"""

from __future__ import annotations

import torch


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token-shifted cross-entropy with `-100` ignored.

    Predicts position `t` from logits at position `t-1`. `labels` is expected to be a
    `[batch, seq]` tensor with `-100` outside the answer span (see
    `ddi.metrics.make_answer_labels`) so cross-entropy is restricted to answer tokens.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
