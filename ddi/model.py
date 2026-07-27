"""HF model wrapper — inject the mask, don't reimplement attention (issue #5, Phase 0.4).

Wraps `AutoModelForCausalLM` and passes a 4D additive mask through the standard `attention_mask`
path. Prefill and decode use HF's own forward and KV cache. Does NOT hand-roll the decoder loop —
that was the archaeology's central bug class.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    raise NotImplementedError("issue #5")


def forward_with_mask(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    additive_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with a caller-supplied 4D additive mask; returns logits."""
    raise NotImplementedError("issue #5")
