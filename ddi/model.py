"""HF model wrapper — inject the mask, don't reimplement attention (issue #5, Phase 0.4).

Wraps `AutoModelForCausalLM` and passes a 4D additive mask through the standard `attention_mask`
path. Prefill and decode use HF's own forward and KV cache. Does NOT hand-roll the decoder loop —
that was the archaeology's central bug class.
"""

from __future__ import annotations

import types

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_ORIGINAL_ATTR = "__ddi_original_update_causal_mask__"

# DDI's own backend names (see ddi.attention.Backend) vs. HF's `attn_implementation` strings.
_HF_ATTN_IMPLEMENTATION = {"fa2": "flash_attention_2"}


def load_model(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    hf_attn_implementation = _HF_ATTN_IMPLEMENTATION.get(
        attn_implementation, attn_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation=hf_attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    patch_model(model)
    return model, tokenizer


def _find_patch_target(model: PreTrainedModel) -> torch.nn.Module:
    """Return the sub-module that owns `_update_causal_mask` (`model` or `model.model`)."""
    if hasattr(model, "_update_causal_mask"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "_update_causal_mask"):
        return inner
    raise AttributeError(
        f"{type(model).__name__} exposes no `_update_causal_mask` on itself or `.model`; "
        "the DDI wrapper is not compatible with this model class"
    )


def patch_model(model: PreTrainedModel) -> PreTrainedModel:
    """Install an instance-level `_update_causal_mask` that returns a 4D float mask unchanged."""
    target = _find_patch_target(model)
    original = target._update_causal_mask

    # WHY only float 4D: HF's own path uses bool 4D internally in some code paths, and 2D
    # attention masks (padding masks) must still be lifted to a causal mask by the original.
    # A DDI additive mask is unambiguously float 4D — short-circuit exactly that shape+dtype,
    # delegate everything else so we don't break any of HF's own callers.
    def _ddi_update_causal_mask(self, attention_mask, *args, **kwargs):
        if (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.dim() == 4
            and attention_mask.dtype.is_floating_point
        ):
            return attention_mask
        return original(attention_mask, *args, **kwargs)

    setattr(target, _ORIGINAL_ATTR, original)
    target._update_causal_mask = types.MethodType(_ddi_update_causal_mask, target)
    return model


def unpatch_model(model: PreTrainedModel) -> None:
    """Restore the original `_update_causal_mask` (for tests and cleanup)."""
    target = _find_patch_target(model)
    if not hasattr(target, _ORIGINAL_ATTR):
        raise AttributeError(
            f"{type(model).__name__} is not patched by DDI; nothing to restore"
        )
    # Deleting the instance attribute unshadows the class-level descriptor so
    # `target._update_causal_mask` returns to the fresh-bind-per-access behaviour it had
    # before patching. Directly assigning `original` back would leave an instance attribute in
    # place, which is a different post-state than we started with.
    del target._update_causal_mask
    delattr(target, _ORIGINAL_ATTR)


def forward_with_mask(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    additive_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass with a caller-supplied 4D additive mask; returns logits.

    Optional `position_ids` is a `[batch, T]` long tensor forwarded straight to HF's `forward`
    (Phase 4.1 uses this to run the same prefill under sequential and restart layouts). When
    omitted, HF's default sequential position handling is preserved — existing callers are
    unaffected.
    """
    if additive_mask.dim() != 4:
        raise ValueError(
            f"additive_mask must be 4D [B, H|1, T, T]; got {additive_mask.dim()}D "
            f"shape={tuple(additive_mask.shape)}"
        )
    if not additive_mask.dtype.is_floating_point:
        raise ValueError(
            f"additive_mask must be a floating-point additive mask (0 / -inf); got dtype "
            f"{additive_mask.dtype}. Use ddi.masks.to_additive to convert a bool allow-matrix."
        )
    if position_ids is not None:
        if position_ids.dim() != 2:
            raise ValueError(
                f"position_ids must be 2D [batch, T]; got {position_ids.dim()}D "
                f"shape={tuple(position_ids.shape)}"
            )
        if position_ids.shape != input_ids.shape:
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} must match input_ids shape "
                f"{tuple(input_ids.shape)}"
            )
        if position_ids.dtype != torch.long:
            raise ValueError(
                f"position_ids must be torch.long; got {position_ids.dtype}"
            )
    out = model(
        input_ids=input_ids,
        attention_mask=additive_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return out.logits
