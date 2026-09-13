"""HF model wrapper — inject the mask, don't reimplement attention (issue #5, Phase 0.4).

Wraps `AutoModelForCausalLM` and passes a 4D additive mask through the standard `attention_mask`
path. Prefill and decode use HF's own forward and KV cache. Does NOT hand-roll the decoder loop —
that was the archaeology's central bug class.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_ORIGINAL_ATTR = "__ddi_original_update_causal_mask__"

_DDI_BACKENDS = {"sdpa", "fa2", "flex", "eager"}

# DDI's own backend names (see ddi.attention.Backend) vs. HF's `attn_implementation` strings.
_HF_ATTN_IMPLEMENTATION = {"fa2": "flash_attention_2", "flex": "eager"}


def _resolve_offline_snapshot(name: str) -> str:
    """Resolve a HF repo id to a local snapshot dir when `HF_HUB_OFFLINE=1`.

    Some patched transformers builds (observed on Yandex-provided containers) call
    `model_info(<repo_id>)` from inside `AutoTokenizer.from_pretrained` regardless of
    `HF_HUB_OFFLINE`, which fails with `OfflineModeIsEnabled` even when the model is
    fully cached. The upstream code short-circuits that path when the argument is a
    local directory (`_is_local` True), so we resolve the id → local snapshot dir up
    front. Silently returns `name` unchanged if we can't find a cache — the downstream
    error will be equivalent to the current behaviour.
    """
    if not os.environ.get("HF_HUB_OFFLINE"):
        return name
    if os.path.isdir(name):  # Already a local path — nothing to do.
        return name

    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{name.replace('/', '--')}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return name

    # Prefer the ref that HF's own from_pretrained would resolve (`refs/main`) so the
    # snapshot we pick matches what an online load would have picked.
    ref = cache_dir / "refs" / "main"
    if ref.is_file():
        snap = snapshots / ref.read_text().strip()
        if snap.is_dir():
            return str(snap)
    # Fall back to any snapshot dir (yt-fetch writes exactly one, named `yt-cache`).
    for entry in sorted(snapshots.iterdir()):
        if entry.is_dir():
            return str(entry)
    return name


def load_model(
    name: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    rope_theta: float | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    hf_attn_implementation = _HF_ATTN_IMPLEMENTATION.get(
        attn_implementation, attn_implementation
    )
    resolved = _resolve_offline_snapshot(name)
    # rope_theta override (Sprint D, VIBEQUEUE-28): ABF-style Adjusted Base
    # Frequency. Mutate the config BEFORE from_pretrained so HF's rotary
    # embedding module initializes with the new base. None = model default.
    kwargs: dict = {"torch_dtype": dtype, "attn_implementation": hf_attn_implementation}
    if rope_theta is not None:
        cfg = AutoConfig.from_pretrained(resolved)
        cfg.rope_theta = rope_theta
        kwargs["config"] = cfg
    model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    patch_model(model)
    if attn_implementation == "flex":
        from ddi.flex_attention import patch_model_for_flex
        patch_model_for_flex(model)
    return model, tokenizer


# Two transformers APIs to support (#30):
#   - <4.53: per-model `_update_causal_mask` method (patched per-instance in `patch_model`).
#   - >=4.53: module-level `transformers.masking_utils.create_causal_mask` (patched globally
#     at import time — the per-model hook was removed in 4.53).
# Same short-circuit semantics on both: 4D float additive mask → pass through unchanged.
try:
    import transformers.masking_utils as _masking_utils  # type: ignore[import-not-found]

    _HAS_MASKING_UTILS = True
except ImportError:
    _masking_utils = None  # type: ignore[assignment]
    _HAS_MASKING_UTILS = False


def _is_ddi_additive_mask(m) -> bool:
    """4D float mask = DDI's additive mask, safe to pass through unchanged.

    HF's own path uses bool 4D internally in some code paths, and 2D attention masks
    (padding masks) must still be lifted to a causal mask by the original. A DDI
    additive mask is unambiguously float 4D — short-circuit exactly that shape+dtype,
    delegate everything else so we don't break any of HF's own callers.
    """
    return (
        isinstance(m, torch.Tensor) and m.dim() == 4 and m.dtype.is_floating_point
    )


# ---- >=4.53 path: patch masking_utils.create_causal_mask once at import ------------------

_ORIGINAL_CREATE_CAUSAL_MASK = None


def _install_create_causal_mask_hook() -> None:
    """Idempotently swap `create_causal_mask` for a DDI-aware version. >=4.53 only."""
    global _ORIGINAL_CREATE_CAUSAL_MASK
    if not _HAS_MASKING_UTILS or _ORIGINAL_CREATE_CAUSAL_MASK is not None:
        return
    _ORIGINAL_CREATE_CAUSAL_MASK = _masking_utils.create_causal_mask

    def _ddi_create_causal_mask(*args, **kwargs):
        m = kwargs.get("attention_mask")
        if m is None and args:
            # Positional call: `create_causal_mask(config, ..., attention_mask, ...)`.
            # Scan positional args for a 4D float tensor rather than guess the signature.
            for a in args:
                if _is_ddi_additive_mask(a):
                    return a
        if _is_ddi_additive_mask(m):
            return m
        return _ORIGINAL_CREATE_CAUSAL_MASK(*args, **kwargs)

    _masking_utils.create_causal_mask = _ddi_create_causal_mask


def _uninstall_create_causal_mask_hook() -> None:
    """Restore the original module-level function. For tests + cleanup."""
    global _ORIGINAL_CREATE_CAUSAL_MASK
    if _ORIGINAL_CREATE_CAUSAL_MASK is None or not _HAS_MASKING_UTILS:
        return
    _masking_utils.create_causal_mask = _ORIGINAL_CREATE_CAUSAL_MASK
    _ORIGINAL_CREATE_CAUSAL_MASK = None


# Install the >=4.53 hook once at import — it's module-scoped so per-model patching is a no-op
# on that path. The <4.53 hook is per-instance and stays in `patch_model` below.
_install_create_causal_mask_hook()


# ---- <4.53 path: per-model `_update_causal_mask` patch -----------------------------------

def _find_patch_target(model: PreTrainedModel) -> torch.nn.Module | None:
    """Return the sub-module that owns `_update_causal_mask`, or None on >=4.53."""
    if hasattr(model, "_update_causal_mask"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "_update_causal_mask"):
        return inner
    return None


def patch_model(model: PreTrainedModel) -> PreTrainedModel:
    """On <4.53 install a per-instance `_update_causal_mask`; on >=4.53 a no-op (global hook)."""
    target = _find_patch_target(model)
    if target is None:
        if _HAS_MASKING_UTILS:
            # >=4.53 — the global create_causal_mask hook is already installed; nothing to do.
            return model
        raise AttributeError(
            f"{type(model).__name__} exposes no `_update_causal_mask` on itself or `.model`, "
            "and transformers.masking_utils is not available either — the DDI wrapper is not "
            "compatible with this transformers version"
        )
    original = target._update_causal_mask

    def _ddi_update_causal_mask(self, attention_mask, *args, **kwargs):
        if _is_ddi_additive_mask(attention_mask):
            return attention_mask
        return original(attention_mask, *args, **kwargs)

    setattr(target, _ORIGINAL_ATTR, original)
    target._update_causal_mask = types.MethodType(_ddi_update_causal_mask, target)
    return model


def unpatch_model(model: PreTrainedModel) -> None:
    """Restore the original `_update_causal_mask` (for tests and cleanup). No-op on >=4.53."""
    target = _find_patch_target(model)
    if target is None or not hasattr(target, _ORIGINAL_ATTR):
        return
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
