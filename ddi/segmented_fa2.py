"""Segmented FlashAttention-2 execution for the HF Qwen2 path (V61 candidate A).

Implements DDI's block-sparse attention with real kernels instead of a dense
additive mask: document rows attend causally within their own document
(``flash_attn_varlen_func`` block-diagonal), query rows attend all document
K/V plus their causal query prefix (one varlen call relying on FA2's verified
bottom-right causal alignment for ``q_len < k_len``), and ``full`` is one
standard causal call. GQA stays native (no K/V head expansion).

This is a bounded execution-contract candidate, not a production default: the
patched layers ignore HF's additive mask while request state is set and fall
back to SDPA otherwise (decode, unpatched callers). Correctness is gated by
``scripts/validate_segmented_parity.py`` against the SDPA reference on the
real checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch

_VARIANTS = ("full", "ddi_qf", "blind")

_FLASH: dict[str, Any] = {}


def _flash_attn_func():
    if "func" not in _FLASH:
        from flash_attn import flash_attn_func

        _FLASH["func"] = flash_attn_func
    return _FLASH["func"]


def _flash_attn_varlen_func():
    if "varlen" not in _FLASH:
        from flash_attn import flash_attn_varlen_func

        _FLASH["varlen"] = flash_attn_varlen_func
    return _FLASH["varlen"]


def set_segmented_state(
    model: torch.nn.Module, doc_lengths: list[int], query_len: int, *, variant: str
) -> None:
    """Arm segmented execution for the next prefill forward; clear it for decode."""
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}")
    if not doc_lengths or any(length < 1 for length in doc_lengths):
        raise ValueError("doc_lengths must be positive and non-empty")
    if query_len < 1:
        raise ValueError("query_len must be positive")
    model._segmented_fa2_state = {
        "doc_lengths": list(doc_lengths),
        "query_len": int(query_len),
        "variant": variant,
    }


def clear_segmented_state(model: torch.nn.Module) -> None:
    model._segmented_fa2_state = None


def _cu_seqlens(doc_lengths: list[int], device: torch.device) -> torch.Tensor:
    cu = torch.zeros(len(doc_lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(doc_lengths, dtype=torch.int32, device=cu.device).cumsum(0)
    return cu


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """[1, heads, tokens, dim] -> [tokens, heads, dim] contiguous for FA2 varlen."""
    return x.squeeze(0).transpose(0, 1).contiguous()


def _unflatten(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def _segmented_prefill(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: dict[str, Any], scale: float
) -> torch.Tensor:
    """Segmented prefill on the full-stream K/V; returns [1, heads, T, dim]."""
    doc_lengths = state["doc_lengths"]
    doc_total = sum(doc_lengths)
    query_len = state["query_len"]
    tokens = q.shape[2]
    if tokens != doc_total + query_len:
        raise ValueError("prefill rows do not match the armed request state")
    variant = state["variant"]
    varlen = _flash_attn_varlen_func()

    if variant == "full":
        func = _flash_attn_func()
        # flash_attn_func expects [batch, seqlen, heads, dim] (batch kept);
        # HF tensors are [batch, heads, seqlen, dim].
        out = func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            causal=True, softmax_scale=scale,
        )
        return out.transpose(1, 2)

    doc_q = _flatten(q[:, :, :doc_total])
    doc_k = _flatten(k[:, :, :doc_total])
    doc_v = _flatten(v[:, :, :doc_total])
    cu = _cu_seqlens(doc_lengths, q.device)
    doc_out = varlen(
        doc_q,
        doc_k,
        doc_v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(doc_lengths),
        max_seqlen_k=max(doc_lengths),
        causal=True,
        softmax_scale=scale,
    )

    query_q = _flatten(q[:, :, doc_total:])
    if variant == "ddi_qf":
        # Bottom-right causal alignment (verified): query row i attends all
        # doc_total keys plus query keys [0..i].
        query_out = varlen(
            query_q,
            _flatten(k),
            _flatten(v),
            cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.tensor([0, tokens], dtype=torch.int32, device=q.device),
            max_seqlen_q=query_len,
            max_seqlen_k=tokens,
            causal=True,
            softmax_scale=scale,
        )
    else:  # blind: the query segment attends only itself
        query_out = varlen(
            query_q,
            _flatten(k[:, :, doc_total:]),
            _flatten(v[:, :, doc_total:]),
            cu_seqlens_q=torch.tensor([0, query_len], dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.tensor([0, query_len], dtype=torch.int32, device=q.device),
            max_seqlen_q=query_len,
            max_seqlen_k=query_len,
            causal=True,
            softmax_scale=scale,
        )
    return _unflatten(torch.cat([doc_out, query_out], dim=0))


def patch_model_for_segmented_fa2(model: torch.nn.Module) -> torch.nn.Module:
    """Patch all Qwen2 attention layers for segmented FA2 execution.

    The model must be loaded with ``attn_implementation='eager'``; the patched
    forward replaces attention entirely. While ``model._segmented_fa2_state``
    is armed, prefill runs the segmented kernels; otherwise (decode, or any
    caller that did not arm state) the layer falls back to SDPA with the
    additive mask HF passed in, exactly like the Flex patch.
    """
    import torch.nn.functional as F
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    model._segmented_fa2_state = None  # type: ignore[attr-defined]

    for layer in model.model.layers:
        attn = layer.self_attn

        def _make_segmented_forward(attn_module, model_ref):
            def segmented_forward(
                hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
                **kwargs,
            ):
                if past_key_value is None:
                    past_key_value = kwargs.get("past_key_values")
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                from ddi.deferred_rope import is_deferred_rope
                if is_deferred_rope(model_ref):
                    query_states, _ = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin)
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_module.layer_idx, cache_kwargs
                    )

                state = getattr(model_ref, "_segmented_fa2_state", None)
                scale = float(getattr(attn_module, "scaling", attn_module.head_dim**-0.5))
                if state is not None and query_states.shape[2] > 1:
                    attn_output = _segmented_prefill(
                        query_states, key_states, value_states, state, scale
                    ).to(query_states.dtype)
                else:
                    # Decode / unarmed fallback: SDPA needs matching head counts,
                    # so expand native-GQA K/V exactly like the Flex patch does.
                    num_kv_groups = (
                        attn_module.config.num_attention_heads
                        // attn_module.config.num_key_value_heads
                    )
                    sdpa_k, sdpa_v = key_states, value_states
                    if num_kv_groups > 1:
                        sdpa_k = key_states.repeat_interleave(num_kv_groups, dim=1)
                        sdpa_v = value_states.repeat_interleave(num_kv_groups, dim=1)
                    if attention_mask is not None:
                        attn_output = F.scaled_dot_product_attention(
                            query_states,
                            sdpa_k,
                            sdpa_v,
                            attn_mask=attention_mask,
                            scale=scale,
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            query_states, sdpa_k, sdpa_v, is_causal=True, scale=scale
                        )

                attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
                attn_output = attn_module.o_proj(attn_output)
                return attn_output, None

            return segmented_forward

        attn.forward = _make_segmented_forward(attn, model)

    return model
