"""flex_attention backend for block-diagonal attention (VIBEQUEUE-21).

Patches Qwen2 attention layers to use `torch.nn.attention.flex_attention`,
which compiles the block-diagonal mask pattern into a block-sparse kernel
that skips masked QK^T blocks entirely. This realizes the speedup that
sdpa cannot — sdpa computes the full dense QK^T then applies the mask,
falling back to the slow math backend when a 4D additive mask is passed.

The block mask is built from `doc_lengths` + `query_len` + `mask_variant`
and cached per (variant, doc_lengths, query_len) so repeated forward passes
with the same shape reuse the compiled mask.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

_flex_compiled: Any = None


def _get_flex_compiled():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention)
    return _flex_compiled


def _make_doc_id_tensor(doc_lengths: list[int], query_len: int, T: int, device: torch.device) -> torch.Tensor:
    """Assign each position a doc index (0..N-1) or N for query."""
    doc_id = torch.empty(T, dtype=torch.long, device=device)
    pos = 0
    for i, dl in enumerate(doc_lengths):
        doc_id[pos : pos + dl] = i
        pos += dl
    if query_len > 0:
        doc_id[pos:] = len(doc_lengths)
    return doc_id


def make_block_mask(
    doc_lengths: list[int],
    query_len: int,
    variant: str,
    device: torch.device = torch.device("cuda"),
) -> Any:
    """Create a flex_attention BlockMask for the DDI mask variant.

    Supports:
    - ``full``: standard causal mask
    - ``ddi_qf``: block-diagonal docs + query sees all docs
    - ``blind``: query sees no docs (docs still block-diagonal among themselves)
    """
    total_doc = sum(doc_lengths)
    T = total_doc + query_len

    if variant == "full":
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return create_block_mask(mask_mod, B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)

    doc_id = _make_doc_id_tensor(doc_lengths, query_len, T, device)
    N = len(doc_lengths)

    if variant == "ddi_qf":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[q_idx]
            k_doc = doc_id[kv_idx]
            q_is_query = q_doc == N
            k_is_query = k_doc == N
            same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query
            q_sees_k = q_is_query & ~k_is_query
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            return same_doc | q_sees_k | q_query_causal
        return create_block_mask(mask_mod, B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)

    if variant == "blind":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[q_idx]
            k_doc = doc_id[kv_idx]
            q_is_query = q_doc == N
            k_is_query = k_doc == N
            same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            return same_doc | q_query_causal
        return create_block_mask(mask_mod, B=1, H=None, Q_LEN=T, KV_LEN=T, device=device)

    raise ValueError(f"flex backend: unknown mask variant {variant!r}")


def patch_model_for_flex(model: torch.nn.Module) -> torch.nn.Module:
    """Patch all Qwen2 attention layers to use flex_attention.

    The model must be loaded with ``attn_implementation='eager'`` first —
    we replace the attention computation entirely, so HF's own attention
    implementation is irrelevant. The DDI mask patch (``patch_model``) is
    still needed so that the ``attention_mask`` kwarg is passed through
    unchanged.

    After patching, each layer's ``forward`` reads ``model._flex_block_mask``.
    When it is a flex_attention ``BlockMask``, the layer uses the compiled
    ``flex_attention`` kernel (fast block-sparse prefill). When it is ``None``
    (decode, or any single-token step), the layer falls back to
    ``scaled_dot_product_attention`` — decode is memory-bound and mask sparsity
    does not help at Q_LEN=1, so flex offers no benefit there.

    Callers set the block_mask by assigning ``model._flex_block_mask = bm``
    before a prefill forward, and reset it to ``None`` before decode.
    """
    import torch.nn.functional as F

    flex_fn = _get_flex_compiled()
    model._flex_block_mask = None  # type: ignore[attr-defined]

    for layer in model.model.layers:
        attn = layer.self_attn

        def _make_flex_forward(attn_module, model_ref):
            def flex_forward(
                hidden_states: torch.Tensor,
                position_embeddings,
                attention_mask=None,
                past_key_value=None,
                cache_position=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_module.head_dim)

                query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_module.layer_idx, cache_kwargs
                    )

                # GQA: expand KV heads to match Q heads
                num_kv_groups = (
                    attn_module.config.num_attention_heads
                    // attn_module.config.num_key_value_heads
                )
                if num_kv_groups > 1:
                    key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
                    value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

                bm = model_ref._flex_block_mask
                if bm is not None:
                    attn_output = flex_fn(query_states, key_states, value_states, block_mask=bm)
                else:
                    # Decode fallback: sdpa with additive mask or causal.
                    # Decode is Q_LEN=1 (memory-bound), so flex's block-sparse
                    # kernel offers no benefit — sdpa is correct and fast here.
                    if attention_mask is not None:
                        attn_output = F.scaled_dot_product_attention(
                            query_states, key_states, value_states, attn_mask=attention_mask
                        )
                    else:
                        attn_output = F.scaled_dot_product_attention(
                            query_states, key_states, value_states, is_causal=True
                        )

                attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, None

            return flex_forward

        attn.forward = _make_flex_forward(attn, model)

    return model
