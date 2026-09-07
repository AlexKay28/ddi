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

from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_flex_compiled: Any = None
_CUDA_DEVICE = torch.device("cuda")


def _get_flex_compiled():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention)
    return _flex_compiled


def _make_doc_id_tensor(
    doc_lengths: list[int], query_len: int, T: int, device: torch.device
) -> torch.Tensor:
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
    device: torch.device = _CUDA_DEVICE,
    block_size: int | tuple[int, int] = 128,
) -> Any:
    """Create a flex_attention BlockMask for the DDI mask variant.

    Supports:
    - ``full``: standard causal mask
    - ``ddi_qf``: block-diagonal docs + query sees all docs
    - ``blind``: query sees no docs (docs still block-diagonal among themselves)
    """
    total_doc = sum(doc_lengths)
    T = total_doc + query_len

    doc_id = _make_doc_id_tensor(doc_lengths, query_len, T, device)
    N = len(doc_lengths)

    if variant == "full":
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
    elif variant == "ddi_qf":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[q_idx]
            k_doc = doc_id[kv_idx]
            q_is_query = q_doc == N
            k_is_query = k_doc == N
            # Documents remain causal internally. Omitting this predicate made
            # the flex path bidirectional within each document, unlike the
            # canonical `ddi.masks.build_mask` semantics.
            same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query & (q_idx >= kv_idx)
            q_sees_k = q_is_query & ~k_is_query
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            return same_doc | q_sees_k | q_query_causal
    elif variant == "blind":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[q_idx]
            k_doc = doc_id[kv_idx]
            q_is_query = q_doc == N
            k_is_query = k_doc == N
            same_doc = (q_doc == k_doc) & ~q_is_query & ~k_is_query & (q_idx >= kv_idx)
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            return same_doc | q_query_causal
    else:
        raise ValueError(f"flex backend: unknown mask variant {variant!r}")

    q_block, kv_block = (
        (block_size, block_size) if isinstance(block_size, int) else block_size
    )
    q_blocks = (T + q_block - 1) // q_block
    kv_blocks = (T + kv_block - 1) // kv_block

    # `create_block_mask` evaluates mask_mod densely before it compresses the
    # result. At N=25,L=2048 that temporary is ~51k^2 elements and OOMs an
    # 80GB A100. Build the sparse block adjacency directly, then retain
    # mask_mod for exact token-level causal boundaries inside each candidate.
    segments_by_block: list[set[int]] = []
    for block in range(kv_blocks):
        start = block * kv_block
        end = min(start + kv_block, T)
        segments_by_block.append(set(doc_id[start:end].tolist()))

    candidates: list[list[int]] = []
    for q_block_idx in range(q_blocks):
        q_start = q_block_idx * q_block
        q_end = min(q_start + q_block, T)
        q_segments = set(doc_id[q_start:q_end].tolist())
        allowed_blocks: list[int] = []
        for kv_block_idx, kv_segments in enumerate(segments_by_block):
            kv_start = kv_block_idx * kv_block
            if variant == "full":
                allow = kv_start < q_end
            elif N in q_segments and variant == "ddi_qf":
                allow = True
            elif N in q_segments:  # blind query block
                allow = N in kv_segments
            else:
                allow = bool((q_segments & kv_segments) - {N})
            if allow:
                allowed_blocks.append(kv_block_idx)
        candidates.append(allowed_blocks)

    kv_num_blocks = torch.tensor(
        [[[len(row) for row in candidates]]], dtype=torch.int32, device=device
    )
    kv_indices = torch.zeros(
        # `BlockMask.from_kv_blocks` uses this trailing dimension as the
        # number of addressable KV blocks while transposing the adjacency;
        # it must be the global block count, not merely the largest row.
        (1, 1, q_blocks, kv_blocks), dtype=torch.int32, device=device
    )
    for block, row in enumerate(candidates):
        kv_indices[0, 0, block, : len(row)] = torch.tensor(row, device=device)
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=(q_block, kv_block),
        mask_mod=mask_mod,
    )
    # torch 2.13 validates the exact sequence shape at flex invocation; 2.5
    # lacks this method but cannot compile flex under the locked Python 3.13 runtime.
    return block_mask._adjust(T, T) if hasattr(block_mask, "_adjust") else block_mask


def make_batched_block_mask(
    doc_lengths: list[list[int]],
    query_lens: list[int],
    variant: str,
    device: torch.device = _CUDA_DEVICE,
    block_size: int | tuple[int, int] = 128,
) -> Any:
    """Create one padded BlockMask with a distinct DDI layout per batch item."""
    if len(doc_lengths) != len(query_lens) or not doc_lengths:
        raise ValueError("doc_lengths and query_lens must be non-empty and have equal length")

    seq_lens = [
        sum(lengths) + query_len
        for lengths, query_len in zip(doc_lengths, query_lens, strict=True)
    ]
    T = max(seq_lens)
    B = len(doc_lengths)
    q_block, kv_block = (
        (block_size, block_size) if isinstance(block_size, int) else block_size
    )
    q_blocks = (T + q_block - 1) // q_block
    kv_blocks = (T + kv_block - 1) // kv_block

    # -1 denotes padding and -2 denotes a query; document ids are local to each sample.
    doc_id = torch.full((B, T), -1, dtype=torch.long, device=device)
    for b, (lengths, query_len) in enumerate(zip(doc_lengths, query_lens, strict=True)):
        pos = 0
        for doc, length in enumerate(lengths):
            doc_id[b, pos : pos + length] = doc
            pos += length
        doc_id[b, pos : pos + query_len] = -2

    if variant == "full":
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
    elif variant == "ddi_qf":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[b, q_idx]
            k_doc = doc_id[b, kv_idx]
            q_is_query = q_doc == -2
            k_is_query = k_doc == -2
            same_doc = (q_doc == k_doc) & (q_doc >= 0) & (q_idx >= kv_idx)
            q_sees_doc = q_is_query & (k_doc >= 0)
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            padding_self = (q_doc == -1) & (k_doc == -1) & (q_idx == kv_idx)
            return same_doc | q_sees_doc | q_query_causal | padding_self
    elif variant == "blind":
        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_id[b, q_idx]
            k_doc = doc_id[b, kv_idx]
            q_is_query = q_doc == -2
            k_is_query = k_doc == -2
            same_doc = (q_doc == k_doc) & (q_doc >= 0) & (q_idx >= kv_idx)
            q_query_causal = q_is_query & k_is_query & (q_idx >= kv_idx)
            padding_self = (q_doc == -1) & (k_doc == -1) & (q_idx == kv_idx)
            return same_doc | q_query_causal | padding_self
    else:
        raise ValueError(f"flex backend: unknown mask variant {variant!r}")

    kv_num_blocks = torch.zeros((B, 1, q_blocks), dtype=torch.int32, device=device)
    kv_indices = torch.zeros((B, 1, q_blocks, kv_blocks), dtype=torch.int32, device=device)
    for b in range(B):
        segments_by_block = [
            set(doc_id[b, start : min(start + kv_block, T)].tolist())
            for start in range(0, T, kv_block)
        ]
        for q_block_idx in range(q_blocks):
            q_start = q_block_idx * q_block
            q_segments = set(doc_id[b, q_start : min(q_start + q_block, T)].tolist())
            candidates: list[int] = []
            for kv_block_idx, kv_segments in enumerate(segments_by_block):
                if variant == "full":
                    allow = kv_block_idx * kv_block < q_start + q_block
                elif -2 in q_segments and variant == "ddi_qf":
                    allow = True
                elif -2 in q_segments:
                    allow = -2 in kv_segments
                else:
                    allow = bool((q_segments & kv_segments) - {-2})
                # Padded rows must retain a diagonal candidate to stay finite.
                if -1 in q_segments and -1 in kv_segments:
                    allow = True
                if allow:
                    candidates.append(kv_block_idx)
            kv_num_blocks[b, 0, q_block_idx] = len(candidates)
            kv_indices[b, 0, q_block_idx, : len(candidates)] = torch.tensor(
                candidates, dtype=torch.int32, device=device
            )
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=(q_block, kv_block),
        mask_mod=mask_mod,
    )
    return block_mask._adjust(T, T) if hasattr(block_mask, "_adjust") else block_mask


def patch_model_for_flex(model: torch.nn.Module) -> torch.nn.Module:
    """Patch all Qwen2 attention layers to use flex_attention.

    The model must be loaded with ``attn_implementation='eager'`` first —
    we replace the attention computation entirely, so HF's own attention
    implementation is irrelevant. The DDI mask patch (``patch_model``) is
    still needed so that the ``attention_mask`` kwarg is passed through
    unchanged.

    After patching, each layer's ``forward`` reads ``model._flex_block_mask``.
    When it is a flex_attention ``BlockMask``, the layer uses the compiled
    ``flex_attention`` kernel for block-sparse prefill. When it is ``None``
    (decode, or any single-token step), the layer falls back to
    ``scaled_dot_product_attention``; the Flex path is not used at Q_LEN=1.

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
                # Transformers >=4.57 passes the layer cache under the plural
                # spelling; retain the older singular argument for compatibility.
                if past_key_value is None:
                    past_key_value = kwargs.get("past_key_values")
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
