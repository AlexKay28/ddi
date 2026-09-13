"""Query-driven token scoring and selection for selective recomputation.

Inspired by ProphetKV (arXiv 2602.02579): run a cheap query-only forward over
cached KV, extract per-token attention weights, then select the top-p% most
query-relevant tokens to recompute with full cross-document attention.

The "crowding-out effect" identified by ProphetKV: globally-salient but
query-irrelevant tokens waste the recomputation budget.  Query-driven
selection ensures the budget goes to the tokens the query actually needs.

Usage::

    from ddi.scoring import score_tokens, select_top_k

    indices = score_tokens(model, query_ids, entries, top_p=0.2)
    # indices: list[set[int]] — one set per document, token positions to recompute
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ddi.cache import CacheEntry


def score_tokens(
    model: PreTrainedModel,
    query_ids: torch.Tensor,
    entries: list[CacheEntry],
    *,
    top_p: float = 0.2,
) -> list[set[int]]:
    """Run a query-only forward over cached KV and select tokens to recompute.

    1. Forward the query against assembled cached KV with ``output_attentions=True``.
    2. Extract attention weights from query tokens to document tokens.
    3. Average across heads and layers (uniform fusion, per ProphetKV ablation).
    4. For each document, select the top-``top_p`` fraction of tokens by score.

    Args:
        model: HF causal LM (eval mode, on GPU).
        query_ids: ``[1, Q]`` query token IDs.
        entries: per-document cache entries (restart or deferred layout).
        top_p: fraction of tokens per document to select for recomputation (0–1).

    Returns:
        List of sets, one per document, containing intra-document token indices
        to recompute with full attention.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if not entries:
        return []

    from ddi.cache import assemble_dynamic_cache
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    doc_lengths = [e.doc_len for e in entries]
    query_len = int(query_ids.shape[-1])
    doc_total = sum(doc_lengths)

    cache = assemble_dynamic_cache(entries, model.device)
    positions = build_positions(doc_lengths, query_len, 0, entries[0].layout)
    query_positions = positions[doc_total:].unsqueeze(0).to(model.device)
    cache_position = torch.arange(
        doc_total, doc_total + query_len, dtype=torch.long, device=model.device
    )

    allow = build_mask(doc_lengths, query_len, 0, "ddi_qf")
    additive = (
        to_additive(allow[doc_total:, :], dtype=model.dtype)
        .to(model.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=query_ids.to(model.device),
            attention_mask=additive,
            position_ids=query_positions,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )

    if out.attentions is None:
        raise RuntimeError(
            "model did not return attention weights; "
            "ensure output_attentions=True is supported"
        )

    scores = _fuse_attention_scores(out.attentions, doc_total, query_len)

    return select_top_k(scores, doc_lengths, top_p)


def _fuse_attention_scores(
    attentions: tuple[torch.Tensor, ...],
    doc_total: int,
    query_len: int,
) -> torch.Tensor:
    """Fuse per-layer attention weights into per-token scores (ProphetKV Eq. 7).

    ProphetKV computes for each layer l::

        α_l(t) = (1/|Q_s|) Σ_{q ∈ Q_s} Φ_{q,t}^{(l)}

    where Φ is the softmax attention weight from query token q to context
    token t.  Then fuses across layers with uniform averaging::

        ᾱ(t) = (1/L) Σ_l α_l(t)

    ``attentions[l]`` has shape ``[batch, heads, query_seq, key_seq]`` where
    ``key_seq = doc_total + query_len``.  We extract the query→document block
    ``[:, :, :, :doc_total]``, average across heads, then average across query
    positions (matching ProphetKV's 1/|Q_s| normalization), then average
    across layers.

    Returns:
        ``[doc_total]`` tensor of fused attention scores.
    """
    layer_scores = []
    for attn in attentions:
        doc_attn = attn[0, :, :, :doc_total]  # [heads, query_seq, doc_total]
        head_avg = doc_attn.mean(dim=0)  # [query_seq, doc_total]
        # ProphetKV: average across query tokens (1/|Q_s|), not sum.
        query_avg = head_avg.mean(dim=0)  # [doc_total]
        layer_scores.append(query_avg)

    stacked = torch.stack(layer_scores)  # [num_layers, doc_total]
    # ProphetKV Eq. 7: uniform fusion across layers.
    return stacked.mean(dim=0)  # [doc_total]


def select_top_k(
    scores: torch.Tensor,
    doc_lengths: list[int],
    top_p: float,
) -> list[set[int]]:
    """Select the top-p% tokens globally across all documents by fused score.

    ProphetKV (Algorithm 1) selects globally, not per-document: the fused score
    is computed for every context token, and the top-p fraction is selected
    across the entire concatenated cache.  This avoids spending the recomputation
    budget on globally-salient but query-irrelevant tokens in documents that
    happen to have many high-attention tokens.

    Args:
        scores: ``[doc_total]`` fused attention scores for all document tokens.
        doc_lengths: per-document token counts.
        top_p: fraction of tokens to select globally.

    Returns:
        List of sets, one per document, of intra-document token indices to recompute.
    """
    doc_total = sum(doc_lengths)
    k = max(1, int(doc_total * top_p))
    k = min(k, doc_total)

    _, top_global_indices = torch.topk(scores, k)
    top_set = set(top_global_indices.tolist())

    # Split global indices into per-document intra-document indices.
    result: list[set[int]] = []
    offset = 0
    for length in doc_lengths:
        doc_indices = {gi - offset for gi in top_set if offset <= gi < offset + length}
        result.append(doc_indices)
        offset += length
    return result
