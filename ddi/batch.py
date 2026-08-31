"""Batched prefill with per-sample block-diagonal masks (Sprint E, VIBEQUEUE-29).

Extends the batch=1 inference path to process B requests in a single forward
pass. Each sample in the batch has its own document structure (different
``doc_lengths``), so the 4D additive mask is ``[B, 1, T, T]`` where each
sample's block-diagonal pattern is embedded independently. Padding positions
are fully masked (``-inf``) so they do not affect any attention computation.

Generation (greedy decode) stays batch=1 — the prefill is where most compute
goes, and batching decode with variable-length outputs and per-sample KV
caches is materially more complex for a marginal gain. The throughput
improvement from batched prefill alone is the Sprint E result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ddi.masks import Variant, build_mask, to_additive
from ddi.positions import build_positions
from ddi.prompt import encode_segments


@dataclass(frozen=True)
class BatchedEncoding:
    """Encoded batch of B samples, padded to the same total length T.

    ``input_ids``      : [B, T] long — padded token IDs.
    ``position_ids``   : [B, T] long — sequential positions per sample; padding = 0.
    ``attention_mask`` : [B, 1, T, T] — additive mask (0 / -inf) with per-sample
                         block-diagonal structure and full masking of padding.
    ``doc_lengths``    : list[list[int]] — per-sample document lengths.
    ``query_lens``     : list[int] — per-sample query lengths.
    ``answer_lens``    : list[int] — per-sample answer lengths (0 for generation-only).
    ``seq_lens``       : list[int] — per-sample unpadded sequence lengths.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    doc_lengths: list[list[int]]
    query_lens: list[int]
    answer_lens: list[int]
    seq_lens: list[int]


def encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    samples: list[tuple[list[str], str]] | list[tuple[list[str], str, str]],
    mask_variant: str = "ddi_qf",
    layout: str = "sequential",
    pad_token_id: int | None = None,
) -> BatchedEncoding:
    """Encode B samples into a padded BatchedEncoding.

    Each sample is either a 2-tuple ``(docs, query)`` (generation, no answer)
    or a 3-tuple ``(docs, query, answer)`` (loss computation). Tokenised
    individually via ``ddi.prompt.encode_segments``. All samples are
    right-padded to the maximum total length. The additive mask embeds each
    sample's block-diagonal structure and fully masks padding positions.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    encoded: list[torch.Tensor] = []
    doc_lens: list[list[int]] = []
    query_lens: list[int] = []
    answer_lens: list[int] = []
    all_masks: list[torch.Tensor] = []

    for sample in samples:
        if len(sample) == 3:
            docs, query, answer = sample
        else:
            docs, query = sample
            answer = None

        input_ids, dl, ql, al = encode_segments(tokenizer, list(docs), query, answer)

        T = input_ids.shape[-1]
        allow = build_mask(
            dl,
            query_len=ql,
            answer_len=al,
            variant=mask_variant,  # type: ignore[arg-type]
        )
        additive = to_additive(allow, dtype=torch.float32)
        encoded.append(input_ids)
        doc_lens.append(dl)
        query_lens.append(ql)
        answer_lens.append(al)
        all_masks.append(additive)

    max_T = max(t.shape[-1] for t in encoded)
    B = len(encoded)
    device = encoded[0].device

    padded_ids = torch.full((B, max_T), pad_token_id, dtype=torch.long, device=device)
    padded_pos = torch.zeros((B, max_T), dtype=torch.long, device=device)
    padded_mask = torch.full(
        (B, 1, max_T, max_T), float("-inf"), dtype=torch.float32, device=device
    )

    for i in range(B):
        T = encoded[i].shape[-1]
        padded_ids[i, :T] = encoded[i]
        positions = build_positions(
            doc_lens[i],
            query_len=query_lens[i],
            answer_len=answer_lens[i],
            layout=layout,
        )
        padded_pos[i, :T] = positions.to(device)
        padded_mask[i, 0, :T, :T] = all_masks[i]
        # Padded rows would be all -inf, producing NaN in softmax that can
        # propagate through shared layers. Set the diagonal to 0 so each
        # padded position attends to itself — the output is discarded but
        # stays finite.
        for j in range(T, max_T):
            padded_mask[i, 0, j, j] = 0.0

    return BatchedEncoding(
        input_ids=padded_ids,
        position_ids=padded_pos,
        attention_mask=padded_mask,
        doc_lengths=doc_lens,
        query_lens=query_lens,
        answer_lens=answer_lens,
        seq_lens=[t.shape[-1] for t in encoded],
    )


def forward_batch(
    model: PreTrainedModel,
    batch: BatchedEncoding,
) -> torch.Tensor:
    """Batched forward pass. Returns logits [B, T, V]."""
    input_ids = batch.input_ids.to(model.device)
    positions = batch.position_ids.to(model.device)
    additive = batch.attention_mask.to(model.device).to(model.dtype)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=False,
        )
    return out.logits


def batched_loss(
    logits: torch.Tensor,
    batch: BatchedEncoding,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[tuple[list[str], str, str]],
) -> list[float]:
    """Compute per-sample cross-entropy loss from batched logits.

    The batched forward must have been run with answer tokens included
    (3-tuple samples passed to ``encode_batch``). Each sample's logits are
    sliced to its unpadded length and loss is computed on answer positions only.
    """
    from ddi.metrics import make_answer_labels
    from ddi.run import shifted_cross_entropy

    losses: list[float] = []
    for i, (docs, query, answer) in enumerate(samples):
        input_ids, dl, ql, al = encode_segments(tokenizer, list(docs), query, answer)
        T = input_ids.shape[-1]
        labels = make_answer_labels(input_ids, T - al, T)
        sample_logits = logits[i, :T].unsqueeze(0).to(torch.float32)
        labels = labels.to(sample_logits.device)
        loss = float(shifted_cross_entropy(sample_logits, labels).item())
        losses.append(loss)
    return losses
