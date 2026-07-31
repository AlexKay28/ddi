"""Answer-token-only perplexity helper (issue #120).

Reuses `ddi.masks.build_mask` + `ddi.positions.build_positions` + the additive-mask
plumbing already exercised by `ddi/run.py::_quality_cell` so a caller that already
runs a masked forward can compute perplexity without introducing any new
attention path. Answer-position isolation goes through `ddi.metrics.make_answer_labels`
so cross-entropy is restricted to answer tokens — matching every existing
loss-emitting cell in the codebase.

The public surface is one function: ``compute_perplexity``. Records use the
common dataset shape (``info_contexts``, ``dialog``, ``possible_answers``); the
last dialog turn is treated as the answer, prior turns as the query.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from ddi.masks import build_mask, to_additive
from ddi.metrics import make_answer_labels
from ddi.positions import build_positions


def compute_perplexity(
    model,
    tokenizer,
    records: Iterable[dict],
    *,
    mask_variant: str,
    mask_kwargs: dict | None = None,
    layout: str = "sequential",
) -> float:
    """Return answer-token-only perplexity across ``records``.

    For each record: build ``docs + query + answer`` token ids; construct the
    additive mask via `ddi.masks.build_mask` for ``mask_variant``; construct
    ``position_ids`` via `ddi.positions.build_positions`; run one forward pass;
    compute shifted cross-entropy restricted to answer tokens via
    `ddi.metrics.make_answer_labels`. The per-token cross-entropy is summed
    across every record and then divided by the total answer-token count before
    exponentiating — this is the standard "corpus perplexity" definition
    (single exp over the mean per-token NLL), not a per-record average.

    Coerces DDI variants to ``full`` on records with no info_contexts, matching
    the ``_quality_cell`` behaviour so a mixed-record test set does not crash
    on the no-doc rows.
    """
    kwargs = dict(mask_kwargs) if mask_kwargs else {}
    total_nll = 0.0
    total_tokens = 0

    was_training = model.training
    model.eval()
    try:
        for record in records:
            if not record["possible_answers"]:
                continue
            doc_texts = list(record["info_contexts"])
            user_dialog = record["dialog"][:-1]
            answer_text = record["dialog"][-1]["content"]

            variant_here = mask_variant
            kwargs_here = kwargs
            if not doc_texts and mask_variant != "full":
                variant_here = "full"
                kwargs_here = {}

            query_text = tokenizer.apply_chat_template(
                user_dialog, tokenize=False, add_generation_prompt=True
            )

            doc_id_tensors = [
                tokenizer(d, return_tensors="pt", add_special_tokens=False).input_ids
                for d in doc_texts
            ]
            doc_lengths = [int(t.shape[-1]) for t in doc_id_tensors]
            query_ids = tokenizer(
                query_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            query_len = int(query_ids.shape[-1])
            answer_ids = tokenizer(
                answer_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            answer_len = int(answer_ids.shape[-1])
            if answer_len == 0:
                continue

            full_ids = torch.cat(
                doc_id_tensors + [query_ids, answer_ids], dim=-1
            ).to(model.device)
            labels = make_answer_labels(
                full_ids,
                full_ids.shape[-1] - answer_len,
                full_ids.shape[-1],
            ).to(model.device)

            allow = build_mask(
                doc_lengths,
                query_len=query_len,
                answer_len=answer_len,
                variant=variant_here,  # type: ignore[arg-type]
                **kwargs_here,
            )
            additive = (
                to_additive(allow, dtype=model.dtype)
                .to(model.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            positions = (
                build_positions(
                    doc_lengths, query_len, answer_len, layout  # type: ignore[arg-type]
                )
                .unsqueeze(0)
                .to(model.device)
            )

            with torch.no_grad():
                out = model(
                    input_ids=full_ids,
                    attention_mask=additive,
                    position_ids=positions,
                    use_cache=False,
                )
            # Standard next-token shift: predict position t from logits at t-1.
            shift_logits = out.logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            per_tok = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid = int((shift_labels != -100).sum().item())
            if valid == 0:
                continue
            total_nll += float(per_tok.item())
            total_tokens += valid
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError(
            "compute_perplexity: no answer tokens found across records "
            "(every record was empty or had no possible_answers)"
        )
    return math.exp(total_nll / total_tokens)
