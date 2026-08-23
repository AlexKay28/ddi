"""Prompt construction — the single place that turns a record into query text.

Before this module existed the same eight lines were inlined in three places
(`ddi/run.py::_quality_cell`, `ddi/finetune.py::_encode_record`,
`ddi/perplexity.py::compute_perplexity`), and all three built a query that

1. never told the model the documents existed, and
2. never asked for a short answer span,

while the metrics (`em`, `f1`) score a short answer span against the gold string.
The result was `em == 0.0000` in every cell of every sweep — the harness had no
dynamic range at all, so `full` and `blind` were indistinguishable and no parity
claim could be made either way (VIBEQUEUE-2 review, issue R3).

Geometry note
-------------
`ddi.masks.build_mask` assumes the token stream is exactly ``[docs][query][answer]``
with the documents as a *prefix*. So the instruction has to ride inside the query
block — it cannot be a system turn placed before the documents without changing
the mask builder's contract. That is why this module rewrites the last user turn
rather than prepending a system message.

Set ``DDI_PROMPT_INSTRUCTION=0`` to restore the pre-review bare-question prompt
(needed only to reproduce historical store rows).
"""

from __future__ import annotations

import os

import torch

# Kept deliberately close to the wording the synthetic_qa loader arrived at
# independently (`ddi/data/synthetic.py`), whose comment records that without an
# explicit "the passages above are the source of truth" line the models "never
# look at the passages at all".
DOC_INSTRUCTION = (
    "Use the passages above as the only source of truth. "
    "Reply with the exact short answer span and nothing else — "
    "no explanation, no full sentence."
)

_ENV_INSTRUCTION = "DDI_PROMPT_INSTRUCTION"


def instruction_enabled() -> bool:
    """False only when ``DDI_PROMPT_INSTRUCTION=0`` — the reproduce-old-numbers escape."""
    return os.environ.get(_ENV_INSTRUCTION, "1") != "0"


def build_user_dialog(record: dict) -> list[dict]:
    """Return the dialog turns up to (not including) the assistant answer.

    The last user turn is rewritten to name the passages and demand a bare span,
    unless the record opts out via ``meta.self_instructed`` (synthetic_qa writes
    its own instruction into the question) or the record carries no documents
    (nothing to point at).
    """
    dialog = [dict(turn) for turn in record["dialog"][:-1]]
    if not dialog:
        return dialog
    if not instruction_enabled():
        return dialog
    if (record.get("meta") or {}).get("self_instructed"):
        return dialog
    if not record.get("info_contexts"):
        return dialog

    last = dialog[-1]
    if last.get("role") != "user":
        return dialog
    last["content"] = f"{DOC_INSTRUCTION}\n\nQuestion: {last['content']}\nAnswer:"
    return dialog


def thinking_disabled(tokenizer) -> bool:
    """True when this tokenizer's chat template understands `enable_thinking`.

    Reasoning-tuned templates (the Qwen3 family) open a `<think>` block as part of the
    generation prompt. That is fatal here: `max_new_tokens` is 64, so the model spends its
    whole budget reasoning and the bare answer span is never emitted.

    Measured on 2026-08-06/07, hotpot_qa at 10 documents, `full` mask, identical records:

        Qwen2.5-7B-Instruct   (no <think> in template)   em 0.5700  f1 0.7011  loss 1.18
        Qwen3.5-4B-Instruct   (<think> in template)      em 0.0000  f1 0.0317  loss 5.98
        Qwen3.6-27B-Instruct  (<think> in template)      em 0.0000  f1 0.0499  loss 5.58

    The signature is diagnostic: `em` is exactly 0 while `presence` stays non-zero (0.15 and
    0.31) — the gold answer does appear in the output, just never on its own, because it is
    buried in reasoning. `loss` is unaffected because it is teacher-forced and never
    generates. That divergence between the loss path and the generation path on the same run
    was mistaken first for a KV-cache defect and then for a real ranking-vs-generation
    effect; it is neither.

    Checked against the template source rather than the model name, so a future
    reasoning-tuned checkpoint is handled without a hardcoded list. Passing the flag to a
    template that ignores it would be silently useless, which is exactly the failure mode
    worth avoiding.
    """
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "enable_thinking" in template


def build_query_text(tokenizer, record: dict) -> str:
    """Chat-templated query text for one record — the `[query]` block of the stream.

    Every caller must go through here so eval, perplexity and training see a
    byte-identical prompt. A train/eval prompt mismatch would show up as a
    quality regression that has nothing to do with the mask.
    """
    kwargs = {"enable_thinking": False} if thinking_disabled(tokenizer) else {}
    return tokenizer.apply_chat_template(
        build_user_dialog(record), tokenize=False, add_generation_prompt=True, **kwargs
    )


def encode_segments(
    tokenizer,
    doc_texts: list[str],
    query_text: str,
    answer_text: str | None = None,
) -> tuple[torch.Tensor, list[int], int, int]:
    """Tokenise ``[docs][query][answer]`` into one stream. The only encoder in the repo.

    Returns ``(full_ids [1, T], doc_lengths, query_len, answer_len)``. ``answer_text=None``
    means "no answer yet" (the generation path) and yields ``answer_len == 0``.

    Every document is tokenised *individually* — the per-document lengths are the block
    structure `ddi.masks.build_mask` and `ddi.positions.build_positions` key on, so
    tokenising the concatenation instead would silently change both. ``add_special_tokens``
    is False on every segment: the chat template already emitted whatever the model needs,
    and a BOS spliced in front of each document would shift every block boundary.

    This function exists because the same block was written out five times (eval loss,
    restart-quality loss, training, perplexity, generation) and drifted. See the module
    docstring: eval, perplexity and training must see a byte-identical stream or the whole
    comparison the research rests on is between three different prompts.
    """
    doc_id_tensors = [
        tokenizer(d, return_tensors="pt", add_special_tokens=False).input_ids
        for d in doc_texts
    ]
    doc_lengths = [int(t.shape[-1]) for t in doc_id_tensors]

    query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids
    query_len = int(query_ids.shape[-1])

    parts = [*doc_id_tensors, query_ids]
    answer_len = 0
    if answer_text is not None:
        answer_ids = tokenizer(
            answer_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        answer_len = int(answer_ids.shape[-1])
        parts.append(answer_ids)

    full_ids = torch.cat(parts, dim=-1)
    return full_ids, doc_lengths, query_len, answer_len


def encode_record(tokenizer, record: dict) -> tuple[torch.Tensor, list[int], int, int]:
    """`encode_segments` for a dataset record: docs from ``info_contexts``, query from
    `build_query_text`, answer from the last dialog turn.

    The generation path deliberately does not come through here — it calls `encode_segments`
    with no answer, because it legitimately generates from a query it built itself
    (`ddi/inference.py::encode`, driven by `ddi/run.py` and by the mask tests' synthetic
    strings). Both entry points share the one tokenisation; only the source of the texts
    differs.

    The returned ``full_ids`` live on the CPU — placing them on a device is the caller's
    business, and the tokenisation must not depend on it.
    """
    return encode_segments(
        tokenizer,
        list(record["info_contexts"]),
        build_query_text(tokenizer, record),
        record["dialog"][-1]["content"],
    )


__all__ = [
    "DOC_INSTRUCTION",
    "build_query_text",
    "build_user_dialog",
    "encode_record",
    "encode_segments",
    "instruction_enabled",
    "thinking_disabled",
]
