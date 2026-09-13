"""Does incremental decoding with a KV cache agree with one masked forward?

Motivated by the 2026-08-06/07 result on Qwen3.6-27B, hotpot_qa: at 5 and 10 documents
`ddi_qf` scored **identically to `blind`** on generation metrics (presence 0.15 vs 0.15)
while `loss` put it within 12-15 % of `full`. Same job, same records, opposite conclusions.

The mask was excluded as the cause by direct comparison — `build_mask` produces byte-identical
prefill blocks, answer rows and positions for both paths, and `ddi_qf` differs from `full`
*only* in the document rows (query and answer rows are identical, so the answer attends to
every document under both). What remains is the one structural difference between how the
two metrics are computed:

* `loss`       — `ddi/run.py::_quality_cell`, ONE forward over docs+query+answer,
                 `use_cache=False`, custom 4D additive mask over the whole sequence.
* `generation` — `ddi/inference.py::generate`, prefill with `use_cache=True`, then one
                 forward per token with a sliced mask row and `past_key_values`.

If those two disagree, every generation metric in this ticket is measuring the cache path
rather than the mask — and cache reuse is the entire point of DDI, so a defect there is the
result, not a nuisance.

This test asserts they agree: teacher-force the same answer tokens through both paths and
compare answer-position logits. It needs `transformers` and a GPU, so it is gated like the
other integration tests — the dev host cannot collect it, which is exactly how the two bugs
in `generate` reached production today.

Run with:  DDI_INTEGRATION=1 python -m pytest tests/test_cache_vs_single_forward.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

_INTEGRATION = os.environ.get("DDI_INTEGRATION") == "1"
_REASON = "needs transformers + a GPU; opt in with DDI_INTEGRATION=1"

MODEL = os.environ.get("DDI_CACHE_TEST_MODEL", "Qwen/Qwen3.5-4B-Instruct")
VARIANTS = ("full", "ddi_qf", "blind")


@pytest.mark.skipif(not _INTEGRATION, reason=_REASON)
@pytest.mark.parametrize("variant", VARIANTS)
def test_cached_decode_matches_single_forward(variant: str) -> None:
    """Answer-position logits must match between one masked forward and prefill+cache.

    Teacher-forced on both sides so the comparison isolates the cache: the same answer
    tokens are fed in, and only the mechanism for producing their logits differs.
    """
    from ddi import inference
    from ddi.masks import build_mask, to_additive

    model, tokenizer = inference.load(MODEL)
    model.eval()

    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "It is named after the engineer Gustave Eiffel, whose company designed it.",
        "Construction ran from 1887 to 1889 as the centrepiece of the 1889 World's Fair.",
    ]
    query = "Question: Who is the Eiffel Tower named after? Answer with a short span."
    answer = " Gustave Eiffel"

    doc_ids = [
        tokenizer(d, return_tensors="pt", add_special_tokens=False).input_ids for d in docs
    ]
    doc_lengths = [int(t.shape[-1]) for t in doc_ids]
    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    query_len = int(query_ids.shape[-1])
    answer_len = int(answer_ids.shape[-1])
    prefix_len = sum(doc_lengths) + query_len

    allow = build_mask(
        doc_lengths, query_len=query_len, answer_len=answer_len, variant=variant
    )
    additive_full = to_additive(allow, dtype=model.dtype).to(model.device)

    full_ids = torch.cat(doc_ids + [query_ids, answer_ids], dim=-1).to(model.device)

    # --- path A: one forward, no cache (what `loss` measures) ---------------------------
    with torch.no_grad():
        single = model(
            input_ids=full_ids,
            attention_mask=additive_full.clone().unsqueeze(0).unsqueeze(0),
            use_cache=False,
        ).logits

    # --- path B: prefill + incremental decode (what `presence`/`f1`/`em` measure) -------
    prefill_ids = full_ids[:, :prefix_len]
    with torch.no_grad():
        out = model(
            input_ids=prefill_ids,
            attention_mask=additive_full[:prefix_len, :prefix_len]
            .clone()
            .unsqueeze(0)
            .unsqueeze(0),
            use_cache=True,
        )
        past = out.past_key_values
        stepped = [out.logits[:, -1:, :]]
        for i in range(answer_len - 1):
            kv_len = prefix_len + i + 1
            step_mask = (
                additive_full[prefix_len + i, :kv_len].clone().reshape(1, 1, 1, kv_len)
            )
            step = model(
                input_ids=full_ids[:, prefix_len + i : prefix_len + i + 1],
                attention_mask=step_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            stepped.append(step.logits[:, -1:, :])
    incremental = torch.cat(stepped, dim=1)

    # Compare the logits that actually decide the generated answer.
    single_answer = single[:, prefix_len - 1 : prefix_len - 1 + answer_len, :].float()
    incremental = incremental.float()

    assert incremental.shape == single_answer.shape, (
        f"{variant}: shape mismatch {incremental.shape} vs {single_answer.shape}"
    )

    # argmax agreement is what generation actually depends on; check it first because a
    # tolerance failure with matching argmax is benign numerics, while a differing argmax
    # changes the emitted text.
    same_token = (incremental.argmax(-1) == single_answer.argmax(-1)).all()
    max_abs = (incremental - single_answer).abs().max().item()
    assert same_token, (
        f"{variant}: cached decode picks different tokens than the single forward "
        f"(max|Δlogit| = {max_abs:.4f}) — generation metrics are measuring the cache path, "
        f"not the mask"
    )
    assert max_abs < 0.5, f"{variant}: logits diverge by {max_abs:.4f} between the two paths"


@pytest.mark.skipif(not _INTEGRATION, reason=_REASON)
def test_assembled_document_cache_matches_ddi_generation(tmp_path) -> None:
    """Original, reordered, and partial-hit requests must preserve greedy generation."""
    from ddi import inference
    from ddi.cache import (
        DocCache,
        compute_doc_kv,
        compute_key,
        forward_query_with_doc_cache,
        generate_tokens_with_doc_cache,
    )
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    model, tokenizer = inference.load(MODEL)
    model.eval()
    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "The structure is named after engineer Gustave Eiffel.",
        "Construction finished for the 1889 World's Fair.",
    ]
    query = "Question: Who is the Eiffel Tower named after? Answer briefly."
    entries = [compute_doc_kv(model, tokenizer, doc, "restart") for doc in docs]
    doc_ids = [
        tokenizer(doc, return_tensors="pt", add_special_tokens=False).input_ids
        for doc in docs
    ]
    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids
    doc_lengths = [int(ids.shape[-1]) for ids in doc_ids]
    query_len = int(query_ids.shape[-1])
    doc_total = sum(doc_lengths)

    cached = forward_query_with_doc_cache(model, query_ids, entries).logits.float()
    all_ids = torch.cat([*doc_ids, query_ids], dim=-1).to(model.device)
    positions = build_positions(doc_lengths, query_len, 0, "restart").unsqueeze(0)
    allow = build_mask(doc_lengths, query_len, 0, "ddi_qf")
    additive = to_additive(allow, dtype=model.dtype)[None, None]
    with torch.no_grad():
        recomputed = model(
            input_ids=all_ids,
            attention_mask=additive.to(model.device),
            position_ids=positions.to(model.device),
            use_cache=False,
        ).logits[:, doc_total:, :].float()

    assert cached.shape == recomputed.shape
    assert cached[:, -1].argmax(-1).item() == recomputed[:, -1].argmax(-1).item()
    cosine = torch.nn.functional.cosine_similarity(
        cached.flatten(), recomputed.flatten(), dim=0
    ).item()
    assert cosine >= 0.999, f"assembled cache cosine similarity is only {cosine:.6f}"

    disk_cache = DocCache(tmp_path / "kv")
    for entry in entries:
        disk_cache.put(entry)
    new_doc = "The tower stands on the Champ de Mars in Paris."
    partial_docs = [docs[0], new_doc, docs[2]]
    partial_entries = []
    hits = 0
    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    for doc in partial_docs:
        entry = disk_cache.get(compute_key(model_id, model.dtype, doc))
        if entry is None:
            entry = compute_doc_kv(model, tokenizer, doc, "restart")
            disk_cache.put(entry)
        else:
            hits += 1
        partial_entries.append(entry)
    assert hits == 2

    scenarios = [
        (docs, entries),
        ([docs[2], docs[0], docs[1]], [entries[2], entries[0], entries[1]]),
        (partial_docs, partial_entries),
    ]
    for scenario_docs, scenario_entries in scenarios:
        batch = inference.encode(tokenizer, scenario_docs, query, layout="restart")
        expected_text = inference.generate(
            model, tokenizer, batch, "ddi_qf", max_new_tokens=16
        )[0]
        actual_tokens = generate_tokens_with_doc_cache(
            model, tokenizer, query_ids, scenario_entries, max_new_tokens=16
        )
        actual_text = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()
        assert actual_text == expected_text
