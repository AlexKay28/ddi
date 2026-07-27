"""Phase 2.1 train-free mask sweep runner (issue #13).

Registers `task: phase2_train_free`. Same shape as `phase1_baseline`, but the mask
variant is a config parameter so the serverless orchestrator can invoke this once per
`{model} × {mask variant} × {dataset}` cell of the sweep. Loss formula is imported
from `ddi/runners/_shared.py` and matches the Phase 1.4 baseline exactly — that
identity is the whole point of `Δ` (RESEARCH_PLAN §8, ground rule 9).

Prefill uses the DDI mask via `forward_with_mask` (patched HF `_update_causal_mask`
short-circuits our 4D float mask). Generation is a greedy cached-KV loop that starts
from a mask-controlled prefill and then streams one token at a time — HF's own
`model.generate` cannot inject a custom prefill mask.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ddi.cli import register_runner
from ddi.data import load
from ddi.masks import build_mask, to_additive
from ddi.metrics import answer_presence, exact_match, make_answer_labels, token_f1
from ddi.model import forward_with_mask, load_model
from ddi.provenance import capture
from ddi.runners._shared import shifted_cross_entropy
from ddi.store import append


def _generate_with_prefill_mask(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefill_ids: torch.Tensor,
    prefill_mask: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Greedy decode: DDI-masked prefill, then cached-KV single-token steps.

    Only the prefill call carries the custom mask; each decode step feeds one new token
    with `past_key_values`, so HF's own KV-cache path handles the causal geometry — we
    do not re-supply an attention mask (the archaeology's central failure mode was
    hand-rolling that loop).
    """
    with torch.no_grad():
        out = model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

        generated: list[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            tok_id = int(next_token.item())
            generated.append(tok_id)
            if eos_id is not None and tok_id == eos_id:
                break
            step = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one cell of the train-free mask sweep; see module docstring."""
    dataset = config["dataset"]
    model_name = config["model"]
    if "mask" not in config:
        raise ValueError("phase2_train_free config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase2_train_free config.mask missing required key: variant")
    mask_variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")
    if mask_variant == "ddi_bridge_k" and bridge_k is None:
        raise ValueError("mask.variant='ddi_bridge_k' requires mask.bridge_k to be set")
    if mask_variant != "ddi_bridge_k" and bridge_k is not None:
        raise ValueError(
            f"mask.bridge_k is only valid for variant='ddi_bridge_k', not {mask_variant!r}"
        )

    split = config.get("split", "val")
    max_new_tokens = int(config.get("max_new_tokens", 64))
    max_records = config.get("max_records")

    model, tokenizer = load_model(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    prov = capture(
        config_hash=config_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=dataset,
    )
    variant_tag = mask_variant if bridge_k is None else f"{mask_variant}_k{bridge_k}"
    run_id_prefix = (
        f"phase2_train_free:{model_name}:{dataset}:{variant_tag}:seed={seed}:{prov.git_sha[:8]}"
    )

    coerced_full = 0

    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break
        if not record["possible_answers"]:
            print(f"[phase2_train_free] skip record id={record['id']}: no possible_answers")
            continue

        doc_ids_list = [
            tokenizer(doc, return_tensors="pt", add_special_tokens=False).input_ids
            for doc in record["info_contexts"]
        ]
        doc_lengths = [ids.shape[-1] for ids in doc_ids_list]

        user_dialog = record["dialog"][:-1]
        answer = record["dialog"][-1]["content"]

        query_text = tokenizer.apply_chat_template(
            user_dialog, tokenize=False, add_generation_prompt=True
        )
        query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids
        query_len = query_ids.shape[-1]

        answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
        answer_len = answer_ids.shape[-1]

        # WHY coerce: DDI variants exist to isolate cross-document leakage. A record with no
        # documents has nothing to isolate, so any DDI variant is definitionally equal to full;
        # forcing the mask builder to accept an empty doc list would just be a special-case bug
        # farm. Coerce silently per record and log the count once at the end.
        variant_here = mask_variant
        bridge_k_here = bridge_k
        if not doc_lengths and mask_variant != "full":
            variant_here = "full"
            bridge_k_here = None
            coerced_full += 1

        full_ids = torch.cat(doc_ids_list + [query_ids, answer_ids], dim=-1).to(model.device)

        answer_start = full_ids.shape[-1] - answer_len
        answer_end = full_ids.shape[-1]
        labels = make_answer_labels(full_ids, answer_start, answer_end).to(model.device)

        # `build_mask`'s T = sum(doc_lengths) + query_len + answer_len already accounts for the
        # query/answer span, so `doc_lengths` must stay `[]` for no-doc records — substituting
        # `full_ids.shape[-1]` here would double-count query+answer into T (issue found via #35's
        # `nq` sweep: `nq` has no info_contexts, so every record hit this and broke on shape).
        # `build_mask([], ...)` already yields plain causal for every variant (empty sum, and its
        # own `len(doc_lengths) <= 1` collapse), so no substitution is needed at all.
        allow = build_mask(
            doc_lengths,
            query_len=query_len,
            answer_len=answer_len,
            variant=variant_here,
            bridge_k=bridge_k_here,
        )
        additive = (
            to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
        )

        with torch.no_grad():
            logits = forward_with_mask(model, full_ids, additive)
        loss = shifted_cross_entropy(logits, labels).item()

        prefill_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)
        prefill_allow = build_mask(
            doc_lengths,
            query_len=query_len,
            answer_len=0,
            variant=variant_here,
            bridge_k=bridge_k_here,
        )
        prefill_mask = (
            to_additive(prefill_allow, dtype=model.dtype)
            .to(model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        pred = _generate_with_prefill_mask(
            model, tokenizer, prefill_ids, prefill_mask, max_new_tokens
        )
        em = exact_match(pred, record["possible_answers"])
        f1 = token_f1(pred, record["possible_answers"])
        presence = answer_presence(pred, record["possible_answers"])

        rid = f"{run_id_prefix}:{record['id']}"
        append(rid, "loss", loss, prov)
        append(rid, "em", em, prov)
        append(rid, "f1", f1, prov)
        append(rid, "presence", presence, prov)

    if coerced_full > 0:
        print(
            f"[phase2_train_free] coerced variant={mask_variant!r} -> 'full' on "
            f"{coerced_full} record(s) with no info_contexts"
        )


register_runner("phase2_train_free")(_run)
