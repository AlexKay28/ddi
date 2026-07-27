"""Phase 5.3 cross-document composition probe runner (issue #28).

Registers `task: phase5_composition`. Same evaluation shape as `phase4_restart_quality`
(mask-controlled prefill + cached-KV greedy decode with explicit position IDs), but the
per-record output row also encodes the probe's two knobs — `distance` (distractors between
doc_A and doc_B) and `pair_position` — so downstream T6/F7 can slice per (variant, layout,
distance) and draw the boundary curve. See RESEARCH_PLAN §8 Phase 5.3 and issue #28.
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
from ddi.positions import Layout, build_positions
from ddi.provenance import capture
from ddi.runners._shared import shifted_cross_entropy
from ddi.store import append


def _generate_with_prefill_mask_and_positions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefill_ids: torch.Tensor,
    prefill_mask: torch.Tensor,
    prefill_positions: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Greedy decode with mask+positions on prefill only; decode steps use cached KV."""
    with torch.no_grad():
        out = model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_positions,
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


def _build_run_id(
    *,
    model_name: str,
    dataset: str,
    variant_tag: str,
    layout: str,
    distance: int,
    pair_position: str,
    seed: int,
    git_sha_short: str,
    record_id: str,
) -> str:
    """Encode the probe knobs into the run_id so T6/F7 can parse them back out."""
    return (
        f"phase5_composition:{model_name}:{dataset}:{variant_tag}:{layout}:"
        f"d{distance}:p{pair_position}:seed={seed}:{git_sha_short}:{record_id}"
    )


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run the Phase 5.3 composition probe; see module docstring."""
    dataset = config.get("dataset", "composition")
    model_name = config["model"]
    if "mask" not in config:
        raise ValueError("phase5_composition config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase5_composition config.mask missing required key: variant")
    mask_variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")
    if mask_variant == "ddi_bridge_k" and bridge_k is None:
        raise ValueError("mask.variant='ddi_bridge_k' requires mask.bridge_k to be set")
    if mask_variant != "ddi_bridge_k" and bridge_k is not None:
        raise ValueError(
            f"mask.bridge_k is only valid for variant='ddi_bridge_k', not {mask_variant!r}"
        )

    layout_cfg = config.get("layout", "restart")
    if layout_cfg not in ("sequential", "restart"):
        raise ValueError(
            f"unknown layout {layout_cfg!r}; expected 'sequential' or 'restart'"
        )
    layout: Layout = layout_cfg  # type: ignore[assignment]

    split = config.get("split", "val")
    max_new_tokens = int(config.get("max_new_tokens", 32))
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
    git_sha_short = prov.git_sha[:8]

    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break
        if not record["possible_answers"]:
            print(f"[phase5_composition] skip record id={record['id']}: no possible_answers")
            continue

        meta = record.get("meta") or {}
        distance = int(meta.get("distance", -1))
        pair_position = str(meta.get("pair_position", "unknown"))

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

        full_ids = torch.cat(doc_ids_list + [query_ids, answer_ids], dim=-1).to(model.device)

        answer_start = full_ids.shape[-1] - answer_len
        answer_end = full_ids.shape[-1]
        labels = make_answer_labels(full_ids, answer_start, answer_end).to(model.device)

        allow = build_mask(
            doc_lengths or [full_ids.shape[-1]],
            query_len=query_len,
            answer_len=answer_len,
            variant=mask_variant,
            bridge_k=bridge_k,
        )
        additive = (
            to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
        )

        prefill_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)
        prefill_allow = build_mask(
            doc_lengths or [prefill_ids.shape[-1]],
            query_len=query_len,
            answer_len=0,
            variant=mask_variant,
            bridge_k=bridge_k,
        )
        prefill_mask = (
            to_additive(prefill_allow, dtype=model.dtype)
            .to(model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        positions_full = (
            build_positions(doc_lengths, query_len, answer_len, layout)
            .unsqueeze(0)
            .to(model.device)
        )
        positions_prefill = (
            build_positions(doc_lengths, query_len, 0, layout)
            .unsqueeze(0)
            .to(model.device)
        )

        with torch.no_grad():
            logits = forward_with_mask(
                model, full_ids, additive, position_ids=positions_full
            )
        loss = shifted_cross_entropy(logits, labels).item()

        pred = _generate_with_prefill_mask_and_positions(
            model,
            tokenizer,
            prefill_ids,
            prefill_mask,
            positions_prefill,
            max_new_tokens,
        )
        em = exact_match(pred, record["possible_answers"])
        f1 = token_f1(pred, record["possible_answers"])
        presence = answer_presence(pred, record["possible_answers"])

        rid = _build_run_id(
            model_name=model_name,
            dataset=dataset,
            variant_tag=variant_tag,
            layout=layout,
            distance=distance,
            pair_position=pair_position,
            seed=seed,
            git_sha_short=git_sha_short,
            record_id=record["id"],
        )
        append(rid, "loss", loss, prov)
        append(rid, "em", em, prov)
        append(rid, "f1", f1, prov)
        append(rid, "presence", presence, prov)


register_runner("phase5_composition")(_run)
