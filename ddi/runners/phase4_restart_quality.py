"""Phase 4.1 position-restart quality runner (issue #22).

Registers `task: phase4_restart_quality`. For each record, runs the phase1-baseline-style eval
(loss + generation metrics) under BOTH `layout=sequential` and `layout=restart` at the same mask
variant, so downstream analysis can measure whether restart costs anything at eval time. This is
the quality gate for the position-layout primitive that unlocks C4 (per-document KV caching):
under restart, a document's KV block is a pure function of its content, so it becomes cache-keyable
across requests regardless of retrieval position.

The mask-injection + prefill-mask generation loop mirror `phase2_train_free`; the only additions
are (i) `position_ids` built by `ddi.positions.build_positions`, (ii) an outer loop over layouts.
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
    """Greedy decode: DDI-masked prefill with explicit positions, then cached-KV steps.

    Only the prefill carries the custom positions — HF's decode-side path derives the next
    token's absolute position from the cached KV length, so passing `position_ids` per step
    would fight that machinery. Same reason `phase2_train_free` doesn't pass a mask on decode.
    """
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


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one cell of the restart-quality comparison; see module docstring."""
    dataset = config["dataset"]
    model_name = config["model"]
    if "mask" not in config:
        raise ValueError("phase4_restart_quality config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase4_restart_quality config.mask missing required key: variant")
    mask_variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")
    if mask_variant == "ddi_bridge_k" and bridge_k is None:
        raise ValueError("mask.variant='ddi_bridge_k' requires mask.bridge_k to be set")
    if mask_variant != "ddi_bridge_k" and bridge_k is not None:
        raise ValueError(
            f"mask.bridge_k is only valid for variant='ddi_bridge_k', not {mask_variant!r}"
        )

    layouts_cfg = config.get("layouts", ["sequential", "restart"])
    if not isinstance(layouts_cfg, list) or not layouts_cfg:
        raise ValueError("phase4_restart_quality config.layouts must be a non-empty list")
    layouts: list[Layout] = []
    for layout in layouts_cfg:
        if layout not in ("sequential", "restart"):
            raise ValueError(
                f"unknown layout {layout!r}; expected 'sequential' or 'restart'"
            )
        layouts.append(layout)  # type: ignore[arg-type]

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

    coerced_full = 0

    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break
        if not record["possible_answers"]:
            print(f"[phase4_restart_quality] skip record id={record['id']}: no possible_answers")
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

        # Same coercion rule as phase2_train_free: a record with no documents has nothing for
        # DDI to isolate, so any DDI variant is definitionally equal to full. Position layout is
        # equivalently a no-op there (sequential and restart both reduce to [0..T-1]) but we
        # still loop over layouts so the store rows exist for both.
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

        allow = build_mask(
            doc_lengths or [full_ids.shape[-1]],
            query_len=query_len,
            answer_len=answer_len,
            variant=variant_here,
            bridge_k=bridge_k_here,
        )
        additive = (
            to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
        )

        prefill_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)
        prefill_allow = build_mask(
            doc_lengths or [prefill_ids.shape[-1]],
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

        for layout in layouts:
            # Empty doc_lengths is fine: build_positions collapses to sequential [0..T-1] under
            # both layouts, which matches HF's default and the mask fallback's semantics
            # (nothing to isolate → nothing to reindex).
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
                logits = forward_with_mask(model, full_ids, additive, position_ids=positions_full)
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

            rid = (
                f"phase4_restart:{model_name}:{dataset}:{variant_tag}:{layout}:"
                f"seed={seed}:{prov.git_sha[:8]}:{record['id']}"
            )
            append(rid, "loss", loss, prov)
            append(rid, "em", em, prov)
            append(rid, "f1", f1, prov)
            append(rid, "presence", presence, prov)

    if coerced_full > 0:
        print(
            f"[phase4_restart_quality] coerced variant={mask_variant!r} -> 'full' on "
            f"{coerced_full} record(s) with no info_contexts"
        )


register_runner("phase4_restart_quality")(_run)
