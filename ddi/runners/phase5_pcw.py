"""Phase 5.2 Parallel Context Windows baseline runner (issue #27).

Implements the Parallel Context Windows (PCW) setup of Ratner et al., ACL 2023
("Parallel Context Windows for Large Language Models") as a first-class baseline
for DDI. PCW is the closest prior work (see RESEARCH_PLAN §2): each retrieved
document forms an independent context window whose RoPE positions restart at 0,
cross-window attention is blocked on the key/value side, and the query attends
to every window.

In DDI's machinery this is exactly `variant=ddi_qf` + `layout=restart`:

- `ddi_qf` isolates documents from each other (block-diagonal within the doc
  region) while letting every query token attend to every doc — matching PCW's
  "windows are independent, query bridges them".
- `restart` reindexes each doc's positions to start at 0 — matching PCW's
  per-window RoPE reset.

This runner wraps that combo with the PCW label so store rows land under a
distinct `phase5_pcw:{model}:{dataset}:...` prefix and the paper's PCW column
in results tables is a first-class citizen, cleanly separated from the
`phase2_train_free:` and `phase4_restart:` sweeps that vary these axes.

The mask variant and position layout are hard-coded here: the paper defines the
setup, so a `variant`/`layout`/`bridge_k` config knob would be a category
error. A follow-up could add an explicit `variant_override` config key for
ablations if we ever want to peel the PCW label away from its DDI mechanics.

Follow-up: the greedy prefill-mask generation helper below is a copy of the
one in `phase4_restart_quality`; both should be factored into
`ddi/runners/_shared.py` alongside `shifted_cross_entropy` once a third caller
lands.
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
from ddi.positions import build_positions
from ddi.provenance import capture
from ddi.runners._shared import shifted_cross_entropy
from ddi.store import append

# PCW's setup in DDI terms. Locked constants — not config knobs; see module docstring.
_PCW_VARIANT = "ddi_qf"
_PCW_BRIDGE_K: int | None = None
_PCW_LAYOUT = "restart"


def _generate_with_prefill_mask_and_positions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefill_ids: torch.Tensor,
    prefill_mask: torch.Tensor,
    prefill_positions: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Greedy decode: DDI-masked prefill with explicit positions, then cached-KV steps.

    Copy of the helper in `phase4_restart_quality` — should be shared via `_shared.py`
    once a third caller appears (see module docstring).
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


def _make_run_id_prefix(model_name: str, dataset: str, seed: int, git_sha: str) -> str:
    """Build the `phase5_pcw:...` run_id prefix.

    Factored so tests can assert the shape without instantiating a model. No `variant_tag`
    segment: PCW is a single fixed setup, not a family of variants like phase2/phase4.
    """
    return f"phase5_pcw:{model_name}:{dataset}:seed={seed}:{git_sha[:8]}"


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one cell of the PCW baseline; see module docstring."""
    if "model" not in config:
        raise ValueError("phase5_pcw config missing required key: model")
    if "dataset" not in config:
        raise ValueError("phase5_pcw config missing required key: dataset")
    dataset = config["dataset"]
    model_name = config["model"]

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
    run_id_prefix = _make_run_id_prefix(model_name, dataset, seed, prov.git_sha)

    coerced_full = 0

    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break
        if not record["possible_answers"]:
            print(f"[phase5_pcw] skip record id={record['id']}: no possible_answers")
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

        # No documents → nothing to isolate → PCW's ddi_qf collapses to full. Same coercion
        # rule as phase2_train_free and phase4_restart_quality; restart layout also reduces to
        # sequential when there are no docs so the position path stays consistent.
        variant_here = _PCW_VARIANT
        bridge_k_here = _PCW_BRIDGE_K
        if not doc_lengths:
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

        positions_full = (
            build_positions(doc_lengths, query_len, answer_len, _PCW_LAYOUT)
            .unsqueeze(0)
            .to(model.device)
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
        positions_prefill = (
            build_positions(doc_lengths, query_len, 0, _PCW_LAYOUT)
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

        rid = f"{run_id_prefix}:{record['id']}"
        append(rid, "loss", loss, prov)
        append(rid, "em", em, prov)
        append(rid, "f1", f1, prov)
        append(rid, "presence", presence, prov)

    if coerced_full > 0:
        print(
            f"[phase5_pcw] coerced variant={_PCW_VARIANT!r} -> 'full' on "
            f"{coerced_full} record(s) with no info_contexts"
        )


register_runner("phase5_pcw")(_run)
