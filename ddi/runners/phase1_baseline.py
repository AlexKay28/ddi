"""Phase 1.4 full-causal baseline runner (issue #12).

Registers `task: phase1_full_baseline`. For each record in a dataset split, computes
answer-only cross-entropy loss under a full causal mask (routed through the DDI mask
injection path so the baseline is measured on the same code path as every DDI variant)
and greedy-generation EM / F1 / answer-presence. Every metric is appended to the store
with full provenance so `ddi.delta.compute_delta` can later group by seed and fix `Δ`.

Ground rule 9: this runner is what makes `Δ` mechanically real — a DDI number cannot
be interpreted as parity or non-parity until `full`'s seed-to-seed variance from these
rows has produced `Δ`.
"""

from __future__ import annotations

from typing import Any

import torch

from ddi.cli import register_runner
from ddi.data import load
from ddi.masks import build_mask, to_additive
from ddi.metrics import answer_presence, exact_match, make_answer_labels, token_f1
from ddi.model import forward_with_mask, load_model
from ddi.provenance import capture
from ddi.runners._shared import shifted_cross_entropy
from ddi.store import append


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run the full-causal baseline; see module docstring for what this measures."""
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
    run_id = f"phase1_baseline:{model_name}:{dataset}:seed={seed}:{prov.git_sha[:8]}"

    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break

        if not record["possible_answers"]:
            # Malformed for scoring: metrics require non-empty candidates. Skip loudly.
            print(f"[phase1_baseline] skip record id={record['id']}: no possible_answers")
            continue

        user_dialog = record["dialog"][:-1]
        answer = record["dialog"][-1]["content"]
        prompt_text = tokenizer.apply_chat_template(
            user_dialog, tokenize=False, add_generation_prompt=True
        )
        if record["info_contexts"]:
            prompt_text = "\n\n".join(record["info_contexts"]) + "\n\n" + prompt_text

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        answer_ids = tokenizer(
            answer, return_tensors="pt", add_special_tokens=False
        ).input_ids
        full_ids = torch.cat([prompt_ids, answer_ids], dim=-1).to(model.device)

        prompt_len = prompt_ids.shape[-1]
        total_len = full_ids.shape[-1]

        # Answer-only labels: -100 everywhere except answer positions, so cross-entropy is
        # computed on answer tokens only (§9).
        labels = make_answer_labels(full_ids, prompt_len, total_len).to(model.device)

        # Full causal mask via the DDI mask builder — same code path as any DDI variant, so
        # the baseline is not accidentally a different implementation.
        allow = build_mask([total_len], query_len=0, answer_len=0, variant="full")
        additive = (
            to_additive(allow, dtype=model.dtype)
            .to(model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            logits = forward_with_mask(model, full_ids, additive)

        # Shared next-token-shifted CE — Phase 2.1 must use the exact same formula so `Δ`
        # (baseline vs. train-free) measures the mask, not a divergent loss formulation.
        loss = shifted_cross_entropy(logits, labels).item()

        # Generation metrics — greedy decode. `model.generate` handles its own mask; on the
        # full baseline we do not need to inject one. Later phases will need a
        # generate-with-mask path for DDI variants.
        with torch.no_grad():
            gen_ids = model.generate(
                prompt_ids.to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        pred = tokenizer.decode(
            gen_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        em = exact_match(pred, record["possible_answers"])
        f1 = token_f1(pred, record["possible_answers"])
        presence = answer_presence(pred, record["possible_answers"])

        record_run_id = f"{run_id}:{record['id']}"
        append(record_run_id, "loss", loss, prov)
        append(record_run_id, "em", em, prov)
        append(record_run_id, "f1", f1, prov)
        append(record_run_id, "presence", presence, prov)


register_runner("phase1_full_baseline")(_run)
