"""Phase 4.3 order-invariance runner (issue #24).

Registers `task: phase4_order_invariance`. For each record, this runner prefills the
prompt under a *reference* retrieval order, records the last-token logits of the query
region, then reorders `record["info_contexts"]` `n_permutations - 1` times and compares
each permutation's last-token logits to the reference via `torch.allclose(..., atol)`.

The hypothesis (paper §8, issue #24): `ddi_qf + layout=restart` is order-invariant by
construction — a document's KV block is a pure function of its content under restart,
and `ddi_qf` blocks cross-document attention — so permuting the doc order must yield
bit-identical (within bf16 tolerance) logits. `layout=sequential` is included as a
control: under sequential, positions leak into KVs and doc order matters.

The runner writes two per-(record, layout) metrics:
  - `invariance` = 1.0 if every permutation matched the reference within tolerance,
    else 0.0.
  - `max_delta` = the maximum `(logits - reference).abs().max()` observed across all
    permutations (a continuous badness score for the F-report layer).

Assertion of the hypothesis is intentionally left to the report layer — this runner
just writes rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ddi.cli import register_runner
from ddi.data import load
from ddi.masks import build_mask, to_additive
from ddi.model import forward_with_mask, load_model
from ddi.positions import Layout, build_positions
from ddi.provenance import capture
from ddi.store import append


def _last_query_logits(
    logits: torch.Tensor, doc_total: int, query_len: int
) -> torch.Tensor:
    """Return the last-token logits of the query region: `logits[:, doc_total+query_len-1, :]`.

    Extracted as a helper so a unit test can exercise the slice arithmetic without a
    real model in the loop.
    """
    if query_len < 1:
        raise ValueError(f"query_len must be >= 1, got {query_len}")
    idx = doc_total + query_len - 1
    if idx >= logits.shape[-2]:
        raise ValueError(
            f"query-tail index {idx} out of bounds for logits with seq len {logits.shape[-2]}"
        )
    return logits[:, idx, :].detach().cpu()


def _prefill_last_logits(
    model,
    tokenizer,
    doc_texts: list[str],
    query_ids: torch.Tensor,
    query_len: int,
    variant: str,
    bridge_k: int | None,
    layout: Layout,
) -> torch.Tensor:
    """Tokenise docs in the given order, build mask + positions, run one forward,
    return last-token query logits on CPU."""
    doc_ids_list = [
        tokenizer(doc, return_tensors="pt", add_special_tokens=False).input_ids
        for doc in doc_texts
    ]
    doc_lengths = [ids.shape[-1] for ids in doc_ids_list]
    doc_total = sum(doc_lengths)

    prefill_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)

    allow = build_mask(
        doc_lengths or [prefill_ids.shape[-1]],
        query_len=query_len,
        answer_len=0,
        variant=variant,
        bridge_k=bridge_k,
    )
    additive = (
        to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
    )

    positions = (
        build_positions(doc_lengths or [prefill_ids.shape[-1]], query_len, 0, layout)
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.no_grad():
        logits = forward_with_mask(model, prefill_ids, additive, position_ids=positions)
    return _last_query_logits(logits, doc_total, query_len)


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one cell of the order-invariance comparison; see module docstring."""
    dataset = config["dataset"]
    model_name = config["model"]

    if "mask" not in config:
        raise ValueError("phase4_order_invariance config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase4_order_invariance config.mask missing required key: variant")
    mask_variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")
    if mask_variant == "ddi_bridge_k" and bridge_k is None:
        raise ValueError("mask.variant='ddi_bridge_k' requires mask.bridge_k to be set")
    if mask_variant != "ddi_bridge_k" and bridge_k is not None:
        raise ValueError(
            f"mask.bridge_k is only valid for variant='ddi_bridge_k', not {mask_variant!r}"
        )
    # Only mask variants that isolate cross-document attention can be order-invariant. `full`
    # / `blind` would either mix documents (full) or ignore them entirely (blind); running them
    # through this runner conflates the two effects.
    if mask_variant not in ("ddi_qf", "ddi_bridge_k"):
        raise ValueError(
            f"phase4_order_invariance only supports 'ddi_qf' and 'ddi_bridge_k'; got {mask_variant!r}"
        )

    layouts_cfg = config.get("layouts", ["sequential", "restart"])
    if not isinstance(layouts_cfg, list) or not layouts_cfg:
        raise ValueError("phase4_order_invariance config.layouts must be a non-empty list")
    layouts: list[Layout] = []
    for layout in layouts_cfg:
        if layout not in ("sequential", "restart"):
            raise ValueError(
                f"unknown layout {layout!r}; expected 'sequential' or 'restart'"
            )
        layouts.append(layout)  # type: ignore[arg-type]

    n_permutations = int(config.get("n_permutations", 4))
    if n_permutations < 2:
        raise ValueError(f"n_permutations must be >= 2, got {n_permutations}")
    tolerance = float(config.get("tolerance", 1e-3))
    split = config.get("split", "val")
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

    skipped_short = 0
    for i, record in enumerate(load(dataset, split, seed)):
        if max_records is not None and i >= max_records:
            break

        doc_texts: list[str] = list(record["info_contexts"])
        n_docs = len(doc_texts)
        # Nothing to permute with < 2 docs — the row would be trivially invariant for either
        # layout, so it carries no signal about the hypothesis. Skip and count.
        if n_docs < 2:
            skipped_short += 1
            continue

        user_dialog = record["dialog"][:-1]
        query_text = tokenizer.apply_chat_template(
            user_dialog, tokenize=False, add_generation_prompt=True
        )
        query_ids = tokenizer(
            query_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
        query_len = query_ids.shape[-1]

        for layout in layouts:
            reference = _prefill_last_logits(
                model,
                tokenizer,
                doc_texts,
                query_ids,
                query_len,
                mask_variant,
                bridge_k,
                layout,
            )

            all_match = True
            max_delta = 0.0
            for perm_i in range(n_permutations - 1):
                rng = np.random.default_rng(seed + perm_i)
                perm = rng.permutation(n_docs)
                permuted_texts = [doc_texts[int(j)] for j in perm]

                candidate = _prefill_last_logits(
                    model,
                    tokenizer,
                    permuted_texts,
                    query_ids,
                    query_len,
                    mask_variant,
                    bridge_k,
                    layout,
                )
                delta = float((candidate - reference).abs().max().item())
                if delta > max_delta:
                    max_delta = delta
                if not torch.allclose(candidate, reference, atol=tolerance):
                    all_match = False

            rid = (
                f"phase4_order:{model_name}:{dataset}:{variant_tag}:{layout}:"
                f"seed={seed}:{prov.git_sha[:8]}:{record['id']}"
            )
            append(rid, "invariance", 1.0 if all_match else 0.0, prov)
            append(rid, "max_delta", max_delta, prov)

    if skipped_short > 0:
        print(
            f"[phase4_order_invariance] skipped {skipped_short} record(s) with < 2 "
            f"info_contexts (nothing to permute)"
        )


register_runner("phase4_order_invariance")(_run)
