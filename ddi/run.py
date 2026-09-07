"""YAML-driven experiment runner (issue #85, Phase 1 layer 2).

One CLI entry — `python -m ddi.run <exp.yaml>` — reads a YAML file, dispatches
on the `kind:` field to one of three built-in modes (`quality`, `timing`,
`cache`), and writes rows to a JSONL store with full provenance.

Replaces every `ddi/runners/phase*.py` file (to be deleted in #89). Each
dispatcher preserves the store-row prefix and metric names of the runner it
replaces so `ddi.aggregate` (#86) and the existing `ddi.reports.*` renderers
can consume rows from either surface without change.

Design:
  - Loud on missing / malformed YAML keys — no silent defaults for `model`,
    `datasets`, `masks`, etc.  A silent default would let a typo in a YAML
    silently rebrand a sweep.
  - Per-cell isolation: one bad cell (a dataset that fails to download, a
    mask variant HF doesn't like on this shape) logs to stderr and the loop
    continues to the next cell.  A sweep of 30 cells must not die on cell 3.
  - Provenance stamped per row via `ddi.provenance.capture`.  A row is traceable
    back to its config through `config_hash` — a SHA256 over the YAML *bytes*
    (`load_yaml`), which is stronger than the source path: two copies of the same
    sweep at different paths join, and an edited YAML at the same path does not.
    The path itself never reaches the store; it appears only in the startup log
    line and in `--set` validation errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ddi import inference
from ddi.data import load as load_dataset
from ddi.data.manifest import select_or_write
from ddi.masks import build_mask, to_additive
from ddi.metrics import (
    answer_presence,
    exact_match,
    make_answer_labels,
    token_f1,
)
from ddi.predictions import append as append_prediction
from ddi.prompt import build_query_text, encode_segments
from ddi.provenance import capture
from ddi.store import DEFAULT_STORE, append
from ddi.timing import Timing, append_timing, measure_decode, measure_prefill


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token-shifted cross-entropy with `-100` ignored.

    Predicts position `t` from logits at position `t-1`. `labels` is expected to
    be a `[batch, seq]` tensor with `-100` outside the answer span (see
    ``ddi.metrics.make_answer_labels``) so cross-entropy is restricted to answer
    tokens. Inlined from the deleted ``ddi.runners._shared`` in #89 so the
    formula stays bit-identical to the pre-#89 baseline / train-free rows.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

# ---------------------------------------------------------------------------
# YAML loading + seed helper.
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    """Read a YAML file and return (config, config_hash).

    `config_hash` is a short SHA256 of the raw bytes, folded into every store
    row's provenance so `results/runs.jsonl` is joinable back to the exact
    YAML text — not just the file name.
    """
    raw = path.read_bytes()
    cfg_hash = hashlib.sha256(raw).hexdigest()[:16]
    parsed = yaml.safe_load(raw) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML at {path} is not a mapping")
    return parsed, cfg_hash


def _require(cfg: dict[str, Any], key: str, ctx: str) -> Any:
    """Loud-on-missing-key accessor.  `ctx` names the dispatcher for error clarity."""
    if key not in cfg:
        raise ValueError(f"{ctx}: missing required key: {key!r}")
    return cfg[key]


def _seed_all(seed: int) -> None:
    """Seed `random`, `numpy`, `torch` (+ CUDA if available).

    Was "mirrors ddi.cli.set_seed"; `ddi/cli.py` is deleted, so this is the only seeder.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _store_path_for(cfg: dict[str, Any]) -> Path:
    """Resolve the JSONL store path; default matches ddi.store.DEFAULT_STORE."""
    raw = cfg.get("output_store")
    return Path(raw) if raw else DEFAULT_STORE


def _log(msg: str) -> None:
    """Print a progress line to stderr with the [ddi.run] prefix (flushed)."""
    print(f"[ddi.run] {msg}", file=sys.stderr, flush=True)


def _log_cell_failure(cell_desc: str, exc: BaseException) -> None:
    """Log a per-cell failure to stderr without aborting the outer sweep loop.

    Counting lives here, not in `_CellTracker.fail`, because this is the one function
    every failure path already goes through. When the count lived only in the tracker,
    the three `kind: cache` tasks — which log failures directly and never build a
    tracker — were outside the guard: a cache run whose every cell raised printed
    tracebacks, wrote no rows and still exited 0. A guard over four of seven failure
    sites is worse than none, because it reads as covering all of them. Anything that
    reports a cell failure at all is now counted, including cells added later.
    """
    _CellTracker.failed_total += 1
    print(f"[ddi.run] cell FAILED: {cell_desc}: {exc!r}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


class _CellTracker:
    """Bookkeeping for per-cell tick/timing + end-of-run summary.

    Prints:
      - `cell i/N: <desc>` when a cell starts.
      - `  → OK (T.Ts)` or `  → FAILED` when it ends.
      - `done — P passed, F failed in T.Ts total` on `.summary(kind)`.
    """

    # Process-wide count of failed cells, across every dispatcher.
    #
    # Cells are deliberately isolated — one bad cell must not abort the rest — but the
    # isolation was quietly turning into indifference: each dispatcher counted its failures
    # and printed them, then `run()` returned None and `main()` returned 0. A run in which
    # every single cell raised still exited 0 and reported success to its caller, so a
    # Nirvana finetune block would go green with no checkpoint written and the eval
    # downstream would score a stale model. Counting keeps per-cell isolation while
    # letting `main()` fail the process. The counter is incremented by
    # `_log_cell_failure`, not by `fail()` below, so the dispatchers that report a
    # failure without owning a tracker are covered too — see that function.
    failed_total = 0

    def __init__(self, kind: str, total: int) -> None:
        self.kind = kind
        self.total = total
        self.i = 0
        self.passed = 0
        self.failed = 0
        self.t0 = time.perf_counter()
        _log(f"{kind}: starting {total} cells")

    def start(self, desc: str) -> float:
        self.i += 1
        _log(f"  cell {self.i}/{self.total}: {desc}")
        return time.perf_counter()

    def ok(self, t: float) -> None:
        self.passed += 1
        _log(f"    → OK ({time.perf_counter() - t:.1f}s)")

    def fail(self, t: float) -> None:
        # Per-tracker bookkeeping only. `failed_total` is owned by `_log_cell_failure`,
        # which every caller of this method invokes first; counting in both would double
        # the number `main()` prints.
        self.failed += 1
        _log(f"    → FAILED ({time.perf_counter() - t:.1f}s)")

    def summary(self) -> None:
        _log(
            f"{self.kind}: done — {self.passed} passed, {self.failed} failed "
            f"in {time.perf_counter() - self.t0:.1f}s"
        )


# ---------------------------------------------------------------------------
# kind: quality — mirrors ddi/runners/phase2_train_free.py store rows.
# ---------------------------------------------------------------------------


def _variant_tag(variant: str, bridge_k: int | None) -> str:
    return variant if bridge_k is None else f"{variant}_k{bridge_k}"


def _mask_kwargs(variant: str, bridge_k: int | None) -> dict[str, Any]:
    """`ddi.masks.build_mask`'s bridge_k is only valid for `ddi_bridge_k`."""
    if variant == "ddi_bridge_k":
        if bridge_k is None:
            raise ValueError("mask 'ddi_bridge_k' requires bridge_k to be set")
        return {"bridge_k": bridge_k}
    if bridge_k is not None:
        raise ValueError(
            f"bridge_k is only valid for variant='ddi_bridge_k', not {variant!r}"
        )
    return {}


def _parse_mask_spec(spec: Any) -> tuple[str, int | None]:
    """Accept either a plain string (`'ddi_qf'`) or `{variant, bridge_k}` mapping."""
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, dict):
        variant = spec.get("variant")
        if not variant:
            raise ValueError(f"mask spec {spec!r} missing 'variant' key")
        return variant, spec.get("bridge_k")
    raise TypeError(f"mask spec must be a string or mapping, got {type(spec).__name__}")


def _quality_cell(
    *,
    model,
    tokenizer,
    dataset: str,
    mask_variant: str,
    bridge_k: int | None,
    model_name: str,
    seed: int,
    max_records: int | None,
    max_new_tokens: int,
    metrics: list[str],
    cfg_hash: str,
    store_path: Path,
    heldout_manifest: Path | None,
    heldout_manifest_mode: str = "write",
    predictions_store: Path | None = None,
    checkpoint: str | None = None,
) -> None:
    """One (dataset, mask) cell of the quality sweep.

    Store row shape matches ddi/runners/phase2_train_free.py exactly:
      - run_id prefix: `phase2_train_free:{model}:{dataset}:{variant_tag}:seed=…:{sha}`
      - per-record rows keyed by `loss`, `em`, `f1`, `presence`
    """
    prov = capture(
        config_hash=cfg_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=dataset,
        checkpoint_revision=checkpoint,
    )
    tag = _variant_tag(mask_variant, bridge_k)
    run_id_prefix = (
        f"phase2_train_free:{model_name}:{dataset}:{tag}:seed={seed}:{prov.git_sha[:8]}"
    )

    # Per-metric flag lookup so a caller can request e.g. just `f1` and skip loss.
    want_loss = "loss" in metrics
    want_em = "em" in metrics
    want_f1 = "f1" in metrics
    want_presence = "presence" in metrics

    coerced_full = 0

    records = list(load_dataset(dataset, "val", seed))
    if heldout_manifest is not None:
        eligible = [record for record in records if record["possible_answers"]]
        manifest_name = f"{dataset}__seed{seed}__val"
        selected_ids = select_or_write(
            manifest_name,
            [record["id"] for record in eligible],
            max_records,
            heldout_manifest,
            allow_create=heldout_manifest_mode == "write",
        )
        records_by_id = {record["id"]: record for record in eligible}
        records = [records_by_id[record_id] for record_id in selected_ids]

    for i, record in enumerate(records):
        if heldout_manifest is None and max_records is not None and i >= max_records:
            break
        if not record["possible_answers"]:
            print(
                f"[ddi.run:quality] skip record id={record['id']}: no possible_answers",
                file=sys.stderr,
            )
            continue

        doc_texts = list(record["info_contexts"])
        # Match phase2_train_free's coercion: DDI variants have nothing to isolate
        # on a no-doc record, so any DDI variant collapses to `full` on those rows.
        variant_here = mask_variant
        bridge_k_here = bridge_k
        if not doc_texts and mask_variant != "full":
            variant_here = "full"
            bridge_k_here = None
            coerced_full += 1

        answer_text = record["dialog"][-1]["content"]

        # Single source of truth for prompt text (ddi.prompt) — eval, perplexity and
        # training must see a byte-identical query or a prompt mismatch shows up as a
        # quality regression that has nothing to do with the mask.
        query_text = build_query_text(tokenizer, record)

        # Loss computation needs the answer tokens concatenated in and a masked labels
        # tensor — that's the phase2_train_free contract.  Use build_mask/to_additive
        # directly here (rather than inference.encode) so we can splice the answer in;
        # the tokenisation itself goes through the same `ddi.prompt` encoder that
        # inference.encode, the trainer and the perplexity path use.
        if want_loss:
            full_ids, doc_lengths, query_len, answer_len = encode_segments(
                tokenizer, doc_texts, query_text, answer_text
            )
            full_ids = full_ids.to(model.device)
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
                **_mask_kwargs(variant_here, bridge_k_here),
            )
            additive = (
                to_additive(allow, dtype=model.dtype)
                .to(model.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            with torch.no_grad():
                out = model(
                    input_ids=full_ids,
                    attention_mask=additive,
                    use_cache=False,
                )
            loss_val = float(shifted_cross_entropy(out.logits, labels).item())
        else:
            loss_val = None

        # Generation: use inference.encode + inference.generate.  This is exactly the
        # phase2 prefill-mask greedy loop, moved into ddi.inference by #84.
        batch = inference.encode(
            tokenizer,
            docs=doc_texts,
            query=query_text,
            layout="sequential",
        )
        preds = inference.generate(
            model,
            tokenizer,
            batch,
            mask_variant=variant_here,
            max_new_tokens=max_new_tokens,
            mask_kwargs=_mask_kwargs(variant_here, bridge_k_here),
        )
        pred = preds[0]

        rid = f"{run_id_prefix}:{record['id']}"
        result_metrics: dict[str, float] = {}
        if want_loss and loss_val is not None:
            append(rid, "loss", loss_val, prov, store=store_path)
            result_metrics["loss"] = loss_val
        if want_em:
            em = exact_match(pred, record["possible_answers"])
            append(rid, "em", em, prov, store=store_path)
            result_metrics["em"] = em
        if want_f1:
            f1 = token_f1(pred, record["possible_answers"])
            append(rid, "f1", f1, prov, store=store_path)
            result_metrics["f1"] = f1
        if want_presence:
            presence = answer_presence(pred, record["possible_answers"])
            append(rid, "presence", presence, prov, store=store_path)
            result_metrics["presence"] = presence
        if predictions_store is not None:
            append_prediction(
                rid,
                record["id"],
                pred,
                list(record["possible_answers"]),
                result_metrics,
                prov,
                predictions_store,
            )

    if coerced_full > 0:
        print(
            f"[ddi.run:quality] coerced variant={mask_variant!r} -> 'full' on "
            f"{coerced_full} record(s) with no info_contexts",
            file=sys.stderr,
        )


def run_quality(cfg: dict[str, Any], cfg_hash: str, seed: int) -> None:
    """Dispatch a `kind: quality` YAML across the (dataset x mask) grid."""
    model_name = _require(cfg, "model", "quality")
    datasets = _require(cfg, "datasets", "quality")
    masks = _require(cfg, "masks", "quality")
    metrics = _require(cfg, "metrics", "quality")

    max_records = cfg.get("max_records")
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    store_path = _store_path_for(cfg)
    heldout_manifest = (
        Path(cfg["heldout_manifest"]) if cfg.get("heldout_manifest") is not None else None
    )
    heldout_manifest_mode = cfg.get("heldout_manifest_mode", "write")
    predictions_store = (
        Path(cfg["predictions_store"]) if cfg.get("predictions_store") is not None else None
    )

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("quality: 'datasets' must be a non-empty list")
    if not isinstance(masks, list) or not masks:
        raise ValueError("quality: 'masks' must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("quality: 'metrics' must be a non-empty list")

    checkpoint = cfg.get("checkpoint")
    evaluation_model = cfg.get("evaluation_model", model_name)
    if checkpoint is not None and "evaluation_model" not in cfg:
        raise ValueError(
            "quality: checkpoint evaluation requires a unique 'evaluation_model' label so "
            "base and checkpoint rows cannot aggregate into one model column"
        )
    if not isinstance(evaluation_model, str) or not evaluation_model:
        raise ValueError("quality: 'evaluation_model' must be a non-empty string")
    if ":" in evaluation_model:
        raise ValueError("quality: 'evaluation_model' must not contain ':'")
    if re.search(r"(?:^|__)seed\d+(?:$|__)", evaluation_model):
        raise ValueError(
            "quality: 'evaluation_model' must identify the arm across seeds, not one seed"
        )
    if checkpoint is not None and evaluation_model == model_name:
        raise ValueError(
            "quality: checkpoint evaluation_model must differ from the base model identity"
        )
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError("quality: 'checkpoint' must be a non-empty string")
    if heldout_manifest_mode not in {"write", "verify"}:
        raise ValueError("quality: 'heldout_manifest_mode' must be 'write' or 'verify'")
    if heldout_manifest_mode == "verify" and heldout_manifest is None:
        raise ValueError("quality: verify-only manifests require 'heldout_manifest'")
    if checkpoint is not None and heldout_manifest_mode != "verify":
        raise ValueError(
            "quality: checkpoint evaluation requires heldout_manifest_mode='verify'"
        )
    rope_theta = cfg.get("rope_theta")
    if checkpoint is not None:
        _log(f"quality: loading model {model_name!r} from checkpoint {checkpoint!r}")
    else:
        _log(f"quality: loading model {model_name!r}")
    if rope_theta is not None:
        _log(f"quality: rope_theta override -> {rope_theta}")
    model, tokenizer = inference.load(
        model_name, checkpoint_dir=checkpoint, rope_theta=rope_theta
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    tracker = _CellTracker("quality", len(datasets) * len(masks))
    for dataset in datasets:
        for mask_spec in masks:
            mask_variant, bridge_k = _parse_mask_spec(mask_spec)
            cell_desc = f"dataset={dataset} mask={mask_variant} bridge_k={bridge_k}"
            t = tracker.start(cell_desc)
            try:
                _quality_cell(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    mask_variant=mask_variant,
                    bridge_k=bridge_k,
                    model_name=evaluation_model,
                    seed=seed,
                    max_records=max_records,
                    max_new_tokens=max_new_tokens,
                    metrics=metrics,
                    cfg_hash=cfg_hash,
                    store_path=store_path,
                    heldout_manifest=heldout_manifest,
                    heldout_manifest_mode=heldout_manifest_mode,
                    predictions_store=predictions_store,
                    checkpoint=checkpoint,
                )
                tracker.ok(t)
            except Exception as exc:  # per-cell isolation
                _log_cell_failure(f"quality({cell_desc})", exc)
                tracker.fail(t)
    tracker.summary()


# ---------------------------------------------------------------------------
# kind: timing — mirrors ddi/runners/phase3_timing.py store rows.
# ---------------------------------------------------------------------------


def _analytical_speedup(variant: str, N: int, L: int, Q: int) -> float:
    """RESEARCH_PLAN §7 analytical speedup — copy of phase3_timing._analytical_speedup.

    Duplicated on purpose: once #89 deletes phase3_timing.py, this stays as the
    single source of truth.  Blind / bridge_k fall through to `full` so the
    column stays populated for every row; aggregators filter on `mask_variant`.
    """
    T = N * L + Q
    full_flops = 0.5 * T * T
    if variant == "full":
        return 1.0
    if variant == "ddi_qf":
        our = 0.5 * N * L * L + Q * N * L + 0.5 * Q * Q
    elif variant == "ddi_strict":
        our = 0.5 * N * L * L + 0.5 * Q * Q + T
    else:
        our = full_flops
    return full_flops / our


def _sparsity(allow: torch.Tensor) -> float:
    """Fraction of causal pairs removed by the mask — matches phase3_timing._sparsity."""
    T = allow.shape[-1]
    causal_pairs = T * (T + 1) / 2
    return 1.0 - float(allow.sum().item()) / causal_pairs


def _measure_prefill_flex(
    model,
    input_ids: torch.Tensor,
    block_mask,
    n_repeats: int,
    warmup: int,
) -> Timing:
    """FTT for flex_attention backend.

    Sets ``model._flex_block_mask`` so the patched attention layers use the
    compiled flex_attention kernel. The first call triggers torch.compile
    warm-up (graph capture), which is why we run an explicit untimed warm-up
    before the timed repeats — the compilation cost must not land in a
    measurement.
    """
    from ddi.timing import _cuda_events_for, measure

    model._flex_block_mask = block_mask

    # Trigger torch.compile graph capture outside the timed region.
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def _fn() -> object:
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=False)

    return measure(
        _fn,
        n_repeats=n_repeats,
        warmup=warmup,
        name="prefill",
        use_cuda_events=_cuda_events_for(input_ids, None),
    )


def _timing_cell(
    *,
    model_name: str,
    mask_variant: str,
    bridge_k: int | None,
    N: int,
    L: int,
    Q: int,
    backend: str,
    n_repeats: int,
    warmup: int,
    n_new_tokens: int,
    seed: int,
    cfg_hash: str,
    store_path: Path,
    checkpoint: str | None = None,
) -> None:
    """One (mask, N, L, backend) grid cell.  Fresh model load per backend (matches phase3)."""
    if backend == "fa2" and not torch.cuda.is_available():
        raise RuntimeError(
            "backend='fa2' requires CUDA; torch.cuda.is_available() is False. "
            "Route this cell to a GPU worker or select backend='sdpa'."
        )

    # Fresh load per backend — HF picks attn kernel at from_pretrained time.
    model, _tokenizer = inference.load(model_name, backend=backend, checkpoint_dir=checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    vocab_size = model.config.vocab_size
    torch.manual_seed(seed)
    doc_lengths = [L] * N
    T_prefill = N * L + Q
    prefill_ids = torch.randint(0, vocab_size, (1, T_prefill), device=model.device)

    kwargs = _mask_kwargs(mask_variant, bridge_k)
    allow = build_mask(
        doc_lengths,
        query_len=Q,
        answer_len=0,
        variant=mask_variant,  # type: ignore[arg-type]
        **kwargs,
    )
    additive = (
        to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)
    )

    # flex backend uses a compiled BlockMask instead of a 4D additive mask.
    flex_block_mask = None
    if backend == "flex":
        from ddi.flex_attention import make_block_mask
        flex_block_mask = make_block_mask(
            doc_lengths, Q, mask_variant, device=model.device,
        )

    prov = capture(
        config_hash=cfg_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=f"synthetic_timing_N{N}_L{L}_Q{Q}",
    )
    tag = _variant_tag(mask_variant, bridge_k)
    run_id = (
        f"phase3_timing:{model_name}:N={N}:L={L}:Q={Q}:{tag}:{backend}:"
        f"seed={seed}:{prov.git_sha[:8]}"
    )

    sparsity = _sparsity(allow)
    append(run_id, "mask_sparsity", sparsity, prov, store=store_path)

    # Two independent measurements, deliberately not one. `measure_decode` runs a prefill of its
    # own each iteration — the decode loop needs a warm KV cache — but times only the decode steps,
    # so `tpot` is time-per-output-token and not `(FTT + n*TPOT) / n`. Do not "save a prefill" by
    # deriving one of these from the other: the prefill term is where mask sparsity pays off, so
    # folding it into `tpot` credits decode with a prefill speedup (see measure_decode's docstring).
    #
    # The four rows `append_timing` writes per metric — `{ftt,tpot}_{median_ms,iqr_low_ms,
    # iqr_high_ms,repeats_ms}` — are parsed by name in `ddi.aggregate`; keep the metric strings.
    if backend == "flex":
        ftt = _measure_prefill_flex(model, prefill_ids, flex_block_mask, n_repeats, warmup)
    else:
        ftt = measure_prefill(model, prefill_ids, additive, n_repeats=n_repeats, warmup=warmup)
    append_timing(store=store_path, run_id=run_id, metric="ftt", timing=ftt, provenance=prov)

    # Both flex and sdpa use the same sdpa decode path at Q_LEN=1. For flex,
    # reset the block_mask so patched layers fall back to sdpa (see
    # patch_model_for_flex).
    if backend == "flex":
        model._flex_block_mask = None  # type: ignore[attr-defined]
    tpot = measure_decode(
        model,
        prefill_ids,
        additive,
        n_new_tokens=n_new_tokens,
        n_repeats=n_repeats,
        warmup=warmup,
    )
    append_timing(store=store_path, run_id=run_id, metric="tpot", timing=tpot, provenance=prov)

    speedup = _analytical_speedup(mask_variant, N, L, Q)
    append(run_id, "analytical_speedup", speedup, prov, store=store_path)
    append(run_id, "backend_used", backend, prov, store=store_path)


def run_timing(cfg: dict[str, Any], cfg_hash: str, seed: int) -> None:
    """Dispatch a `kind: timing` YAML across the (mask x N x L x backend) grid."""
    model_name = _require(cfg, "model", "timing")
    Ns = _require(cfg, "Ns", "timing")
    Ls = _require(cfg, "Ls", "timing")
    backends = _require(cfg, "backends", "timing")
    masks = _require(cfg, "masks", "timing")

    # `Q` and `n_new_tokens` are held constant across a sweep; the schema keeps them
    # as top-level knobs, defaulting to values that match phase3_timing's config.
    Q = int(cfg.get("Q", 64))
    n_repeats = int(cfg.get("n_repeats", 5))
    warmup = int(cfg.get("warmup", 3))
    n_new_tokens = int(cfg.get("n_new_tokens", 16))
    store_path = _store_path_for(cfg)
    checkpoint = cfg.get("checkpoint")

    for lst_name, lst in (("Ns", Ns), ("Ls", Ls), ("backends", backends), ("masks", masks)):
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"timing: {lst_name!r} must be a non-empty list")

    tracker = _CellTracker("timing", len(masks) * len(Ns) * len(Ls) * len(backends))
    for mask_spec in masks:
        mask_variant, bridge_k = _parse_mask_spec(mask_spec)
        for N in Ns:
            for L in Ls:
                for backend in backends:
                    cell_desc = f"mask={mask_variant} N={N} L={L} backend={backend}"
                    t = tracker.start(cell_desc)
                    try:
                        _timing_cell(
                            model_name=model_name,
                            mask_variant=mask_variant,
                            bridge_k=bridge_k,
                            N=int(N),
                            L=int(L),
                            Q=Q,
                            backend=backend,
                            n_repeats=n_repeats,
                            warmup=warmup,
                            n_new_tokens=n_new_tokens,
                            seed=seed,
                            cfg_hash=cfg_hash,
                            store_path=store_path,
                            checkpoint=checkpoint,
                        )
                        tracker.ok(t)
                    except Exception as exc:
                        _log_cell_failure(f"timing({cell_desc})", exc)
                        tracker.fail(t)
                    finally:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    tracker.summary()


# ---------------------------------------------------------------------------
# kind: cache — dispatches on task: correctness | throughput | restart_quality.
# ---------------------------------------------------------------------------


def _cache_correctness(cfg: dict[str, Any], cfg_hash: str, seed: int, store_path: Path) -> None:
    """`cache/correctness` — synthetic (mode × N × L × query_i) grid vs full-causal reference.

    Rewritten in #105: the old dataset-driven runner emitted per-doc
    ``kv_deterministic`` / ``kv_cache_roundtrip`` rows keyed by record; no
    on-`main` YAML uses that shape any more, and ``cache_correctness.yaml``
    requires the mode/N/L grid instead. The new runner follows ``_timing_cell``:
    synthesise ``N`` docs of length ``L`` and one query of length ``Q``, then
    run one forward pass per mode and compare each mode's logits against
    ``full_causal`` — the baseline — via ``max_abs_logit_diff``.

    Store rows:
      - run_id: ``phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha}:q{i}``
      - metric: ``max_abs_logit_diff`` (``full_causal`` row is always 0.0).
    """
    from ddi.cache import (
        DocCache,
        compute_doc_kv,
        compute_key,
        forward_query_with_doc_cache,
    )
    from ddi.positions import build_positions

    model_name = _require(cfg, "model", "cache/correctness")
    modes = _require(cfg, "modes", "cache/correctness")
    Ns = _require(cfg, "Ns", "cache/correctness")
    Ls = _require(cfg, "Ls", "cache/correctness")
    n_queries_per_doc = int(cfg.get("n_queries_per_doc", 5))
    Q = int(cfg.get("Q", 64))
    answer_len = int(cfg.get("answer_len", 16))
    atol_check = float(cfg.get("atol_check", 1e-4))
    cache_dir = Path(cfg.get("cache_dir", "cache/doc_kv"))

    if not isinstance(modes, list) or not modes:
        raise ValueError("cache/correctness: 'modes' must be a non-empty list")
    if not isinstance(Ns, list) or not Ns:
        raise ValueError("cache/correctness: 'Ns' must be a non-empty list")
    if not isinstance(Ls, list) or not Ls:
        raise ValueError("cache/correctness: 'Ls' must be a non-empty list")

    valid_modes = {"full_causal", "ddi_recompute", "ddi_cache"}
    for m in modes:
        if m not in valid_modes:
            raise ValueError(
                f"cache/correctness: unknown mode {m!r}; expected {valid_modes}"
            )
    if "full_causal" not in modes:
        raise ValueError(
            "cache/correctness: 'full_causal' must be in modes (baseline reference)"
        )

    checkpoint = cfg.get("checkpoint")
    model, tokenizer = inference.load(model_name, checkpoint_dir=checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    vocab_size = model.config.vocab_size
    prov = capture(
        config_hash=cfg_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=f"synthetic_cache_correctness_Q={Q}_A={answer_len}",
    )
    # Cache is only exercised in `ddi_cache` mode but instantiated once so
    # per-doc hits accumulate across queries inside a single (N, L) shape.
    doc_cache = DocCache(cache_dir)
    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__

    def _full_causal_logits(input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[-1]
        allow = build_mask([T], query_len=0, answer_len=0, variant="full")
        additive = (
            to_additive(allow, dtype=model.dtype)
            .to(model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=additive,
                use_cache=False,
            )
        return out.logits.detach()

    def _ddi_recompute_logits(
        doc_ids_list: list[torch.Tensor],
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        doc_lengths = [int(t.shape[-1]) for t in doc_ids_list]
        query_len = int(query_ids.shape[-1])
        input_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)
        allow = build_mask(
            doc_lengths,
            query_len=query_len,
            answer_len=0,
            variant="ddi_qf",
        )
        additive = (
            to_additive(allow, dtype=model.dtype)
            .to(model.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        positions = (
            build_positions(doc_lengths, query_len, 0, "restart")
            .unsqueeze(0)
            .to(model.device)
        )
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=additive,
                position_ids=positions,
                use_cache=False,
            )
        return out.logits[:, -query_len:, :].detach()

    def _ddi_cache_logits(
        doc_texts: list[str],
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        entries = []
        for doc_text in doc_texts:
            key = compute_key(model_id, model.dtype, doc_text)
            entry = doc_cache.get(key)
            if entry is None:
                entry = compute_doc_kv(model, tokenizer, doc_text, "restart")
                doc_cache.put(entry)
            entries.append(entry)
        return forward_query_with_doc_cache(model, query_ids, entries).logits.detach()

    # `atol_check` is a `--set`-able config key (see `_SET_KEYS`), so it has to do
    # something. It stays advisory rather than raising: the store row carries the real
    # `max_abs_logit_diff` and a numerical tolerance is a judgement call that belongs to
    # whoever reads the sweep — but a silently-ignored tolerance is a knob that documents
    # a guarantee nobody checks, and this is the cell whose whole job is proving the
    # cached path reproduces the uncached one.
    over_tolerance = 0
    compared = 0

    for L in Ls:
        for N in Ns:
            # Synthesise per-shape docs + queries. Fresh generator per (L, N) so a
            # cell failure doesn't perturb the RNG state seen by subsequent cells.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + N * 1_000_003 + L)
            L_int = int(L)
            N_int = int(N)
            sampled_doc_ids = [
                torch.randint(0, vocab_size, (1, L_int), generator=gen) for _ in range(N_int)
            ]
            # Decoded doc text so DocCache keys per-doc content (str -> sha256), matching
            # the production API in ddi.cache.compute_key.
            doc_texts = [
                tokenizer.decode(t[0].tolist(), skip_special_tokens=True)
                for t in sampled_doc_ids
            ]
            # Both reference and cache paths must consume the retokenized text. Decode is
            # not generally tokenization-invertible, so comparing sampled ids on one side
            # with retokenized text on the other silently compares different requests.
            doc_id_tensors = [
                tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
                    model.device
                )
                for text in doc_texts
            ]

            for query_i in range(n_queries_per_doc):
                query_ids = torch.randint(0, vocab_size, (1, Q), generator=gen).to(model.device)
                full_input_ids = torch.cat(doc_id_tensors + [query_ids], dim=-1)

                # Compute both references once: DDI is compared with full causal, while
                # assembled cache is compared with the matching fresh DDI query logits.
                try:
                    ref_logits = _full_causal_logits(full_input_ids)[:, -Q:, :]
                    ddi_logits = _ddi_recompute_logits(doc_id_tensors, query_ids)
                except Exception as exc:
                    _log_cell_failure(
                        f"cache/correctness(baseline, N={N_int}, L={L_int}, q={query_i})",
                        exc,
                    )
                    continue

                for mode in modes:
                    cell_desc = (
                        f"cache/correctness(mode={mode}, N={N_int}, L={L_int}, q={query_i})"
                    )
                    try:
                        if mode == "full_causal":
                            diff = 0.0
                        elif mode == "ddi_recompute":
                            diff = float((ref_logits - ddi_logits).abs().max().item())
                        elif mode == "ddi_cache":
                            variant_logits = _ddi_cache_logits(doc_texts, query_ids)
                            diff = float(
                                (ddi_logits - variant_logits).abs().max().item()
                            )
                        else:  # unreachable — validated above.
                            raise AssertionError(f"unreachable mode {mode!r}")

                        if mode == "ddi_cache":
                            compared += 1
                            if diff > atol_check:
                                over_tolerance += 1

                        run_id = (
                            f"phase4_cache:{model_name}:{mode}:N={N_int}:L={L_int}:"
                            f"seed={seed}:{prov.git_sha[:8]}:q{query_i}"
                        )
                        append(
                            run_id,
                            "max_abs_logit_diff",
                            diff,
                            prov,
                            store=store_path,
                        )
                    except Exception as exc:
                        _log_cell_failure(cell_desc, exc)

    if over_tolerance:
        _log(
            f"cache/correctness: {over_tolerance}/{compared} cache row(s) exceeded "
            f"atol_check={atol_check:g} — assembled cache does NOT reproduce fresh DDI "
            f"recomputation at that tolerance"
        )
    elif compared:
        _log(f"cache/correctness: all {compared} row(s) within atol_check={atol_check:g}")


# ---------------------------------------------------------------------------
# Throughput inner loop — inlined from ddi/runners/phase4_throughput.py in #89.
#
# The helpers below (`_sample_doc_indices`, `_time_full_causal`, `_one_request`) were previously
# imported lazily from `ddi.runners.phase4_throughput`. That module is deleted
# by #89. The request implementations now execute their named paths end to end;
# historical `ddi_cache` and `ddi_recompute` timing semantics were invalid.
# ---------------------------------------------------------------------------


def _sample_doc_indices(
    rng: np.random.Generator,
    n_requests: int,
    docs_per_request: int,
    top_docs: int,
    distribution: str,
    zipf_a: float,
) -> list[list[int]]:
    """Return one list of `docs_per_request` distinct doc indices per request."""
    if distribution not in ("zipf", "uniform", "fixed"):
        raise ValueError(
            f"unknown reuse_distribution {distribution!r}; expected zipf | uniform | fixed"
        )
    if docs_per_request > top_docs:
        raise ValueError(
            f"docs_per_request={docs_per_request} must be <= top_docs={top_docs}"
        )

    if distribution == "fixed":
        base = list(range(docs_per_request))
        return [list(base) for _ in range(n_requests)]

    requests: list[list[int]] = []
    if distribution == "uniform":
        for _ in range(n_requests):
            requests.append(
                rng.choice(top_docs, size=docs_per_request, replace=False).tolist()
            )
        return requests

    ranks = np.arange(1, top_docs + 1, dtype=np.float64)
    weights = ranks ** (-zipf_a)
    weights /= weights.sum()
    for _ in range(n_requests):
        picked: list[int] = []
        seen: set[int] = set()
        for _ in range(docs_per_request * 20):
            idx = int(rng.choice(top_docs, p=weights))
            if idx in seen:
                continue
            seen.add(idx)
            picked.append(idx)
            if len(picked) == docs_per_request:
                break
        for tail in range(top_docs - 1, -1, -1):
            if len(picked) == docs_per_request:
                break
            if tail not in seen:
                seen.add(tail)
                picked.append(tail)
        requests.append(picked)
    return requests


def _time_full_causal(
    model,
    tokenizer,
    doc_texts: list[str],
    query_text: str,
    n_new_tokens: int,
) -> float:
    """Time one full-causal prefill on `docs + query` + greedy decode; returns elapsed ms."""
    doc_ids_list = [
        tokenizer(doc, return_tensors="pt", add_special_tokens=False).input_ids
        for doc in doc_texts
    ]
    query_ids = tokenizer(
        query_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    full_ids = torch.cat(doc_ids_list + [query_ids], dim=-1).to(model.device)
    T = full_ids.shape[-1]

    allow = build_mask([T], query_len=0, answer_len=0, variant="full")
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=additive, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
        for _ in range(n_new_tokens):
            step = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = step.past_key_values
            next_token = step.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _one_request(
    mode: str,
    model,
    tokenizer,
    doc_indices: list[int],
    docs: list[str],
    doc_keys: list[str],
    cache: Any,
    query_text: str,
    n_new_tokens: int,
    layout: str,
) -> tuple[int, int]:
    """Execute one request under `mode`; return (hits, lookups) for cache accounting.

    Non-cache modes always return (0, 0).
    """
    if mode == "full_causal":
        _time_full_causal(
            model, tokenizer, [docs[i] for i in doc_indices], query_text, n_new_tokens
        )
        return 0, 0

    request_docs = [docs[i] for i in doc_indices]
    if mode == "ddi_recompute":
        batch = inference.encode(tokenizer, request_docs, query_text, layout=layout)
        inference.generate(
            model, tokenizer, batch, "ddi_qf", max_new_tokens=n_new_tokens
        )
        return 0, 0

    if cache is None:
        raise ValueError("ddi_cache mode requires a cache")
    if layout != "restart":
        raise ValueError("ddi_cache mode requires layout='restart'")
    from ddi.cache import compute_doc_kv, generate_tokens_with_doc_cache

    entries = []
    hits = 0
    for doc_idx in doc_indices:
        entry = cache.get(doc_keys[doc_idx])
        if entry is None:
            entry = compute_doc_kv(model, tokenizer, docs[doc_idx], "restart")
            cache.put(entry)
        else:
            hits += 1
        entries.append(entry)
    query_ids = tokenizer(
        query_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    generate_tokens_with_doc_cache(
        model, tokenizer, query_ids, entries, max_new_tokens=n_new_tokens
    )
    return hits, len(doc_indices)


def _cache_throughput(cfg: dict[str, Any], cfg_hash: str, seed: int, store_path: Path) -> None:
    """One row-block per `mode` in `modes:`; row shape matches phase4_throughput exactly.

    Uses the same synthetic doc universe and per-request timing helpers as the
    original runner; only difference is the outer loop over `modes:` from the
    YAML rather than one runner invocation per mode.
    """
    from pathlib import Path as _Path

    from ddi.cache import DocCache, compute_key

    model_name = _require(cfg, "model", "cache/throughput")
    modes = _require(cfg, "modes", "cache/throughput")
    if not isinstance(modes, list) or not modes:
        raise ValueError("cache/throughput: 'modes' must be a non-empty list")

    layout = cfg.get("layout", "restart")
    if layout not in ("sequential", "restart"):
        raise ValueError(f"cache/throughput: unknown layout {layout!r}")

    n_requests = int(cfg.get("n_requests", 200))
    warmup_requests = int(cfg.get("warmup_requests", 20))
    reuse_distribution = cfg.get("reuse_distribution", "zipf")
    zipf_a = float(cfg.get("zipf_a", 1.5))
    top_docs = int(cfg.get("top_docs", 32))
    docs_per_request = int(cfg.get("docs_per_request", 5))
    doc_token_len = int(cfg.get("doc_token_len", 500))
    query_token_len = int(cfg.get("query_token_len", 32))
    n_new_tokens = int(cfg.get("n_new_tokens", 16))
    cache_dir = _Path(cfg.get("cache_dir", "cache/doc_kv"))

    if warmup_requests >= n_requests:
        raise ValueError(
            f"cache/throughput: warmup_requests={warmup_requests} must be < n_requests={n_requests}"
        )

    checkpoint = cfg.get("checkpoint")
    model, tokenizer = inference.load(model_name, checkpoint_dir=checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    rng = np.random.default_rng(seed)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    docs = [
        tokenizer.decode(
            rng.integers(low=0, high=vocab_size, size=doc_token_len).tolist(),
            skip_special_tokens=True,
        )
        for _ in range(top_docs)
    ]
    query_text = tokenizer.decode(
        rng.integers(low=0, high=vocab_size, size=query_token_len).tolist(),
        skip_special_tokens=True,
    )

    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    doc_keys = [compute_key(model_id, model.dtype, d) for d in docs]

    for mode in modes:
        cell_desc = f"cache/throughput(mode={mode})"
        try:
            if mode not in ("ddi_cache", "ddi_recompute", "full_causal"):
                raise ValueError(f"unknown mode {mode!r}")

            cache: DocCache | None = DocCache(cache_dir) if mode == "ddi_cache" else None

            # Independent doc-index stream per mode so warmup / measured requests
            # are directly comparable between modes even if the RNG has advanced.
            mode_rng = np.random.default_rng(seed)
            index_stream: list[list[int]] = _sample_doc_indices(
                mode_rng,
                n_requests + warmup_requests,
                docs_per_request,
                top_docs,
                reuse_distribution,
                zipf_a,
            )

            prov = capture(
                config_hash=cfg_hash,
                seed=seed,
                model_revision=model_name,
                dataset_revision=(
                    f"synthetic_throughput_top={top_docs}_N={docs_per_request}_"
                    f"L={doc_token_len}_Q={query_token_len}_A={n_new_tokens}_"
                    f"reuse={reuse_distribution}"
                ),
            )
            reuse_name = (
                f"{reuse_distribution}(a={zipf_a})"
                if reuse_distribution == "zipf"
                else reuse_distribution
            )
            reuse_tag = (
                f"{reuse_name},top={top_docs},N={docs_per_request},L={doc_token_len},"
                f"Q={query_token_len},A={n_new_tokens}"
            )
            run_id = (
                f"phase4_throughput:{model_name}:{mode}:{layout}:reuse={reuse_tag}:"
                f"seed={seed}:{prov.git_sha[:8]}"
            )

            for req_i in range(warmup_requests):
                _one_request(
                    mode, model, tokenizer, index_stream[req_i], docs, doc_keys,
                    cache, query_text, n_new_tokens, layout,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            hits = 0
            lookups = 0
            total_request_s = 0.0
            for req_i in range(n_requests):
                indices = index_stream[warmup_requests + req_i]
                req_start = time.perf_counter()
                req_hits, req_lookups = _one_request(
                    mode, model, tokenizer, indices, docs, doc_keys,
                    cache, query_text, n_new_tokens, layout,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                req_ms = (time.perf_counter() - req_start) * 1000.0
                total_request_s += req_ms / 1000.0
                append(
                    f"{run_id}:req{req_i}",
                    "per_request_latency_ms",
                    req_ms,
                    prov,
                    store=store_path,
                )
                hits += req_hits
                lookups += req_lookups

            qps = n_requests / total_request_s if total_request_s > 0 else float("inf")
            append(run_id, "qps", qps, prov, store=store_path)
            append(run_id, "n_requests", n_requests, prov, store=store_path)
            append(run_id, "total_elapsed_s", total_request_s, prov, store=store_path)
            hit_rate = hits / lookups if (mode == "ddi_cache" and lookups > 0) else 0.0
            append(run_id, "hit_rate", hit_rate, prov, store=store_path)
        except Exception as exc:
            _log_cell_failure(cell_desc, exc)


def _cache_restart_quality(
    cfg: dict[str, Any], cfg_hash: str, seed: int, store_path: Path
) -> None:
    """Preserves phase4_restart: run_id prefix and per-record loss/em/f1/presence rows."""
    from ddi.positions import build_positions

    model_name = _require(cfg, "model", "cache/restart_quality")
    datasets = _require(cfg, "datasets", "cache/restart_quality")
    masks = _require(cfg, "masks", "cache/restart_quality")
    layouts_cfg = cfg.get("layouts", ["sequential", "restart"])

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("cache/restart_quality: 'datasets' must be a non-empty list")
    if not isinstance(masks, list) or not masks:
        raise ValueError("cache/restart_quality: 'masks' must be a non-empty list")
    if not isinstance(layouts_cfg, list) or not layouts_cfg:
        raise ValueError("cache/restart_quality: 'layouts' must be a non-empty list")
    for layout in layouts_cfg:
        if layout not in ("sequential", "restart"):
            raise ValueError(f"cache/restart_quality: unknown layout {layout!r}")

    max_records = cfg.get("max_records")
    max_new_tokens = int(cfg.get("max_new_tokens", 64))

    checkpoint = cfg.get("checkpoint")
    rope_theta = cfg.get("rope_theta")
    if rope_theta is not None:
        _log(f"cache/restart_quality: rope_theta override -> {rope_theta}")
    model, tokenizer = inference.load(
        model_name, checkpoint_dir=checkpoint, rope_theta=rope_theta
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    for dataset in datasets:
        prov = capture(
            config_hash=cfg_hash,
            seed=seed,
            model_revision=model_name,
            dataset_revision=dataset,
        )

        for mask_spec in masks:
            mask_variant, bridge_k = _parse_mask_spec(mask_spec)
            tag = _variant_tag(mask_variant, bridge_k)
            for layout in layouts_cfg:
                cell_desc = (
                    f"cache/restart_quality(dataset={dataset}, mask={mask_variant}, "
                    f"layout={layout})"
                )
                try:
                    coerced_full = 0
                    for i, record in enumerate(load_dataset(dataset, "val", seed)):
                        if max_records is not None and i >= max_records:
                            break
                        if not record["possible_answers"]:
                            continue
                        doc_texts = list(record["info_contexts"])

                        variant_here = mask_variant
                        bridge_k_here = bridge_k
                        if not doc_texts and mask_variant != "full":
                            variant_here = "full"
                            bridge_k_here = None
                            coerced_full += 1

                        answer_text = record["dialog"][-1]["content"]
                        # Same single source of truth for prompt text as `_quality_cell`.
                        # This was the one caller in the repo still templating the dialog
                        # inline, which silently dropped both of the things `ddi.prompt`
                        # exists to add: `DOC_INSTRUCTION` (without it `em` is 0.0000 in
                        # every cell) and `enable_thinking=False` (without it a Qwen3
                        # template burns all 64 max_new_tokens inside `<think>`). Both
                        # failures look like a restart-layout effect from the rows alone.
                        query_text = build_query_text(tokenizer, record)

                        full_ids, doc_lengths, query_len, answer_len = encode_segments(
                            tokenizer, doc_texts, query_text, answer_text
                        )
                        full_ids = full_ids.to(model.device)
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
                            **_mask_kwargs(variant_here, bridge_k_here),
                        )
                        additive = (
                            to_additive(allow, dtype=model.dtype)
                            .to(model.device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        positions = (
                            build_positions(doc_lengths, query_len, answer_len, layout)  # type: ignore[arg-type]
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
                        loss_val = float(shifted_cross_entropy(out.logits, labels).item())

                        # Generation reuses inference.encode + inference.generate with the
                        # right layout so the KV-cached decode path sees the same positions.
                        batch = inference.encode(
                            tokenizer, docs=doc_texts, query=query_text, layout=layout
                        )
                        preds = inference.generate(
                            model, tokenizer, batch,
                            mask_variant=variant_here,
                            max_new_tokens=max_new_tokens,
                            mask_kwargs=_mask_kwargs(variant_here, bridge_k_here),
                        )
                        pred = preds[0]

                        rid = (
                            f"phase4_restart:{model_name}:{dataset}:{tag}:{layout}:"
                            f"seed={seed}:{prov.git_sha[:8]}:{record['id']}"
                        )
                        answers = record["possible_answers"]
                        append(rid, "loss", loss_val, prov, store=store_path)
                        append(rid, "em", exact_match(pred, answers), prov, store=store_path)
                        append(rid, "f1", token_f1(pred, answers), prov, store=store_path)
                        append(
                            rid,
                            "presence",
                            answer_presence(pred, answers),
                            prov,
                            store=store_path,
                        )
                    if coerced_full > 0:
                        print(
                            f"[ddi.run:cache/restart_quality] coerced variant="
                            f"{mask_variant!r} -> 'full' on {coerced_full} record(s)",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    _log_cell_failure(cell_desc, exc)


def run_cache(cfg: dict[str, Any], cfg_hash: str, seed: int) -> None:
    """Dispatch a `kind: cache` YAML on `task: correctness | throughput | restart_quality`."""
    task = _require(cfg, "task", "cache")
    store_path = _store_path_for(cfg)

    t0 = time.perf_counter()
    _log(f"cache/{task}: starting")
    if task == "correctness":
        _cache_correctness(cfg, cfg_hash, seed, store_path)
    elif task == "throughput":
        _cache_throughput(cfg, cfg_hash, seed, store_path)
    elif task == "restart_quality":
        _cache_restart_quality(cfg, cfg_hash, seed, store_path)
    else:
        raise ValueError(
            f"cache: unknown task {task!r}; expected correctness | throughput | restart_quality"
        )
    _log(f"cache/{task}: done in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# kind: finetune — self-distillation trainer for ddi_qf (issue #120).
# ---------------------------------------------------------------------------


def run_finetune(cfg: dict[str, Any], cfg_hash: str, seed: int) -> None:
    """Dispatch a `kind: finetune` YAML: teacher=full, student=ddi_qf, KL loss.

    Wraps `ddi.finetune.finetune` in a `_CellTracker` so the progress log shape
    matches every other kind.  The finetuner emits its own per-step / per-epoch
    rows; this dispatcher owns only the "cell 1/1" boundary + provenance stamp.
    """
    from ddi.finetune import finetune as _do_finetune

    model_name = _require(cfg, "model", "finetune")
    datasets = _require(cfg, "datasets", "finetune")
    output_dir = Path(_require(cfg, "output_dir", "finetune"))

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("finetune: 'datasets' must be a non-empty list")

    max_records = cfg.get("max_records")
    held_out_frac = float(cfg.get("held_out_frac", 0.05))
    epochs = int(cfg.get("epochs", 1))
    lr = float(cfg.get("lr", 1e-5))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    kl_temperature = float(cfg.get("kl_temperature", 1.0))
    teacher_first_token_weight = float(cfg.get("teacher_first_token_weight", 0.0))
    full_anchor_weight = float(cfg.get("full_anchor_weight", 0.0))
    blind_anchor_weight = float(cfg.get("blind_anchor_weight", 0.0))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    warmup_steps = int(cfg.get("warmup_steps", 100))
    log_every = int(cfg.get("log_every", 50))
    max_steps = cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    # Single-host data parallelism knob (issue #123). Default 1 preserves the
    # original single-device behavior byte-for-byte; >1 requires the caller to
    # launch under `torchrun --nproc_per_node=N` (the Makefile `finetune` target
    # does this when `NUM_GPUS>1`).
    num_gpus = int(cfg.get("num_gpus", 1))
    if num_gpus < 1:
        raise ValueError(f"finetune: 'num_gpus' must be >= 1, got {num_gpus}")
    # Records with <2 documents produce an identical student/teacher mask and hence a
    # zero KL — see `ddi.finetune._prefilter_records`. Default 2 skips them.
    min_docs = int(cfg.get("min_docs", 2))
    if min_docs < 1:
        raise ValueError(f"finetune: 'min_docs' must be >= 1, got {min_docs}")
    # Freeze all but the top N decoder layers. `null` (the default) trains the whole
    # student, which is what every cell up to the 7B does. It exists for the 27B, whose
    # full-finetune footprint is ~8 bytes/param per rank — teacher + student + grads +
    # AdamW8bit, all replicated by DDP — i.e. ~222 GB against a 141 GB card. Freezing the
    # trunk drops the grad and optimizer terms and brings it to ~112 GB.
    trainable_layers = cfg.get("trainable_layers")
    if trainable_layers is not None:
        trainable_layers = int(trainable_layers)
        if trainable_layers < 1:
            raise ValueError(
                f"finetune: 'trainable_layers' must be >= 1, got {trainable_layers}"
            )
    # Cap on how many holdout records the epoch perplexities score. They run on rank 0
    # only while every other rank waits at a barrier, so on an 8-GPU cell this is
    # single-GPU work blocking the whole allocation — and it is repaid in full after any
    # restart. `null` scores the entire holdout, which is the pre-cap behaviour.
    holdout_eval_records = cfg.get("holdout_eval_records")
    if holdout_eval_records is not None:
        holdout_eval_records = int(holdout_eval_records)
        if holdout_eval_records < 1:
            raise ValueError(
                f"finetune: 'holdout_eval_records' must be >= 1, got {holdout_eval_records}"
            )
    # Gradient clipping and the divergence budget. Defaults are deliberate, not tuning:
    # without them one gradient spike took Qwen2.5-7B to NaN and the run still reported
    # success after 8 GPU-hours (2026-08-07).
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    max_skipped_steps = int(cfg.get("max_skipped_steps", 20))
    layout = str(cfg.get("layout", "sequential"))
    if layout not in ("sequential", "restart", "mixed"):
        raise ValueError(
            f"finetune: 'layout' must be sequential/restart/mixed, got {layout!r}"
        )
    store_path = _store_path_for(cfg)

    prov = capture(
        config_hash=cfg_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision="+".join(datasets),
    )

    tracker = _CellTracker("finetune", 1)
    cell_desc = f"model={model_name} datasets={datasets} out={output_dir}"
    t = tracker.start(cell_desc)
    try:
        _do_finetune(
            model_name,
            datasets,
            output_dir,
            seed=seed,
            max_records=max_records,
            held_out_frac=held_out_frac,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            kl_temperature=kl_temperature,
            teacher_first_token_weight=teacher_first_token_weight,
            full_anchor_weight=full_anchor_weight,
            blind_anchor_weight=blind_anchor_weight,
            grad_accum_steps=grad_accum_steps,
            warmup_steps=warmup_steps,
            log_every=log_every,
            max_steps=max_steps,
            num_gpus=num_gpus,
            min_docs=min_docs,
            trainable_layers=trainable_layers,
            holdout_eval_records=holdout_eval_records,
            max_grad_norm=max_grad_norm,
            max_skipped_steps=max_skipped_steps,
            layout=layout,
            provenance=prov,
            store_path=store_path,
        )
        tracker.ok(t)
    except Exception as exc:  # per-cell isolation matches every other dispatcher
        _log_cell_failure(f"finetune({cell_desc})", exc)
        tracker.fail(t)
    tracker.summary()


# ---------------------------------------------------------------------------
# Dispatch table + CLI entry.
# ---------------------------------------------------------------------------


_DISPATCH = {
    "quality": run_quality,
    "timing": run_timing,
    "cache": run_cache,
    "finetune": run_finetune,
}


# ---------------------------------------------------------------------------
# What `--set` is allowed to override, per kind.
#
# Every dispatcher reads its optional keys through `cfg.get(...)`, so an unknown key
# merged into the config is not an error — it is folded into cfg_hash (which makes the
# store row look distinct) and then ignored. That failure mode is not theoretical and
# it is not cheap: `nirvana/common/constant.py` records a 27B graph pinned to a revision
# that predated `trainable_layers`. The `--set trainable_layers=` it passed was accepted,
# ignored, and the job attempted a full finetune and OOMed after taking an 8-GPU slot.
# Nothing anywhere reported an error.
#
# The sets below are what the dispatchers actually read, enumerated from the `_require`
# / `cfg.get` calls in each `run_*` and `_*_cell`. Adding a config key means adding it
# here, or `--set` will refuse it — which is the intended coupling: a knob the job-side
# code cannot read must not look like it was applied.
# ---------------------------------------------------------------------------

# Read by `run()` itself or by `_store_path_for`, so they are valid for every kind.
# `report` and `metrics` are read from the same YAML by `ddi.aggregate` and the report
# renderers rather than by any dispatcher here; they are accepted so retargeting a
# report block from the command line is not rejected as a typo.
_SET_KEYS_COMMON = frozenset(
    {"model", "seed", "output_store", "require_gpu", "report", "metrics"}
)

_SET_KEYS: dict[str, frozenset[str]] = {
    "quality": _SET_KEYS_COMMON
    | {
        "datasets",
        "masks",
        "max_records",
        "max_new_tokens",
        "checkpoint",
        "evaluation_model",
        "heldout_manifest_mode",
        "rope_theta",
        "heldout_manifest",
        "predictions_store",
    },
    "timing": _SET_KEYS_COMMON
    | {
        "Ns",
        "Ls",
        "backends",
        "masks",
        "Q",
        "n_repeats",
        "warmup",
        "n_new_tokens",
        "checkpoint",
    },
    # Union over the three `task:` values, because `task` is itself overridable and the
    # sub-dispatch happens inside `run_cache`, after these are merged.
    "cache": _SET_KEYS_COMMON
    | {
        "task",
        "checkpoint",
        # correctness
        "modes",
        "Ns",
        "Ls",
        "n_queries_per_doc",
        "Q",
        "answer_len",
        "atol_check",
        "cache_dir",
        # throughput
        "layout",
        "n_requests",
        "warmup_requests",
        "reuse_distribution",
        "zipf_a",
        "top_docs",
        "docs_per_request",
        "doc_token_len",
        "query_token_len",
        "n_new_tokens",
        # restart_quality
        "datasets",
        "masks",
        "layouts",
        "max_records",
        "max_new_tokens",
        "rope_theta",
    },
    "finetune": _SET_KEYS_COMMON
    | {
        "datasets",
        "output_dir",
        "max_records",
        "held_out_frac",
        "epochs",
        "lr",
        "weight_decay",
        "kl_temperature",
        "teacher_first_token_weight",
        "full_anchor_weight",
        "blind_anchor_weight",
        "grad_accum_steps",
        "warmup_steps",
        "log_every",
        "max_steps",
        "num_gpus",
        "min_docs",
        "trainable_layers",
        "holdout_eval_records",
        "max_grad_norm",
        "max_skipped_steps",
        "layout",
    },
}


def _validate_set_keys(kind: str, keys: list[str], yaml_path: Path) -> None:
    """Reject a `--set` key this kind's dispatcher will never read.

    Use `--set-unchecked` for the case this cannot know about: a key that job-side code
    at a *different* revision reads. That is an explicit request to merge blind, which is
    the whole point — the default must not be blind.
    """
    allowed = _SET_KEYS[kind]
    for key in keys:
        if key in allowed:
            continue
        if key == "kind":
            # Special-cased because it is not merely unread, it is unreadable: `run()`
            # binds `kind` from the YAML above, before overrides are applied, so the
            # dispatch is already chosen. `--set kind=` has never re-dispatched anything;
            # it only perturbed cfg_hash.
            raise ValueError(
                "--set kind=...: 'kind' cannot be overridden — it selects the dispatcher "
                "and is read before --set is applied, so the override has never taken "
                "effect; it only perturbed cfg_hash. Point the CLI at a different YAML."
            )
        raise ValueError(
            f"--set {key}=...: kind {kind!r} ({yaml_path}) has no config key {key!r}, "
            "so the override would be merged into the config, folded into the config hash "
            "and then never read — the run would look distinct and behave identically. "
            f"Valid keys for kind={kind!r}: {', '.join(sorted(allowed))}. "
            f"If job-side code at another revision does read {key!r}, pass it as "
            f"--set-unchecked {key}=... ."
        )


def run(
    yaml_path: Path,
    seed_override: int | None = None,
    model_override: str | None = None,
    backends_override: list[str] | None = None,
    num_gpus_override: int | None = None,
    set_overrides: list[tuple[str, Any]] | None = None,
    unchecked_overrides: list[tuple[str, Any]] | None = None,
    enforce_require_gpu: bool = False,
) -> None:
    """Load a YAML file and dispatch on `kind:`.

    `enforce_require_gpu` is set by `main()` and only by `main()` — see `_configure_gpu`.
    """
    cfg, cfg_hash = load_yaml(yaml_path)
    kind = cfg.get("kind")
    if not kind:
        raise ValueError(f"YAML at {yaml_path}: missing required key: 'kind'")
    if kind not in _DISPATCH:
        raise ValueError(
            f"YAML at {yaml_path}: unknown kind {kind!r}; expected one of {sorted(_DISPATCH)}"
        )

    if model_override is not None:
        cfg["model"] = model_override
    elif "model" not in cfg:
        raise ValueError(
            f"YAML at {yaml_path}: 'model' must be set in the YAML or overridden with --model"
        )

    if backends_override is not None:
        # Only `kind: timing` actually consumes `backends:`; setting it on other kinds is a no-op
        # (the dispatcher ignores the field), which is fine for a uniform CLI shape.
        cfg["backends"] = backends_override

    if num_gpus_override is not None:
        # Only `kind: finetune` consumes `num_gpus:`; ignored elsewhere for a uniform CLI shape.
        cfg["num_gpus"] = num_gpus_override

    if set_overrides:
        _validate_set_keys(kind, [key for key, _ in set_overrides], yaml_path)

    # `--set` and `--set-unchecked` differ only in whether the key was checked above;
    # both merge and both must reach cfg_hash, so they share one list from here on. With
    # no `--set-unchecked` the digest is byte-identical to the pre-check behaviour, which
    # keeps every already-published run joinable to its rows.
    all_overrides = list(set_overrides or []) + list(unchecked_overrides or [])
    if all_overrides:
        # Generic per-key override, so a grid cell can vary `lr`, `kl_temperature`,
        # `epochs`, `max_records` or `output_dir` without a YAML per cell. Applied last,
        # so it wins over the typed flags above.
        for key, value in all_overrides:
            cfg[key] = value
        # cfg_hash is the join key between results/runs.jsonl and the exact config that
        # produced them. Two cells differing only by --set would otherwise share a hash
        # and be indistinguishable in the store, so fold the overrides into it.
        digest = json.dumps(sorted((k, v) for k, v in all_overrides), sort_keys=True, default=str)
        cfg_hash = hashlib.sha256((cfg_hash + digest).encode()).hexdigest()[:16]
        _log(f"applied {len(all_overrides)} --set override(s); cfg_hash -> {cfg_hash[:8]}")
        if unchecked_overrides:
            _log(
                "  "
                + str(len(unchecked_overrides))
                + " of them via --set-unchecked (key not validated against this kind): "
                + ", ".join(sorted(key for key, _ in unchecked_overrides))
            )

    if seed_override is not None:
        seed = seed_override
    elif "seed" in cfg:
        seed = int(cfg["seed"])
    else:
        raise ValueError(
            f"YAML at {yaml_path}: 'seed' must be set in the YAML or overridden with --seed"
        )

    _configure_gpu(cfg, enforce_require_gpu=enforce_require_gpu)
    _seed_all(seed)
    _log(f"loading YAML {yaml_path} (kind={kind}, seed={seed}, cfg_hash={cfg_hash[:8]})")
    _DISPATCH[kind](cfg, cfg_hash, seed)


def _configure_gpu(cfg: dict[str, Any], *, enforce_require_gpu: bool) -> None:
    """Print the device summary, enable TF32, and (from the CLI) refuse to run blind on CPU.

    `ddi.gpu` was written to be called once at startup and, until this call existed, never
    was. Its only caller was `ddi.cli.main`, which had been unimportable since the
    `ddi/runners/` package it imports was deleted, while `pyproject.toml` pointed
    `ddi-run` at `ddi.run:main` instead. Two things were silently lost: TF32 /
    `set_float32_matmul_precision("high")` was never enabled, so every published
    `kind: timing` number was measured under different kernel selection than `ddi/gpu.py`
    documents; and `require_gpu` was never honoured, so a sweep could run to completion on
    CPU and write rows indistinguishable from GPU rows.

    `configure()` is unconditional — it is what makes the timing numbers mean what the
    module says they mean, and no caller should have to remember it. The CPU gate is not,
    because `run()` is also called directly (in-process, against stub models) by callers
    that legitimately have no GPU; `main()` turns it on, so every `ddi-run` / `python -m
    ddi.run` invocation is gated. Both happen after `--set` is merged, so
    `--set require_gpu=false` works, and before seeding and any model load.
    """
    from ddi import gpu  # local import so the module stays importable without a GPU stack

    gpu.configure()
    # Default: every experiment run REQUIRES CUDA — silently running a real DDI sweep on
    # CPU would burn hours and produce meaningless numbers. Opt out with
    # `require_gpu: false` in the config for smoke tests and dry-runs.
    if enforce_require_gpu and cfg.get("require_gpu", True):
        gpu.require_cuda()


def _parse_set(value: str) -> tuple[str, Any]:
    """Parse `key=value`; the value is YAML-scalar-parsed so types survive.

    `lr=2e-5` becomes a float, `epochs=2` an int, `max_records=null` None, and anything
    unquoted that is not a YAML scalar stays a string. That matters because the config
    readers call `float()` / `int()` on these and a stray string would fail late, inside
    the trainer, rather than here.
    """
    key, sep, raw = value.partition("=")
    key = key.strip()
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--set expected 'key=value', got {value!r}")
    parsed = yaml.safe_load(raw)
    if isinstance(parsed, str):
        # YAML 1.1 does not accept `2e-5` as a float (it wants `2.0e-5`), so the most
        # natural way to write a learning rate comes back as a string. Recover it here
        # rather than letting it reach the trainer, where only keys that happen to be
        # wrapped in float() would survive.
        try:
            return key, int(parsed)
        except ValueError:
            pass
        try:
            return key, float(parsed)
        except ValueError:
            pass
    return key, parsed


def _parse_backends(value: str) -> list[str]:
    """Comma-separated backend list, e.g. `sdpa` or `sdpa,fa2`."""
    items = [b.strip() for b in value.split(",") if b.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            f"--backends expected a non-empty comma-separated list, got {value!r}"
        )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi.run")
    parser.add_argument("yaml_path", type=Path, help="path to experiment YAML")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override seed from YAML (falls back to YAML `seed:` if unset)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="model to run (required unless the YAML sets `model:`)",
    )
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=None,
        help=(
            "override `backends:` from YAML (comma-separated: sdpa,fa2). "
            "Only used by kind: timing; ignored for other kinds."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "override `num_gpus:` from YAML (issue #123). Only used by kind: finetune; "
            "ignored for other kinds. >1 requires launching under `torchrun --nproc_per_node=N`."
        ),
    )
    parser.add_argument(
        "--set",
        type=_parse_set,
        action="append",
        dest="set_overrides",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "override a top-level YAML key this kind reads, repeatable: "
            "--set lr=2e-5 --set output_dir=results/checkpoints/foo. "
            "Applied after --model/--backends/--num-gpus and folded into the config hash. "
            "A key the kind's dispatcher does not read is rejected (see --set-unchecked)."
        ),
    )
    parser.add_argument(
        "--set-unchecked",
        type=_parse_set,
        action="append",
        dest="unchecked_overrides",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "like --set but skips the per-kind key check. For the one case the check "
            "cannot see: job-side code at a different revision that reads a key this "
            "revision does not. The override is merged and hashed exactly like --set, and "
            "the keys are echoed to the log so the bypass is visible in the run's output."
        ),
    )
    args = parser.parse_args(argv)
    run(
        args.yaml_path,
        seed_override=args.seed,
        model_override=args.model,
        backends_override=args.backends,
        num_gpus_override=args.num_gpus,
        set_overrides=args.set_overrides,
        unchecked_overrides=args.unchecked_overrides,
        # This is the CLI, so honour `require_gpu`. A real sweep that silently ran on CPU
        # would burn hours and produce numbers that look like every other row.
        enforce_require_gpu=True,
    )
    if _CellTracker.failed_total:
        _log(f"FAILED: {_CellTracker.failed_total} cell(s) failed — exiting non-zero")
        return 1
    return 0


def _exit(code: int) -> None:
    """Leave the process without running interpreter shutdown.

    Every rank of an 8-GPU finetune segfaults in NCCL/UCX teardown *after* the work is
    finished — the logs show `finetune: done` and `1 passed, 0 failed` on all 8 ranks,
    then `Caught signal 11 (address not mapped ... 0x...1a0)` on all 8, at identical
    offsets. It is a library atexit/destructor ordering bug at interpreter shutdown, not
    ours: `ddi.finetune` already ends with `dist.barrier()` + `dist.destroy_process_group()`,
    and the crash happens after that.

    It would be harmless if it stayed cosmetic, but torchrun reads the child exit codes:
    one rank exiting -11 becomes `ChildFailedError`, `make` returns 1, and the caller sees
    a failed run. On 2026-08-06 that discarded a complete 4h18m 8-GPU training run whose
    checkpoint was already written — the graph's guard exited before uploading it.

    So once the work is done and the streams are flushed, skip shutdown entirely. There is
    nothing left to clean up: `ddi` registers no `atexit` hooks and no `__del__`, results are
    written and closed as they are produced, and the OS reclaims memory and file handles.
    Set `DDI_HARD_EXIT=0` to restore normal shutdown when debugging the teardown itself.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("DDI_HARD_EXIT", "1") == "1":
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    _exit(main())
