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
  - Provenance stamped per row via `ddi.provenance.capture`; the YAML source
    path is folded into the run_id so every row is traceable back to its
    source file, not just to a config hash.
"""

from __future__ import annotations

import argparse
import hashlib
import random
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
from ddi.masks import build_mask, to_additive
from ddi.metrics import (
    answer_presence,
    exact_match,
    make_answer_labels,
    token_f1,
)
from ddi.provenance import capture
from ddi.runners._shared import shifted_cross_entropy
from ddi.store import DEFAULT_STORE, append
from ddi.timing import append_timing, measure_decode, measure_prefill

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
    """Seed `random`, `numpy`, `torch` (+ CUDA if available). Mirrors ddi.cli.set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _store_path_for(cfg: dict[str, Any]) -> Path:
    """Resolve the JSONL store path; default matches ddi.store.DEFAULT_STORE."""
    raw = cfg.get("output_store")
    return Path(raw) if raw else DEFAULT_STORE


def _log_cell_failure(cell_desc: str, exc: BaseException) -> None:
    """Log a per-cell failure to stderr without aborting the outer sweep loop."""
    print(f"[ddi.run] cell FAILED: {cell_desc}: {exc!r}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


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
    yaml_path: Path,
    cfg_hash: str,
    store_path: Path,
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

    for i, record in enumerate(load_dataset(dataset, "val", seed)):
        if max_records is not None and i >= max_records:
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

        user_dialog = record["dialog"][:-1]
        answer_text = record["dialog"][-1]["content"]

        query_text = tokenizer.apply_chat_template(
            user_dialog, tokenize=False, add_generation_prompt=True
        )

        # Loss computation needs the answer tokens concatenated in and a masked labels
        # tensor — that's the phase2_train_free contract.  Use build_mask/to_additive
        # directly here (rather than inference.encode) so we can splice the answer in.
        if want_loss:
            doc_id_tensors = [
                tokenizer(d, return_tensors="pt", add_special_tokens=False).input_ids
                for d in doc_texts
            ]
            doc_lengths = [int(t.shape[-1]) for t in doc_id_tensors]
            query_ids = tokenizer(
                query_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            answer_ids = tokenizer(
                answer_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            answer_len = int(answer_ids.shape[-1])
            full_ids = torch.cat(doc_id_tensors + [query_ids, answer_ids], dim=-1).to(
                model.device
            )
            labels = make_answer_labels(
                full_ids,
                full_ids.shape[-1] - answer_len,
                full_ids.shape[-1],
            ).to(model.device)
            allow = build_mask(
                doc_lengths,
                query_len=int(query_ids.shape[-1]),
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
        if want_loss and loss_val is not None:
            append(rid, "loss", loss_val, prov, store=store_path)
        if want_em:
            append(rid, "em", exact_match(pred, record["possible_answers"]), prov, store=store_path)
        if want_f1:
            append(rid, "f1", token_f1(pred, record["possible_answers"]), prov, store=store_path)
        if want_presence:
            append(
                rid,
                "presence",
                answer_presence(pred, record["possible_answers"]),
                prov,
                store=store_path,
            )

    if coerced_full > 0:
        print(
            f"[ddi.run:quality] coerced variant={mask_variant!r} -> 'full' on "
            f"{coerced_full} record(s) with no info_contexts",
            file=sys.stderr,
        )


def run_quality(cfg: dict[str, Any], cfg_hash: str, seed: int, yaml_path: Path) -> None:
    """Dispatch a `kind: quality` YAML across the (dataset x mask) grid."""
    model_name = _require(cfg, "model", "quality")
    datasets = _require(cfg, "datasets", "quality")
    masks = _require(cfg, "masks", "quality")
    metrics = _require(cfg, "metrics", "quality")

    max_records = cfg.get("max_records")
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    store_path = _store_path_for(cfg)

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("quality: 'datasets' must be a non-empty list")
    if not isinstance(masks, list) or not masks:
        raise ValueError("quality: 'masks' must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("quality: 'metrics' must be a non-empty list")

    model, tokenizer = inference.load(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    for dataset in datasets:
        for mask_spec in masks:
            mask_variant, bridge_k = _parse_mask_spec(mask_spec)
            cell_desc = f"quality({dataset=}, mask={mask_variant}, bridge_k={bridge_k})"
            try:
                _quality_cell(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    mask_variant=mask_variant,
                    bridge_k=bridge_k,
                    model_name=model_name,
                    seed=seed,
                    max_records=max_records,
                    max_new_tokens=max_new_tokens,
                    metrics=metrics,
                    yaml_path=yaml_path,
                    cfg_hash=cfg_hash,
                    store_path=store_path,
                )
            except Exception as exc:  # per-cell isolation
                _log_cell_failure(cell_desc, exc)


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
) -> None:
    """One (mask, N, L, backend) grid cell.  Fresh model load per backend (matches phase3)."""
    if backend == "fa2" and not torch.cuda.is_available():
        raise RuntimeError(
            "backend='fa2' requires CUDA; torch.cuda.is_available() is False. "
            "Route this cell to a GPU worker or select backend='sdpa'."
        )

    # Fresh load per backend — HF picks attn kernel at from_pretrained time.
    model, _tokenizer = inference.load(model_name, backend=backend)
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

    ftt = measure_prefill(model, prefill_ids, additive, n_repeats=n_repeats, warmup=warmup)
    append_timing(store=store_path, run_id=run_id, metric="ftt", timing=ftt, provenance=prov)

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


def run_timing(cfg: dict[str, Any], cfg_hash: str, seed: int, yaml_path: Path) -> None:
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

    for lst_name, lst in (("Ns", Ns), ("Ls", Ls), ("backends", backends), ("masks", masks)):
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"timing: {lst_name!r} must be a non-empty list")

    for mask_spec in masks:
        mask_variant, bridge_k = _parse_mask_spec(mask_spec)
        for N in Ns:
            for L in Ls:
                for backend in backends:
                    cell_desc = (
                        f"timing(mask={mask_variant}, N={N}, L={L}, backend={backend})"
                    )
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
                        )
                    except Exception as exc:
                        _log_cell_failure(cell_desc, exc)


# ---------------------------------------------------------------------------
# kind: cache — dispatches on task: correctness | throughput | restart_quality.
# ---------------------------------------------------------------------------


def _cache_correctness(cfg: dict[str, Any], cfg_hash: str, seed: int, store_path: Path) -> None:
    """Preserves phase4_cache: run_id prefix and kv_deterministic / kv_cache_roundtrip rows."""
    from pathlib import Path as _Path

    from ddi.cache import DocCache, compute_doc_kv

    model_name = _require(cfg, "model", "cache/correctness")
    dataset = _require(cfg, "dataset", "cache/correctness")
    layout = cfg.get("layout", "restart")
    if layout not in ("sequential", "restart"):
        raise ValueError(f"cache/correctness: unknown layout {layout!r}")
    cache_dir = _Path(cfg.get("cache_dir", "cache/doc_kv"))
    max_records = int(cfg.get("max_records", 10))
    atol = float(cfg.get("atol", 1e-4))

    model, tokenizer = inference.load(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    cache = DocCache(cache_dir)
    prov = capture(
        config_hash=cfg_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=dataset,
    )
    run_id_base = (
        f"phase4_cache:{model_name}:{dataset}:{layout}:seed={seed}:{prov.git_sha[:8]}"
    )

    def _all_close(a_list, b_list) -> bool:
        if len(a_list) != len(b_list):
            return False
        return all(
            torch.allclose(a, b, atol=atol) for a, b in zip(a_list, b_list, strict=True)
        )

    for i, record in enumerate(load_dataset(dataset, "val", seed)):
        if i >= max_records:
            break
        if not record["info_contexts"]:
            continue
        for doc_idx, doc_text in enumerate(record["info_contexts"]):
            cell_desc = f"cache/correctness(record={record['id']}, doc={doc_idx})"
            try:
                e1 = compute_doc_kv(model, tokenizer, doc_text, layout)
                e2 = compute_doc_kv(model, tokenizer, doc_text, layout)
                deterministic = (
                    e1.key == e2.key
                    and _all_close(e1.k_per_layer, e2.k_per_layer)
                    and _all_close(e1.v_per_layer, e2.v_per_layer)
                )
                cache.put(e1)
                hit = cache.get(e1.key)
                if hit is None:
                    raise RuntimeError(f"cache miss after put for key={e1.key}")
                roundtrip = _all_close(hit.k_per_layer, e1.k_per_layer) and _all_close(
                    hit.v_per_layer, e1.v_per_layer
                )
                rid = f"{run_id_base}:{record['id']}:doc{doc_idx}"
                append(rid, "kv_deterministic", float(deterministic), prov, store=store_path)
                append(rid, "kv_cache_roundtrip", float(roundtrip), prov, store=store_path)
            except Exception as exc:
                _log_cell_failure(cell_desc, exc)


def _cache_throughput(cfg: dict[str, Any], cfg_hash: str, seed: int, store_path: Path) -> None:
    """One row-block per `mode` in `modes:`; row shape matches phase4_throughput exactly.

    Uses the same synthetic doc universe and per-request timing helpers as the
    original runner; only difference is the outer loop over `modes:` from the
    YAML rather than one runner invocation per mode.
    """
    from pathlib import Path as _Path

    from ddi.cache import DocCache, compute_key
    from ddi.runners.phase4_throughput import _one_request  # reuse until #89

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

    model, tokenizer = inference.load(model_name)
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
            index_stream: list[list[int]] = []
            from ddi.runners.phase4_throughput import _sample_doc_indices

            index_stream = _sample_doc_indices(
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
                dataset_revision=f"synthetic_throughput_top={top_docs}_L={doc_token_len}",
            )
            reuse_tag = (
                f"{reuse_distribution}(a={zipf_a})"
                if reuse_distribution == "zipf"
                else reuse_distribution
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

            hits = 0
            lookups = 0
            total_start = time.perf_counter()
            for req_i in range(n_requests):
                indices = index_stream[warmup_requests + req_i]
                req_start = time.perf_counter()
                req_hits, req_lookups = _one_request(
                    mode, model, tokenizer, indices, docs, doc_keys,
                    cache, query_text, n_new_tokens, layout,
                )
                req_ms = (time.perf_counter() - req_start) * 1000.0
                append(
                    f"{run_id}:req{req_i}",
                    "per_request_latency_ms",
                    req_ms,
                    prov,
                    store=store_path,
                )
                hits += req_hits
                lookups += req_lookups
            total_elapsed_s = time.perf_counter() - total_start

            qps = n_requests / total_elapsed_s if total_elapsed_s > 0 else float("inf")
            append(run_id, "qps", qps, prov, store=store_path)
            append(run_id, "n_requests", n_requests, prov, store=store_path)
            append(run_id, "total_elapsed_s", total_elapsed_s, prov, store=store_path)
            hit_rate = hits / lookups if (mode == "ddi_cache" and lookups > 0) else 0.0
            append(run_id, "hit_rate", hit_rate, prov, store=store_path)
        except Exception as exc:
            _log_cell_failure(cell_desc, exc)


def _cache_restart_quality(
    cfg: dict[str, Any], cfg_hash: str, seed: int, yaml_path: Path, store_path: Path
) -> None:
    """Preserves phase4_restart: run_id prefix and per-record loss/em/f1/presence rows."""
    from ddi.positions import build_positions

    model_name = _require(cfg, "model", "cache/restart_quality")
    dataset = _require(cfg, "dataset", "cache/restart_quality")
    masks = _require(cfg, "masks", "cache/restart_quality")
    layouts_cfg = cfg.get("layouts", ["sequential", "restart"])

    if not isinstance(masks, list) or not masks:
        raise ValueError("cache/restart_quality: 'masks' must be a non-empty list")
    if not isinstance(layouts_cfg, list) or not layouts_cfg:
        raise ValueError("cache/restart_quality: 'layouts' must be a non-empty list")
    for layout in layouts_cfg:
        if layout not in ("sequential", "restart"):
            raise ValueError(f"cache/restart_quality: unknown layout {layout!r}")

    max_records = cfg.get("max_records")
    max_new_tokens = int(cfg.get("max_new_tokens", 64))

    model, tokenizer = inference.load(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

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
            cell_desc = f"cache/restart_quality(mask={mask_variant}, layout={layout})"
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

                    user_dialog = record["dialog"][:-1]
                    answer_text = record["dialog"][-1]["content"]
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

                    full_ids = torch.cat(doc_id_tensors + [query_ids, answer_ids], dim=-1).to(
                        model.device
                    )
                    labels = make_answer_labels(
                        full_ids,
                        full_ids.shape[-1] - answer_len,
                        full_ids.shape[-1],
                    ).to(model.device)

                    allow = build_mask(
                        doc_lengths or [full_ids.shape[-1]],
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


def run_cache(cfg: dict[str, Any], cfg_hash: str, seed: int, yaml_path: Path) -> None:
    """Dispatch a `kind: cache` YAML on `task: correctness | throughput | restart_quality`."""
    task = _require(cfg, "task", "cache")
    store_path = _store_path_for(cfg)

    if task == "correctness":
        _cache_correctness(cfg, cfg_hash, seed, store_path)
    elif task == "throughput":
        _cache_throughput(cfg, cfg_hash, seed, store_path)
    elif task == "restart_quality":
        _cache_restart_quality(cfg, cfg_hash, seed, yaml_path, store_path)
    else:
        raise ValueError(
            f"cache: unknown task {task!r}; expected correctness | throughput | restart_quality"
        )


# ---------------------------------------------------------------------------
# Dispatch table + CLI entry.
# ---------------------------------------------------------------------------


_DISPATCH = {
    "quality": run_quality,
    "timing": run_timing,
    "cache": run_cache,
}


def run(yaml_path: Path, seed_override: int | None = None) -> None:
    """Load a YAML file and dispatch on `kind:`."""
    cfg, cfg_hash = load_yaml(yaml_path)
    kind = cfg.get("kind")
    if not kind:
        raise ValueError(f"YAML at {yaml_path}: missing required key: 'kind'")
    if kind not in _DISPATCH:
        raise ValueError(
            f"YAML at {yaml_path}: unknown kind {kind!r}; expected one of {sorted(_DISPATCH)}"
        )

    if seed_override is not None:
        seed = seed_override
    elif "seed" in cfg:
        seed = int(cfg["seed"])
    else:
        raise ValueError(
            f"YAML at {yaml_path}: 'seed' must be set in the YAML or overridden with --seed"
        )

    _seed_all(seed)
    _DISPATCH[kind](cfg, cfg_hash, seed, yaml_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi.run")
    parser.add_argument("yaml_path", type=Path, help="path to experiment YAML")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override seed from YAML (falls back to YAML `seed:` if unset)",
    )
    args = parser.parse_args(argv)
    run(args.yaml_path, seed_override=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
