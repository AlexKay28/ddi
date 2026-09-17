#!/usr/bin/env python3
"""V112 (VIBEQUEUE-114): genuine prefix-cache redo.

Replaces the withdrawn V73 baseline (whose hit branch stored a flag and always
re-ran full generation). Arms, gates and data rules are frozen in
experiments/v112_prefix_cache_redo_preregistration.md.

Arms: monolithic | prefix-fixed | prefix-reordered | ddi-fixed | ddi-reordered.
The prefix cache stores real document-prefix K/V tensors keyed by the ordered
document tuple; hits load the K/Vs and prefill only the query tail (the
instrumented positive control). The DDI arms use the order-independent
canonical per-document cache with an all-hit pre-state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from ddi.canonical_cache import CanonicalCache  # noqa: E402
from ddi.cache import DocCache  # noqa: E402
from ddi.data import load as load_dataset  # noqa: E402
from ddi.inference import load  # noqa: E402
from ddi.prompt import build_query_text  # noqa: E402
from scripts.run_canonical_matrix import _checkpoint_identity, _contract  # noqa: E402
from scripts.run_v66_boundary import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _build_pool,
    _load_dev_content,
    pack_documents,
)

N_DOCS = 5
DOC_LEN = 500
N_REQUESTS = 20
BUDGET = 16
REPEATS = 3
QUERY_CAP = 96
SHUFFLE_SEED = 47
ARMS = (
    "monolithic",
    "prefix-fixed",
    "prefix-reordered",
    "ddi-fixed",
    "ddi-reordered",
)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()[:16]


class PrefixCache:
    """Exact-prefix KV cache keyed by the ordered document token tuple."""

    def __init__(self) -> None:
        self.store: dict[tuple, tuple] = {}
        self.hit_prefill_tokens = 0
        self.miss_count = 0
        self.hit_count = 0

    def key(self, docs) -> tuple:
        return tuple(int(t) for stream in docs for t in stream[0].tolist())

    def get(self, docs):
        return self.store.get(self.key(docs))

    def put(self, docs, kv) -> None:
        self.store[self.key(docs)] = kv


def _prefill_docs_kv(model, docs):
    """Full-causal prefill of the document prefix alone (sequential positions)."""
    ids = torch.cat([s[0] for s in docs])[None].to(model.device)
    positions = torch.arange(ids.shape[1], device=model.device)[None]
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
    return out.past_key_values, out.logits


def _kv_pairs(past) -> list[tuple]:
    """Normalize HF cache (DynamicCache or legacy tuple) into (k, v) pairs."""
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    return [tuple(layer) for layer in past]


def _clone_kv(past) -> tuple:
    return tuple(
        (k.detach().clone(), v.detach().clone()) for k, v in _kv_pairs(past)
    )


def _paste_kv(model, kv):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for index, (k, v) in enumerate(kv):
        cache.update(k.to(model.device), v.to(model.device), index)
    return cache


def _decode(model, past, next_token, budget, eos_id):
    """Greedy decode from a ready cache; returns ids, first-step ms, rest ms."""
    generated: list[int] = []
    pos = past.get_seq_length()
    first_ms = None
    rest_start = None
    for step_index in range(budget):
        tok = int(next_token.item())
        generated.append(tok)
        if eos_id is not None and tok == eos_id:
            break
        step_t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                attention_mask=torch.ones(
                    (1, pos + 1), dtype=torch.long, device=model.device
                ),
                position_ids=torch.tensor([[pos]], device=model.device),
                past_key_values=past,
                use_cache=True,
            )
        if first_ms is None:
            first_ms = (time.perf_counter() - step_t0) * 1000
            rest_start = time.perf_counter()
        past = out.past_key_values
        pos += 1
        next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    rest_ms = (time.perf_counter() - rest_start) * 1000 if rest_start is not None else 0.0
    return generated, first_ms or 0.0, rest_ms


def _monolithic_request(model, docs, query_ids, budget):
    ids = torch.cat([s[0] for s in docs] + [query_ids[0]])[None].to(model.device)
    positions = torch.arange(ids.shape[1], device=model.device)[None]
    mask = torch.ones_like(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    gen, first_ms, rest_ms = _decode(
        model, out.past_key_values, next_token, budget, model.config.eos_token_id
    )
    return gen, prefill_ms, prefill_ms + first_ms, prefill_ms + first_ms + rest_ms


def _prefix_request(model, cache: PrefixCache, docs, query_ids, budget):
    cached = cache.get(docs)
    q = query_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if cached is None:
        cache.miss_count += 1
        doc_past, _ = _prefill_docs_kv(model, docs)
        past = _paste_kv(model, _clone_kv(doc_past))
        cache.put(docs, _clone_kv(doc_past))  # store the doc-prefix K/V only
    else:
        cache.hit_count += 1
        cache.hit_prefill_tokens += int(q.shape[1])
        past = _paste_kv(model, cached)
    start_pos = past.get_seq_length()
    positions = torch.arange(start_pos, start_pos + q.shape[1], device=model.device)[None]
    mask = torch.ones((1, start_pos + q.shape[1]), dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model(
            input_ids=q,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    next_token = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
    gen, first_ms, rest_ms = _decode(
        model, out.past_key_values, next_token, budget, model.config.eos_token_id
    )
    return gen, prefill_ms, prefill_ms + first_ms, prefill_ms + first_ms + rest_ms


def _ddi_request(model, contract, store, docs, query_ids, budget):
    cache = CanonicalCache(model, contract, store)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = cache.request(
        [s for s in docs], query_ids, max_new_tokens=budget, eos_token_id=None
    )
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    ids = list(result) if not isinstance(result, dict) else list(result["generated_ids"])
    return ids, None, None, total_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--requests", type=int, default=N_REQUESTS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    n_requests = 3 if args.smoke else args.requests

    out_dir = ROOT / "results" / (
        "v112_prefix_smoke" if args.smoke else f"v112_prefix_cache_redo_{_utc()}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    log(f"start smoke={args.smoke} checkpoint={args.checkpoint.name}")

    model, tokenizer = load(str(args.checkpoint), dtype="bf16", backend="eager")
    model = model.to("cuda").eval()
    mode, digest = _checkpoint_identity(args.checkpoint)
    contract = _contract(model, args.checkpoint)
    log(f"checkpoint {mode} {digest[:16]}")

    dev_ids, records, _query_id, _tok = _load_dev_content(args.checkpoint)
    pool = _build_pool(tokenizer, records, dev_ids)
    docs, _cursor = pack_documents(pool, DOC_LEN, N_DOCS)
    queries = []
    for record_id in dev_ids:
        q = tokenizer(
            build_query_text(tokenizer, records[record_id]),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0:1, :QUERY_CAP]
        queries.append(q)
        if len(queries) >= n_requests:
            break
    assert len(queries) == n_requests
    log(f"docs: {len(docs)} x {DOC_LEN}; queries: {len(queries)}")

    gen_rng = torch.Generator().manual_seed(SHUFFLE_SEED)

    def reordered(docs_list):
        perm = torch.randperm(len(docs_list), generator=gen_rng).tolist()
        return [docs_list[i] for i in perm]

    results: dict[str, list] = {arm: [] for arm in ARMS}
    identity: dict[str, list] = {arm: [] for arm in ARMS}
    prefix_stats = {"hits": 0, "misses": 0, "hit_prefill_tokens_only": 0}

    for repeat in range(REPEATS):
        log(f"repeat {repeat + 1}/{REPEATS}")
        prefix_fixed = PrefixCache()
        prefix_reordered = PrefixCache()
        cache_dir_f = out_dir / f"ddi_store_fixed_r{repeat}"
        cache_dir_r = out_dir / f"ddi_store_reordered_r{repeat}"
        store_fixed = _ddi_store(model, contract, docs, cache_dir_f)
        store_reordered = _ddi_store(model, contract, docs, cache_dir_r)

        for index in range(n_requests):
            q = queries[index]
            for arm in ARMS:
                if arm == "monolithic":
                    gen, pre_ms, ttft_ms, total_ms = _monolithic_request(model, docs, q, BUDGET)
                elif arm == "prefix-fixed":
                    gen, pre_ms, ttft_ms, total_ms = _prefix_request(
                        model, prefix_fixed, docs, q, BUDGET
                    )
                elif arm == "prefix-reordered":
                    gen, pre_ms, ttft_ms, total_ms = _prefix_request(
                        model, prefix_reordered, reordered(docs), q, BUDGET
                    )
                elif arm == "ddi-fixed":
                    gen, pre_ms, ttft_ms, total_ms = _ddi_request(
                        model, contract, store_fixed, docs, q, BUDGET
                    )
                else:
                    gen, pre_ms, ttft_ms, total_ms = _ddi_request(
                        model, contract, store_reordered, reordered(docs), q, BUDGET
                    )
                results[arm].append(
                    {
                        "repeat": repeat,
                        "request": index,
                        "prefill_ms": pre_ms,
                        "ttft_ms": ttft_ms,
                        "total_ms": total_ms,
                        "ids_sha256": _ids_sha256(gen),
                    }
                )
                identity[arm].append(list(gen))
        prefix_stats["hits"] += prefix_fixed.hit_count
        prefix_stats["misses"] += prefix_fixed.miss_count
        prefix_stats["hit_prefill_tokens_only"] += prefix_fixed.hit_prefill_tokens
        log(f"  repeat done: prefix-fixed hits/misses = "
            f"{prefix_fixed.hit_count}/{prefix_fixed.miss_count}")

    summary = {
        "schema": "v112-prefix-redo-summary-v1",
        "created_utc": _utc(),
        "smoke": args.smoke,
        "prereg": "experiments/v112_prefix_cache_redo_preregistration.md",
        "config": {
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": digest,
            "n_docs": N_DOCS,
            "doc_len": DOC_LEN,
            "n_requests": n_requests,
            "budget": BUDGET,
            "repeats": REPEATS,
            "shuffle_seed": SHUFFLE_SEED,
        },
        "arms": {},
        "gates": {},
        "prefix_cache": prefix_stats,
    }
    def _median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    for arm in ARMS:
        pre = [r["prefill_ms"] for r in results[arm]]
        summary["arms"][arm] = {
            "median_prefill_ms": _median(pre),
            "median_ttft_ms": _median([r["ttft_ms"] for r in results[arm]]),
            "median_total_ms": _median([r["total_ms"] for r in results[arm]]),
            "n_rows": len(results[arm]),
        }

    mono_pre = summary["arms"]["monolithic"]["median_prefill_ms"]
    mono_total = summary["arms"]["monolithic"]["median_total_ms"]
    summary["gates"]["G1_first_request_identical"] = (
        _ids_sha256(identity["prefix-fixed"][0]) == _ids_sha256(identity["monolithic"][0])
    )
    summary["gates"]["G2_hit_forwards_query_only"] = (
        prefix_stats["hit_prefill_tokens_only"] > 0
        and prefix_stats["hits"] > 0
    )
    summary["gates"]["G4_prefix_fixed_under_half_mono"] = (
        summary["arms"]["prefix-fixed"]["median_prefill_ms"] < 0.5 * mono_pre
    )
    pre_re = summary["arms"]["prefix-reordered"]["median_prefill_ms"]
    summary["gates"]["G5a_prefix_reordered_no_win"] = (
        pre_re is not None and abs(pre_re - mono_pre) <= 0.10 * mono_pre
    )
    ddi_re_total = summary["arms"]["ddi-reordered"]["median_total_ms"]
    summary["gates"]["G5b_ddi_reordered_total_under_mono_total"] = (
        ddi_re_total is not None and ddi_re_total < mono_total
    )
    match = sum(
        1
        for a, b in zip(identity["prefix-fixed"], identity["monolithic"])
        if _ids_sha256(a) == _ids_sha256(b)
    )
    summary["gates"]["prefix_hit_output_match_fraction_vs_mono"] = round(match / len(identity["prefix-fixed"]), 4)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "rows.json").write_text(json.dumps(results, indent=2) + "\n")
    log(f"done: {out_dir}")
    print(json.dumps(summary["gates"], indent=1))
    return 0


def _ddi_store(model, contract, docs, cache_dir: Path) -> DocCache:
    """All-hit DocCache pre-state for the five packed documents."""
    store = DocCache(cache_dir)
    writer = CanonicalCache(model, contract, store)
    writer.documents([s for s in docs])
    return store


if __name__ == "__main__":
    raise SystemExit(main())
