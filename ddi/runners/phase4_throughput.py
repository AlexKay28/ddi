"""Phase 4.4 throughput-under-reuse runner (issue #25).

Registers `task: phase4_throughput`. Simulates a realistic RAG workload — a stream of
requests each drawing docs from a small universe under a reuse distribution — and
measures end-to-end QPS under three modes:

  - `ddi_cache`     — `DocCache` served; misses run `compute_doc_kv`, hits skip it.
  - `ddi_recompute` — cache always empty; every doc costs a fresh `compute_doc_kv`.
  - `full_causal`   — no per-doc computation; one causal forward on the full prompt.

WHY the first-pass cache path skips `compute_doc_kv` on hit rather than injecting cached
KVs into the model's `past_key_values`: full injection needs stitching per-doc caches
into HF's `DynamicCache` at the right per-layer indices, which is a non-trivial engineering
task (issue #25 §4.4 follow-up). Skipping the doc-prefill compute on hit measures the
compute that the cache would save; that is Phase 4.4's first-pass claim. Full injection
follows once the tensor-plumbing is worked out.

The doc universe is synthetic: `top_docs` random doc-length token sequences generated
deterministically from the seed. Real retrieval corpora would give more realistic
distributions of doc length and content, but neither shape affects the QPS story that
this runner is measuring — that is dominated by (docs_per_request, doc_len, hit_rate).
Swap-in of a real corpus is a follow-up.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ddi.cache import DocCache, compute_key
from ddi.cli import register_runner
from ddi.masks import build_mask, to_additive
from ddi.model import load_model
from ddi.positions import Layout, build_positions
from ddi.provenance import capture
from ddi.store import append

# Small enough to keep synthetic doc costs reasonable on smoke runs; the runner reads
# `doc_token_len` from the config, defaulting here.
_DEFAULT_DOC_TOKEN_LEN = 500
_DEFAULT_QUERY_TOKEN_LEN = 32
_DEFAULT_ANSWER_TOKEN_LEN = 16


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
        # Every request picks the same top-k slice; hit rate saturates immediately after
        # the first request. Useful as an upper bound / degenerate control.
        base = list(range(docs_per_request))
        return [list(base) for _ in range(n_requests)]

    requests: list[list[int]] = []
    if distribution == "uniform":
        for _ in range(n_requests):
            requests.append(
                rng.choice(top_docs, size=docs_per_request, replace=False).tolist()
            )
        return requests

    # zipf: sample with weights proportional to (rank + 1) ** -a, then reject-resample
    # duplicates to keep each request a *set* of docs.
    ranks = np.arange(1, top_docs + 1, dtype=np.float64)
    weights = ranks ** (-zipf_a)
    weights /= weights.sum()
    for _ in range(n_requests):
        picked: list[int] = []
        seen: set[int] = set()
        # Cap the retry budget so a pathological (docs_per_request ~= top_docs, a >> 1)
        # config still terminates; if we can't find enough distinct indices, fill from
        # the least-frequent tail.
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


def _make_synthetic_universe(
    rng: np.random.Generator,
    tokenizer,
    top_docs: int,
    doc_token_len: int,
) -> list[str]:
    """Build `top_docs` synthetic doc texts, each ~ `doc_token_len` tokens after tokenisation.

    Uses random-integer token ids in [0, vocab_size) decoded through the tokenizer. The
    resulting text is nonsense but roundtrips to a stable token sequence per seed, which
    is all this runner needs — throughput here is a function of shape, not semantics.
    """
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    docs: list[str] = []
    for _ in range(top_docs):
        ids = rng.integers(low=0, high=vocab_size, size=doc_token_len).tolist()
        docs.append(tokenizer.decode(ids, skip_special_tokens=True))
    return docs


def _time_doc_prefill(model, tokenizer, doc_text: str, layout: Layout) -> float:
    """Time one `compute_doc_kv`-shaped forward and return elapsed ms."""
    doc_ids = tokenizer(doc_text, return_tensors="pt", add_special_tokens=False).input_ids
    doc_len = int(doc_ids.shape[-1])
    input_ids = doc_ids.to(model.device)
    positions = build_positions([doc_len], 0, 0, layout).unsqueeze(0).to(model.device)
    allow = build_mask([doc_len], query_len=0, answer_len=0, variant="full")
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=additive,
            position_ids=positions,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _time_query_prefill(
    model,
    tokenizer,
    query_text: str,
    n_new_tokens: int,
) -> float:
    """Time a query-only masked prefill + greedy decode; returns elapsed ms.

    Used by ddi_cache and ddi_recompute modes for the "post-doc" portion of a request.
    Docs are assumed already prefilled (or would be, in the injection follow-up); this
    is the shared per-request tail cost.
    """
    ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    q_len = ids.shape[-1]
    allow = build_mask([q_len], query_len=0, answer_len=0, variant="full")
    additive = to_additive(allow, dtype=model.dtype).to(model.device).unsqueeze(0).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=additive, use_cache=True)
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
        # forward_with_mask disables use_cache, but this is one prefill + a decode loop,
        # so call the model directly to get past_key_values back for the decode path.
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


def _run(config: dict[str, Any], config_hash: str, seed: int) -> None:
    """Run one throughput workload; see module docstring."""
    model_name = config["model"]

    if "mask" not in config:
        raise ValueError("phase4_throughput config missing required key: mask")
    mask_cfg = config["mask"]
    if "variant" not in mask_cfg:
        raise ValueError("phase4_throughput config.mask missing required key: variant")
    mask_variant: str = mask_cfg["variant"]
    bridge_k = mask_cfg.get("bridge_k")

    layout_raw = config.get("layout", "restart")
    if layout_raw not in ("sequential", "restart"):
        raise ValueError(f"unknown layout {layout_raw!r}; expected 'sequential' or 'restart'")
    layout: Layout = layout_raw  # type: ignore[assignment]

    mode = config.get("mode", "ddi_cache")
    if mode not in ("ddi_cache", "ddi_recompute", "full_causal"):
        raise ValueError(
            f"unknown mode {mode!r}; expected ddi_cache | ddi_recompute | full_causal"
        )

    dataset = config.get("dataset", "synthetic")
    n_requests = int(config.get("n_requests", 200))
    warmup_requests = int(config.get("warmup_requests", 20))
    reuse_distribution = config.get("reuse_distribution", "zipf")
    zipf_a = float(config.get("zipf_a", 1.5))
    top_docs = int(config.get("top_docs", 32))
    docs_per_request = int(config.get("docs_per_request", 5))
    doc_token_len = int(config.get("doc_token_len", _DEFAULT_DOC_TOKEN_LEN))
    query_token_len = int(config.get("query_token_len", _DEFAULT_QUERY_TOKEN_LEN))
    n_new_tokens = int(config.get("n_new_tokens", _DEFAULT_ANSWER_TOKEN_LEN))
    cache_dir = Path(config.get("cache_dir", "cache/doc_kv"))

    if n_requests < 1:
        raise ValueError(f"n_requests must be >= 1, got {n_requests}")
    if warmup_requests < 0:
        raise ValueError(f"warmup_requests must be >= 0, got {warmup_requests}")
    if warmup_requests >= n_requests:
        raise ValueError(
            f"warmup_requests={warmup_requests} must be < n_requests={n_requests}"
        )

    model, tokenizer = load_model(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    rng = np.random.default_rng(seed)
    docs = _make_synthetic_universe(rng, tokenizer, top_docs, doc_token_len)
    query_text = tokenizer.decode(
        rng.integers(
            low=0,
            high=getattr(tokenizer, "vocab_size", None) or len(tokenizer),
            size=query_token_len,
        ).tolist(),
        skip_special_tokens=True,
    )

    doc_index_stream = _sample_doc_indices(
        rng,
        n_requests + warmup_requests,
        docs_per_request,
        top_docs,
        reuse_distribution,
        zipf_a,
    )

    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    doc_keys = [compute_key(model_id, model.dtype, docs[i]) for i in range(top_docs)]

    cache: DocCache | None = None
    if mode == "ddi_cache":
        cache = DocCache(cache_dir)

    prov = capture(
        config_hash=config_hash,
        seed=seed,
        model_revision=model_name,
        dataset_revision=f"synthetic_throughput_top={top_docs}_L={doc_token_len}",
    )
    reuse_tag = (
        f"{reuse_distribution}(a={zipf_a})" if reuse_distribution == "zipf" else reuse_distribution
    )
    run_id = (
        f"phase4_throughput:{model_name}:{mode}:{layout}:reuse={reuse_tag}:"
        f"seed={seed}:{prov.git_sha[:8]}"
    )

    # Warmup: run the same request loop with timing discarded so JIT / kernel caches /
    # allocator state are warm before we start the counted portion.
    for req_i in range(warmup_requests):
        indices = doc_index_stream[req_i]
        _one_request(
            mode, model, tokenizer, indices, docs, doc_keys, cache, query_text, n_new_tokens, layout
        )

    hits = 0
    lookups = 0
    total_start = time.perf_counter()
    per_request_latency_ms: list[float] = []
    for req_i in range(n_requests):
        indices = doc_index_stream[warmup_requests + req_i]
        req_start = time.perf_counter()
        req_hits, req_lookups = _one_request(
            mode, model, tokenizer, indices, docs, doc_keys, cache, query_text, n_new_tokens, layout
        )
        req_ms = (time.perf_counter() - req_start) * 1000.0
        per_request_latency_ms.append(req_ms)
        hits += req_hits
        lookups += req_lookups
        append(f"{run_id}:req{req_i}", "per_request_latency_ms", req_ms, prov)
    total_elapsed_s = time.perf_counter() - total_start
    _ = dataset  # dataset config key is reserved for a future real-corpus swap-in.

    qps = n_requests / total_elapsed_s if total_elapsed_s > 0 else float("inf")
    append(run_id, "qps", qps, prov)
    append(run_id, "n_requests", n_requests, prov)
    append(run_id, "total_elapsed_s", total_elapsed_s, prov)
    if mode == "ddi_cache" and lookups > 0:
        append(run_id, "hit_rate", hits / lookups, prov)
    else:
        # Every non-cache mode has zero hit rate by construction; write the row anyway
        # so F5's join is one row per (run_id, hit_rate) regardless of mode.
        append(run_id, "hit_rate", 0.0, prov)


def _one_request(
    mode: str,
    model,
    tokenizer,
    doc_indices: list[int],
    docs: list[str],
    doc_keys: list[str],
    cache: DocCache | None,
    query_text: str,
    n_new_tokens: int,
    layout: Layout,
) -> tuple[int, int]:
    """Execute one request under `mode`; return (hits, lookups) for the cache accounting.

    Non-cache modes always return (0, 0).
    """
    if mode == "full_causal":
        _time_full_causal(
            model, tokenizer, [docs[i] for i in doc_indices], query_text, n_new_tokens
        )
        return 0, 0

    # ddi_recompute: cache is None; every doc counts as a miss and pays compute_doc_kv.
    # ddi_cache: hit → skip compute (first-pass simplification, see module docstring);
    #            miss → pay compute_doc_kv, then put into cache.
    hits = 0
    lookups = len(doc_indices)
    for doc_idx in doc_indices:
        if mode == "ddi_cache" and cache is not None and cache.has(doc_keys[doc_idx]):
            hits += 1
            continue
        _time_doc_prefill(model, tokenizer, docs[doc_idx], layout)
        if mode == "ddi_cache" and cache is not None:
            from ddi.cache import CacheEntry  # local import: only ddi_cache mode needs this

            # First-pass: we don't store the actual KVs (that would double compute cost during
            # measurement of the miss); we just record the key so subsequent hits are counted.
            # A follow-up will call compute_doc_kv() and put(entry) here to fill the cache.
            cache.put(
                CacheEntry(
                    key=doc_keys[doc_idx],
                    k_per_layer=[],
                    v_per_layer=[],
                    doc_len=0,
                    layout=layout,
                )
            )
    _time_query_prefill(model, tokenizer, query_text, n_new_tokens)
    return hits, lookups


register_runner("phase4_throughput")(_run)
