"""V59 staged fresh/cold/warm document-KV TTFT and crossover benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """Clear CUDA peak counters so the next arm measures its own footprint."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_block(device: torch.device) -> dict:
    """Capture peak CUDA memory for one arm, or nulls when no device is reported."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def summarize(values: list[float]) -> dict:
    return {
        "raw_ms": values,
        "median_ms": statistics.median(values),
        "iqr_ms": np.quantile(np.asarray(values), [0.25, 0.75]).tolist(),
    }


def partial_hit_schedule(n_docs: int, hit_count: int, seed: int) -> list[bool]:
    """Choose an exact, reproducible set of cache hits for one request."""
    if n_docs < 1:
        raise ValueError(f"n_docs must be >= 1, got {n_docs}")
    if not 0 <= hit_count <= n_docs:
        raise ValueError(f"hit_count must be in [0, {n_docs}], got {hit_count}")
    hits = np.zeros(n_docs, dtype=bool)
    hits[np.random.default_rng(seed).choice(n_docs, size=hit_count, replace=False)] = True
    return hits.tolist()


def stage_accounting(total_ms: float, stages: dict[str, float]) -> dict:
    """Make the uncovered portion of an end-to-end timing explicit."""
    stage_total_ms = sum(stages.values())
    return {
        "stage_total_ms": stage_total_ms,
        "residual_ms": total_ms - stage_total_ms,
    }


def paired_overhead(instrumented: dict, control: dict) -> dict:
    """Report the timing cost of stage instrumentation against its paired control."""
    instrumented_ms = instrumented["total_ms"]
    control_ms = control["total_ms"]
    return {
        "instrumented_ms": instrumented_ms,
        "control_ms": control_ms,
        "overhead_ms": instrumented_ms - control_ms,
    }


def _query_inputs(model, lengths: list[int], query: list[int]):
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    query_len = len(query)
    doc_total = sum(lengths)
    positions = build_positions(lengths, query_len, 0, "restart")
    query_positions = positions[doc_total:].unsqueeze(0).to(model.device)
    cache_position = torch.arange(
        doc_total, doc_total + query_len, dtype=torch.long, device=model.device
    )
    allow = build_mask(lengths, query_len, 0, "ddi_qf")
    mask = (
        to_additive(allow[doc_total:, :], model.dtype)
        .to(model.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    query_ids = torch.tensor([query], dtype=torch.long, device=model.device)
    return query_ids, query_positions, cache_position, mask


def _forced_decode(
    model,
    lengths: list[int],
    query_len: int,
    past_key_values,
    tokens: list[int],
) -> float:
    """Run a fixed number of cached decode forwards with externally chosen inputs."""
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    if not tokens:
        return 0.0
    device = model.device
    prefix_len = sum(lengths) + query_len
    allow = build_mask(lengths, query_len, len(tokens), "ddi_qf")
    additive = to_additive(allow, model.dtype).to(device)
    positions = build_positions(lengths, query_len, len(tokens), "restart").to(device)
    past = past_key_values
    sync(device)
    started = time.perf_counter()
    for offset, token in enumerate(tokens):
        absolute = prefix_len + offset
        out = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            attention_mask=additive[absolute : absolute + 1, : absolute + 1]
            .clone()
            .unsqueeze(0)
            .unsqueeze(0),
            position_ids=positions[absolute : absolute + 1].clone().unsqueeze(0),
            cache_position=torch.tensor([absolute], dtype=torch.long, device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
    sync(device)
    return (time.perf_counter() - started) * 1000


def _natural_decode(
    model, lengths, query_len, past_key_values, first_token, max_length, eos_token_id
) -> dict:
    """Generate from the prefill argmax; this is diagnostic, not arm-comparable work."""
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions
    from scripts.validate_cache_workload import token_digest

    if max_length < 1:
        raise ValueError(f"natural decode max length must be >= 1, got {max_length}")
    device = model.device
    prefix_len = sum(lengths) + query_len
    tokens = [first_token]
    past = past_key_values
    forward_count = 0
    sync(device)
    started = time.perf_counter()
    while len(tokens) < max_length and (eos_token_id is None or tokens[-1] != eos_token_id):
        absolute = prefix_len + forward_count
        allow = build_mask(lengths, query_len, forward_count + 1, "ddi_qf")
        additive = to_additive(allow, model.dtype).to(device)
        positions = build_positions(lengths, query_len, forward_count + 1, "restart").to(device)
        out = model(
            input_ids=torch.tensor([[tokens[-1]]], dtype=torch.long, device=device),
            attention_mask=additive[absolute : absolute + 1, : absolute + 1]
            .clone()
            .unsqueeze(0)
            .unsqueeze(0),
            position_ids=positions[absolute : absolute + 1].clone().unsqueeze(0),
            cache_position=torch.tensor([absolute], dtype=torch.long, device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        tokens.append(int(out.logits[0, -1].argmax().item()))
        forward_count += 1
    sync(device)
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "label": "natural_diagnostic_not_arm_comparable",
        "max_length": max_length,
        "eos_token_id": eos_token_id,
        "generated_token_ids": tokens,
        "token_ids_sha256": token_digest(tokens),
        "stop_reason": (
            "eos" if eos_token_id is not None and tokens[-1] == eos_token_id else "limit"
        ),
        "decode_forward_count": forward_count,
        "natural_decode_total_ms": total_ms,
    }


@torch.no_grad()
def fresh_ttft(
    model,
    docs: list[list[int]],
    query: list[int],
    *,
    decode_length: int = 0,
    natural_decode_max_length: int = 0,
    eos_token_id: int | None = None,
    capture_memory: bool = True,
) -> dict:
    from ddi.masks import build_mask, to_additive
    from ddi.positions import build_positions

    device = model.device
    if capture_memory:
        reset_peak_memory(device)
    lengths = [len(ids) for ids in docs]
    sync(device)
    started = time.perf_counter()
    prefix = sum(docs, []) + query
    positions = build_positions(lengths, len(query), 0, "restart").unsqueeze(0).to(device)
    allow = build_mask(lengths, len(query), 0, "ddi_qf")
    mask = to_additive(allow, model.dtype).to(device).unsqueeze(0).unsqueeze(0)
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    sync(device)
    prepared = time.perf_counter()
    out = model(
        input_ids=input_ids,
        position_ids=positions,
        attention_mask=mask,
        # Keep the combined DDI prefill; caching only exposes KV for opt-in decode.
        use_cache=decode_length > 0 or natural_decode_max_length > 0,
    )
    token = int(out.logits[0, -1].argmax().item())
    sync(device)
    finished = time.perf_counter()
    result = {
        "total_ms": (finished - started) * 1000,
        "prepare_ms": (prepared - started) * 1000,
        "forward_ms": (finished - prepared) * 1000,
        "token": token,
    }
    if decode_length:
        from scripts.validate_cache_workload import token_digest

        decode_tokens = [token] * decode_length
        decode_total_ms = _forced_decode(
            model, lengths, len(query), out.past_key_values, decode_tokens
        )
        result["metrics"] = {
            "ttft_ms": result["total_ms"],
            "decode_total_ms": decode_total_ms,
            "request_total_ms": result["total_ms"] + decode_total_ms,
            "tpot_ms": decode_total_ms / decode_length,
        }
        result["decode"] = {
            "length": decode_length,
            "work_count": decode_length,
            "token_ids_sha256": token_digest(decode_tokens),
        }
    if natural_decode_max_length:
        natural_decode = _natural_decode(
            model,
            lengths,
            len(query),
            out.past_key_values,
            token,
            natural_decode_max_length,
            eos_token_id,
        )
        natural_decode["natural_request_total_ms"] = result["total_ms"] + natural_decode[
            "natural_decode_total_ms"
        ]
        result["natural_decode"] = natural_decode
    if capture_memory:
        result["memory"] = peak_memory_block(device)
    return result


def _entry(model, ids: list[int], identity: str):
    from ddi.cache import compute_doc_kv
    from scripts.validate_cache_workload import TokenInput, digest, json_bytes

    key = digest(json_bytes([identity, ids]))
    return replace(
        compute_doc_kv(
            model,
            TokenInput(),
            json.dumps(ids),
            "restart",
            cache_identity=identity,
        ),
        key=key,
    )


@torch.no_grad()
def cached_ttft(
    model,
    docs: list[list[int]],
    query: list[int],
    cache,
    identity: str,
    *,
    populate: bool,
    instrumented: bool = True,
    pair_id: str | None = None,
    expected_hits: list[bool] | None = None,
    decode_tokens: list[int] | None = None,
    natural_decode_max_length: int = 0,
    eos_token_id: int | None = None,
    capture_memory: bool = True,
) -> dict:
    from ddi.cache import assemble_dynamic_cache
    from scripts.validate_cache_workload import digest, json_bytes, token_digest

    device = model.device
    if capture_memory:
        reset_peak_memory(device)
    sync(device)
    started = time.perf_counter()
    entries = []
    hits = []
    for ids in docs:
        key = digest(json_bytes([identity, ids]))
        entry = cache.get(
            key,
            expected_identity=identity,
            expected_token_ids_sha256=token_digest(ids),
        )
        hits.append(entry is not None)
        if entry is None:
            if not populate:
                raise RuntimeError(f"missing warm cache entry {key}")
            entry = _entry(model, ids, identity)
            cache.put(entry)
        entries.append(entry)
    built = time.perf_counter() if instrumented else None

    # Reload after a cold write so both cold and warm arms consume persisted payloads.
    if populate:
        entries = [
            cache.get(
                entry.key,
                expected_identity=identity,
                expected_token_ids_sha256=entry.token_ids_sha256,
            )
            for entry in entries
        ]
        if any(entry is None for entry in entries):
            raise RuntimeError("persisted cache reload failed")
    loaded = time.perf_counter() if instrumented else None
    cache_state = assemble_dynamic_cache(entries, device)  # type: ignore[arg-type]
    sync(device)
    assembled = time.perf_counter() if instrumented else None
    query_ids, positions, cache_position, mask = _query_inputs(
        model, [len(ids) for ids in docs], query
    )
    sync(device)
    prepared = time.perf_counter() if instrumented else None
    out = model(
        input_ids=query_ids,
        attention_mask=mask,
        position_ids=positions,
        cache_position=cache_position,
        past_key_values=cache_state,
        use_cache=True,
    )
    token = int(out.logits[0, -1].argmax().item())
    sync(device)
    finished = time.perf_counter()
    total_ms = (finished - started) * 1000
    if expected_hits is not None and hits != expected_hits:
        raise RuntimeError(f"cache hit schedule mismatch: expected {expected_hits}, got {hits}")
    result = {
        "total_ms": total_ms,
        "hits": hits,
        "token": token,
        "metrics": {"ttft_ms": total_ms, "request_total_ms": None, "tpot_ms": None},
        "measurement": {"instrumented": instrumented, "pair_id": pair_id},
    }
    if decode_tokens:
        decode_total_ms = _forced_decode(
            model,
            [len(ids) for ids in docs],
            len(query),
            out.past_key_values,
            decode_tokens,
        )
        result["metrics"] = {
            "ttft_ms": total_ms,
            "decode_total_ms": decode_total_ms,
            "request_total_ms": total_ms + decode_total_ms,
            "tpot_ms": decode_total_ms / len(decode_tokens),
        }
        result["decode"] = {
            "length": len(decode_tokens),
            "work_count": len(decode_tokens),
            "token_ids_sha256": token_digest(decode_tokens),
        }
    if natural_decode_max_length:
        natural_decode = _natural_decode(
            model,
            [len(ids) for ids in docs],
            len(query),
            out.past_key_values,
            token,
            natural_decode_max_length,
            eos_token_id,
        )
        natural_decode["natural_request_total_ms"] = total_ms + natural_decode[
            "natural_decode_total_ms"
        ]
        result["natural_decode"] = natural_decode
    if instrumented:
        stages = {
            "build_or_lookup_ms": (built - started) * 1000,  # type: ignore[operator]
            "persisted_reload_ms": (loaded - built) * 1000,  # type: ignore[operator]
            "h2d_assembly_ms": (assembled - loaded) * 1000,  # type: ignore[operator]
            "query_prepare_ms": (prepared - assembled) * 1000,  # type: ignore[operator]
            "query_forward_ms": (finished - prepared) * 1000,  # type: ignore[operator]
        }
        result["stages"] = stages
        result["accounting"] = stage_accounting(total_ms, stages)
        result["stage_sum_error_ms"] = abs(result["accounting"]["residual_ms"])
    if capture_memory:
        result["memory"] = peak_memory_block(device)
    return result


def crossover(fresh_ms: float, cold_ms: float, warm_ms: float, build_ms: float) -> dict:
    hit_rate = None
    if cold_ms != warm_ms:
        candidate = (cold_ms - fresh_ms) / (cold_ms - warm_ms)
        if 0.0 <= candidate <= 1.0:
            hit_rate = candidate
    reuse_count = None
    if fresh_ms > warm_ms:
        reuse_count = build_ms / (fresh_ms - warm_ms)
    return {"hit_rate": hit_rate, "reuse_count": reuse_count}


def _seed_cache(cache, entries: list, hits: list[bool]) -> None:
    for entry, hit in zip(entries, hits, strict=True):
        if hit:
            cache.put(entry)


def partial_hit_pair(
    model,
    docs: list[list[int]],
    query: list[int],
    entries: list,
    identity: str,
    hits: list[bool],
    cache_root: Path,
    pair_id: str,
    decode_tokens: list[int] | None = None,
    natural_decode_max_length: int = 0,
    eos_token_id: int | None = None,
) -> dict:
    """Run one actual mixed-hit request twice from identical persisted cache state."""
    from ddi.cache import DocCache

    control_cache = DocCache(cache_root / "control")
    instrumented_cache = DocCache(cache_root / "instrumented")
    _seed_cache(control_cache, entries, hits)
    _seed_cache(instrumented_cache, entries, hits)
    control = cached_ttft(
        model,
        docs,
        query,
        control_cache,
        identity,
        populate=True,
        instrumented=False,
        pair_id=pair_id,
        expected_hits=hits,
        decode_tokens=decode_tokens,
        natural_decode_max_length=natural_decode_max_length,
        eos_token_id=eos_token_id,
        # Peak-memory resets stay out of the control's timing path.
        capture_memory=False,
    )
    instrumented = cached_ttft(
        model,
        docs,
        query,
        instrumented_cache,
        identity,
        populate=True,
        instrumented=True,
        pair_id=pair_id,
        expected_hits=hits,
        decode_tokens=decode_tokens,
        natural_decode_max_length=natural_decode_max_length,
        eos_token_id=eos_token_id,
        capture_memory=True,
    )
    return {
        "pair_id": pair_id,
        "control": control,
        "instrumented": instrumented,
        "overhead": paired_overhead(instrumented, control),
    }


def main() -> None:
    import transformers

    from ddi.cache import DocCache
    from ddi.inference import load as load_model
    from scripts.validate_cache_workload import digest, json_bytes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--Ns", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--Ls", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--Q", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--decode-length",
        type=int,
        default=0,
        help="fixed cached decode forwards per request; zero disables decode measurement",
    )
    parser.add_argument(
        "--natural-decode-max-length",
        type=int,
        default=0,
        help="bounded natural cached generation for diagnostics; not arm-comparable timing work",
    )
    parser.add_argument(
        "--partial-hit-sweep",
        action="store_true",
        help="retain actual mixed-hit requests for every hit count from zero through N",
    )
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("V59 requires at least ten retained repeats")
    if args.decode_length < 0:
        raise ValueError("decode length must be >= 0")
    if args.natural_decode_max_length < 0:
        raise ValueError("natural decode max length must be >= 0")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        *sorted((root / "ddi").rglob("*.py")),
        root / "scripts" / "validate_cache_workload.py",
        Path(__file__).resolve(),
    ]
    source_hashes = {
        str(path.relative_to(root)): digest(path.read_bytes()) for path in source_paths
    }
    source_dir = args.output / "source"
    for path in source_paths:
        target = source_dir / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    (args.output / "working.patch").write_bytes(diff)
    config = vars(args) | {
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "dtype": "bf16",
        "backend": "sdpa",
        "layout": "restart",
    }
    manifest = {
        "schema_version": 3,
        "config": config,
        "source_hashes": source_hashes,
        "diff_sha256": digest(diff),
        "base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "started_unix": time.time(),
        "clean_revision": False,
        "admissible_paper_evidence": False,
    }
    identity = digest(json_bytes(manifest))
    manifest["cache_identity"] = identity
    (args.output / "manifest.json").write_bytes(json_bytes(manifest))

    model, tokenizer = load_model(str(args.checkpoint))
    model = model.to("cuda").eval()
    rows = []
    eos_token_id = tokenizer.eos_token_id
    with (args.output / "cells.jsonl").open("x") as output:
        for seed in args.seeds:
            for n_docs in args.Ns:
                for doc_length in args.Ls:
                    rng = np.random.default_rng(seed * 10_000 + n_docs * 100 + doc_length)
                    docs = rng.integers(
                        0, tokenizer.vocab_size, (n_docs, doc_length)
                    ).tolist()
                    query = rng.integers(0, tokenizer.vocab_size, args.Q).tolist()
                    warm_cache = DocCache(
                        args.output / "cache" / f"seed{seed}_N{n_docs}_L{doc_length}"
                    )
                    entries = [_entry(model, ids, identity) for ids in docs]
                    for entry in entries:
                        warm_cache.put(entry)
                    storage_bytes = sum(
                        path.stat().st_size for path in warm_cache.cache_dir.glob("*.pt")
                    )
                    for _ in range(args.warmups):
                        warmup_fresh = fresh_ttft(
                            model,
                            docs,
                            query,
                            decode_length=args.decode_length,
                            natural_decode_max_length=args.natural_decode_max_length,
                            eos_token_id=eos_token_id,
                        )
                        cached_ttft(
                            model,
                            docs,
                            query,
                            warm_cache,
                            identity,
                            populate=False,
                            decode_tokens=[warmup_fresh["token"]] * args.decode_length,
                            natural_decode_max_length=args.natural_decode_max_length,
                            eos_token_id=eos_token_id,
                        )

                    fresh_rows = []
                    warm_rows = []
                    cold_rows = []
                    for _repeat in range(args.repeats):
                        fresh = fresh_ttft(
                            model,
                            docs,
                            query,
                            decode_length=args.decode_length,
                            natural_decode_max_length=args.natural_decode_max_length,
                            eos_token_id=eos_token_id,
                        )
                        fresh_rows.append(fresh)
                        decode_tokens = [fresh["token"]] * args.decode_length
                        warm_rows.append(
                            cached_ttft(
                                model,
                                docs,
                                query,
                                warm_cache,
                                identity,
                                populate=False,
                                decode_tokens=decode_tokens,
                                natural_decode_max_length=args.natural_decode_max_length,
                                eos_token_id=eos_token_id,
                            )
                        )
                        with tempfile.TemporaryDirectory(
                            prefix="v59-cold-", dir="/var/tmp/opencode"
                        ) as temp_dir:
                            cold_rows.append(
                                cached_ttft(
                                    model,
                                    docs,
                                    query,
                                    DocCache(Path(temp_dir)),
                                    identity,
                                    populate=True,
                                    decode_tokens=decode_tokens,
                                    natural_decode_max_length=args.natural_decode_max_length,
                                    eos_token_id=eos_token_id,
                                )
                            )
                    fresh_values = [item["total_ms"] for item in fresh_rows]
                    cold_values = [item["total_ms"] for item in cold_rows]
                    warm_values = [item["total_ms"] for item in warm_rows]
                    build_values = [
                        item["stages"]["build_or_lookup_ms"] for item in cold_rows
                    ]
                    medians = {
                        "fresh_ms": statistics.median(fresh_values),
                        "cold_ms": statistics.median(cold_values),
                        "warm_ms": statistics.median(warm_values),
                        "build_ms": statistics.median(build_values),
                    }
                    row = {
                        "seed": seed,
                        "N": n_docs,
                        "L": doc_length,
                        "Q": args.Q,
                        "input_sha256": digest(json_bytes({"docs": docs, "query": query})),
                        "fresh": {"summary": summarize(fresh_values), "repeats": fresh_rows},
                        "cold": {"summary": summarize(cold_values), "repeats": cold_rows},
                        "warm": {"summary": summarize(warm_values), "repeats": warm_rows},
                        "storage_bytes": storage_bytes,
                        "bytes_per_doc": storage_bytes / n_docs,
                        "bytes_per_doc_token": storage_bytes / (n_docs * doc_length),
                        "warm_speedup": medians["fresh_ms"] / medians["warm_ms"],
                        "crossover": crossover(**medians),
                        "partial_hit": [],
                    }
                    if args.partial_hit_sweep:
                        for hit_count in range(n_docs + 1):
                            schedule_seed = (
                                seed * 1_000_000 + n_docs * 10_000 + doc_length * 10 + hit_count
                            )
                            hits = partial_hit_schedule(n_docs, hit_count, schedule_seed)
                            pairs = []
                            for repeat in range(args.repeats):
                                with tempfile.TemporaryDirectory(
                                    prefix="v59-partial-", dir="/var/tmp/opencode"
                                ) as temp_dir:
                                    pair_id = (
                                        f"partial-s{seed}-N{n_docs}-L{doc_length}-"
                                        f"h{hit_count}-r{repeat}"
                                    )
                                    pairs.append(
                                        partial_hit_pair(
                                            model,
                                            docs,
                                            query,
                                            entries,
                                            identity,
                                            hits,
                                            Path(temp_dir),
                                            pair_id,
                                            [fresh_rows[repeat]["token"]] * args.decode_length,
                                            args.natural_decode_max_length,
                                            eos_token_id,
                                        )
                                    )
                            row["partial_hit"].append(
                                {
                                    "hit_count": hit_count,
                                    "schedule_seed": schedule_seed,
                                    "hits": hits,
                                    "pairs": pairs,
                                }
                            )
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    rows.append(row)
                    print(f"seed={seed} N={n_docs} L={doc_length}: complete", flush=True)

    summary = {
        "cell_count": len(rows),
        "warm_speedup": {
            "median": statistics.median(row["warm_speedup"] for row in rows),
            "range": [
                min(row["warm_speedup"] for row in rows),
                max(row["warm_speedup"] for row in rows),
            ],
        },
        "feasible_hit_rate_crossovers": sum(
            row["crossover"]["hit_rate"] is not None for row in rows
        ),
        "finite_reuse_crossovers": sum(
            row["crossover"]["reuse_count"] is not None for row in rows
        ),
        "max_stage_sum_error_ms": max(
            repeat["stage_sum_error_ms"]
            for row in rows
            for arm in ("cold", "warm")
            for repeat in row[arm]["repeats"]
        ),
        "wall_seconds": time.time() - manifest["started_unix"],
        "admissible_paper_evidence": False,
    }
    (args.output / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
