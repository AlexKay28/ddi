"""Independently audit V59 staged TTFT and crossover artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from scripts.benchmark_cache_ttft import (
    crossover,
    paired_overhead,
    partial_hit_schedule,
)


def _audit_decode(
    repeat: dict,
    decode_length: int,
    failures: list[str],
    label: str,
    expected_digest: str | None = None,
    require_metrics: bool = True,
) -> None:
    metrics = repeat.get("metrics")
    if decode_length == 0:
        expected_metrics = {
            "ttft_ms": repeat.get("total_ms"),
            "request_total_ms": None,
            "tpot_ms": None,
        }
        malformed_metrics = metrics != expected_metrics if require_metrics else metrics is not None
        if malformed_metrics or "decode" in repeat:
            failures.append(f"{label}: malformed disabled decode metrics")
        return
    decode = repeat.get("decode")
    if not isinstance(decode, dict) or set(decode) != {
        "length",
        "work_count",
        "token_ids_sha256",
    }:
        failures.append(f"{label}: malformed decode metadata")
        return
    if decode["length"] != decode_length or decode["work_count"] != decode_length:
        failures.append(f"{label}: decode work count mismatch")
    if not isinstance(decode["token_ids_sha256"], str) or not decode["token_ids_sha256"]:
        failures.append(f"{label}: decode token digest is malformed")
    if expected_digest is not None and decode["token_ids_sha256"] != expected_digest:
        failures.append(f"{label}: forced decode tokens differ from fresh pair")
    if not isinstance(metrics, dict) or set(metrics) != {
        "ttft_ms",
        "decode_total_ms",
        "request_total_ms",
        "tpot_ms",
    }:
        failures.append(f"{label}: malformed enabled decode metrics")
        return
    if metrics["ttft_ms"] != repeat.get("total_ms"):
        failures.append(f"{label}: TTFT metric mismatch")
        return
    if not all(
        isinstance(metrics[name], (int, float)) and math.isfinite(metrics[name]) for name in metrics
    ):
        failures.append(f"{label}: enabled decode metric is non-numeric")
        return
    if not math.isclose(
        metrics["request_total_ms"], metrics["ttft_ms"] + metrics["decode_total_ms"], abs_tol=1e-9
    ):
        failures.append(f"{label}: request total is inconsistent with TTFT and decode")
    if not math.isclose(
        metrics["tpot_ms"], metrics["decode_total_ms"] / decode_length, abs_tol=1e-9
    ):
        failures.append(f"{label}: TPOT mismatch")


def _audit_natural_decode(
    repeat: dict, max_length: int, failures: list[str], label: str
) -> None:
    natural = repeat.get("natural_decode")
    if max_length == 0:
        if natural is not None:
            failures.append(f"{label}: unlabeled natural decode data")
        return
    expected_fields = {
        "label",
        "max_length",
        "eos_token_id",
        "generated_token_ids",
        "token_ids_sha256",
        "stop_reason",
        "decode_forward_count",
        "natural_decode_total_ms",
        "natural_request_total_ms",
    }
    if not isinstance(natural, dict) or set(natural) != expected_fields:
        failures.append(f"{label}: malformed natural decode metadata")
        return
    from scripts.validate_cache_workload import token_digest

    if natural["label"] != "natural_diagnostic_not_arm_comparable":
        failures.append(f"{label}: natural decode is not labeled diagnostic")
    tokens = natural["generated_token_ids"]
    if (
        natural["max_length"] != max_length
        or not isinstance(tokens, list)
        or not 1 <= len(tokens) <= max_length
        or not all(isinstance(token, int) for token in tokens)
    ):
        failures.append(f"{label}: natural decode token bounds mismatch")
    if natural["stop_reason"] not in {"eos", "limit"}:
        failures.append(f"{label}: natural decode stop reason is malformed")
    elif natural["stop_reason"] == "eos" and (
        not isinstance(natural["eos_token_id"], int) or tokens[-1] != natural["eos_token_id"]
    ):
        failures.append(f"{label}: natural decode EOS stop is inconsistent")
    elif natural["stop_reason"] == "limit" and len(tokens) != max_length:
        failures.append(f"{label}: natural decode limit stop is inconsistent")
    if natural["decode_forward_count"] != len(tokens) - 1:
        failures.append(f"{label}: natural decode forward count mismatch")
    if not isinstance(natural["token_ids_sha256"], str) or not natural["token_ids_sha256"]:
        failures.append(f"{label}: natural decode token digest is malformed")
    elif natural["token_ids_sha256"] != token_digest(tokens):
        failures.append(f"{label}: natural decode token digest mismatch")
    if not all(
        isinstance(natural[name], (int, float)) and math.isfinite(natural[name])
        for name in ("natural_decode_total_ms", "natural_request_total_ms")
    ):
        failures.append(f"{label}: natural decode timing is non-numeric")
    elif not math.isclose(
        natural["natural_request_total_ms"],
        repeat.get("total_ms") + natural["natural_decode_total_ms"],
        abs_tol=1e-9,
    ):
        failures.append(f"{label}: natural request total is inconsistent")


def _audit_memory(repeat: dict, failures: list[str], label: str) -> None:
    memory = repeat.get("memory")
    if "memory" not in repeat:
        return
    if not isinstance(memory, dict) or set(memory) != {
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }:
        failures.append(f"{label}: malformed peak memory block")
        return
    values = (memory["peak_allocated_bytes"], memory["peak_reserved_bytes"])
    if not all(
        value is None
        or (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        for value in values
    ):
        failures.append(f"{label}: peak memory values must be positive integers or null")
        return
    if any(value is None for value in values) and torch.cuda.is_available():
        failures.append(f"{label}: null peak memory requires a host without CUDA devices")


def _audit_instrumented_repeat(
    repeat: dict,
    decode_length: int,
    failures: list[str],
    label: str,
    expected_digest: str | None = None,
    natural_decode_max_length: int = 0,
) -> None:
    measurement = repeat.get("measurement")
    accounting = repeat.get("accounting")
    if measurement is None or measurement.get("instrumented") is not True:
        failures.append(f"{label}: missing instrumented measurement marker")
    _audit_memory(repeat, failures, label)
    _audit_decode(repeat, decode_length, failures, label, expected_digest)
    _audit_natural_decode(repeat, natural_decode_max_length, failures, label)
    if not isinstance(accounting, dict) or set(accounting) != {
        "stage_total_ms",
        "residual_ms",
    }:
        failures.append(f"{label}: malformed stage accounting")
        return
    stage_total = sum(repeat.get("stages", {}).values())
    if not math.isclose(stage_total, accounting["stage_total_ms"], abs_tol=1e-9):
        failures.append(f"{label}: stage total mismatch")
    if not math.isclose(
        repeat["total_ms"], accounting["stage_total_ms"] + accounting["residual_ms"], abs_tol=1e-9
    ):
        failures.append(f"{label}: residual mismatch")


def _audit_partial_hit(
    row: dict,
    repeats: int,
    n_docs: int,
    decode_length: int,
    natural_decode_max_length: int,
    failures: list[str],
) -> None:
    schedules = row.get("partial_hit")
    if not isinstance(schedules, list):
        failures.append("partial hit schedules missing")
        return
    if not schedules:
        return
    if [item.get("hit_count") for item in schedules] != list(range(n_docs + 1)):
        failures.append("partial hit schedule membership differs from N")
        return
    for schedule in schedules:
        hits = schedule.get("hits")
        schedule_seed = schedule.get("schedule_seed")
        if not isinstance(schedule_seed, int):
            failures.append("partial hit schedule seed is malformed")
            continue
        expected = partial_hit_schedule(n_docs, schedule["hit_count"], schedule_seed)
        if hits != expected:
            failures.append("partial hit schedule is not reproducible")
        pairs = schedule.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != repeats:
            failures.append("partial hit pair count mismatch")
            continue
        for repeat_index, pair in enumerate(pairs):
            control, instrumented = pair.get("control"), pair.get("instrumented")
            if not isinstance(control, dict) or not isinstance(instrumented, dict):
                failures.append("partial hit pair is incomplete")
                continue
            if pair.get("pair_id") != control.get("measurement", {}).get("pair_id") or pair.get(
                "pair_id"
            ) != instrumented.get("measurement", {}).get("pair_id"):
                failures.append("partial hit pair id mismatch")
            if control.get("measurement", {}).get("instrumented") is not False:
                failures.append("partial hit control is not marked uninstrumented")
            if "stages" in control or "accounting" in control:
                failures.append("partial hit control contains instrumentation")
            if control.get("hits") != hits or instrumented.get("hits") != hits:
                failures.append("partial hit pair differs from its schedule")
            fresh_decode = row["fresh"]["repeats"][repeat_index].get("decode", {})
            expected_digest = fresh_decode.get("token_ids_sha256")
            _audit_decode(
                control,
                decode_length,
                failures,
                "partial hit control run",
                expected_digest,
            )
            _audit_natural_decode(
                control, natural_decode_max_length, failures, "partial hit control run"
            )
            _audit_memory(control, failures, "partial hit control run")
            _audit_instrumented_repeat(
                instrumented,
                decode_length,
                failures,
                "partial hit instrumented run",
                expected_digest,
                natural_decode_max_length,
            )
            if pair.get("overhead") != paired_overhead(instrumented, control):
                failures.append("partial hit overhead mismatch")


def audit_schema_fields(
    row: dict,
    schema_version: int,
    repeats: int,
    decode_length: int = 0,
    natural_decode_max_length: int = 0,
) -> list[str]:
    """Validate only fields introduced after the retained V59 artifact."""
    if schema_version < 2:
        return []
    failures: list[str] = []
    for repeat_index, fresh in enumerate(row["fresh"]["repeats"]):
        _audit_decode(
            fresh,
            decode_length,
            failures,
            "fresh run",
            require_metrics=decode_length > 0,
        )
        _audit_natural_decode(fresh, natural_decode_max_length, failures, "fresh run")
        _audit_memory(fresh, failures, "fresh run")
        expected_digest = fresh.get("decode", {}).get("token_ids_sha256")
        for arm in ("cold", "warm"):
            _audit_instrumented_repeat(
                row[arm]["repeats"][repeat_index],
                decode_length,
                failures,
                "retained arm",
                expected_digest,
                natural_decode_max_length,
            )
    _audit_partial_hit(
        row, repeats, row["N"], decode_length, natural_decode_max_length, failures
    )
    return failures


def main() -> None:
    from transformers import AutoTokenizer

    from scripts.validate_cache_workload import digest, json_bytes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.attempt / "manifest.json").read_text())
    config = manifest["config"]
    rows = [json.loads(line) for line in (args.attempt / "cells.jsonl").read_text().splitlines()]
    failures = []

    if digest((args.attempt / "working.patch").read_bytes()) != manifest["diff_sha256"]:
        failures.append("working patch digest mismatch")
    for relative, expected in manifest["source_hashes"].items():
        source = args.attempt / "source" / relative
        if not source.is_file() or digest(source.read_bytes()) != expected:
            failures.append(f"source digest mismatch: {relative}")

    checkpoint = Path(config["checkpoint"])
    vocab_size = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True).vocab_size
    expected_cells = [
        (seed, n_docs, doc_length)
        for seed in config["seeds"]
        for n_docs in config["Ns"]
        for doc_length in config["Ls"]
    ]
    if [(row["seed"], row["N"], row["L"]) for row in rows] != expected_cells:
        failures.append("cell order or membership differs from manifest")

    schema_version = manifest.get("schema_version", 1)
    if schema_version not in (1, 2, 3):
        failures.append(f"unsupported schema version: {schema_version}")
    token_mismatch_cells = 0
    for row in rows:
        seed, n_docs, doc_length = row["seed"], row["N"], row["L"]
        rng = np.random.default_rng(seed * 10_000 + n_docs * 100 + doc_length)
        docs = rng.integers(0, vocab_size, (n_docs, doc_length)).tolist()
        query = rng.integers(0, vocab_size, config["Q"]).tolist()
        if digest(json_bytes({"docs": docs, "query": query})) != row["input_sha256"]:
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: input digest mismatch")

        for arm in ("fresh", "cold", "warm"):
            repeats = row[arm]["repeats"]
            values = [repeat["total_ms"] for repeat in repeats]
            if len(repeats) != config["repeats"]:
                failures.append(f"seed={seed} N={n_docs} L={doc_length}: repeat mismatch")
            if not math.isclose(
                statistics.median(values), row[arm]["summary"]["median_ms"]
            ):
                failures.append(f"seed={seed} N={n_docs} L={doc_length}: median mismatch")
        if any(any(repeat["hits"]) for repeat in row["cold"]["repeats"]):
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: cold hit")
        if any(not all(repeat["hits"]) for repeat in row["warm"]["repeats"]):
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: warm miss")
        for arm in ("cold", "warm"):
            for repeat in row[arm]["repeats"]:
                stage_sum = sum(repeat["stages"].values())
                if not math.isclose(stage_sum, repeat["total_ms"], abs_tol=1e-9):
                    failures.append(
                        f"seed={seed} N={n_docs} L={doc_length}: stage sum mismatch"
                    )
        fresh_ms = row["fresh"]["summary"]["median_ms"]
        cold_ms = row["cold"]["summary"]["median_ms"]
        warm_ms = row["warm"]["summary"]["median_ms"]
        build_ms = statistics.median(
            repeat["stages"]["build_or_lookup_ms"]
            for repeat in row["cold"]["repeats"]
        )
        if row["crossover"] != crossover(fresh_ms, cold_ms, warm_ms, build_ms):
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: crossover mismatch")
        if not math.isclose(row["warm_speedup"], fresh_ms / warm_ms):
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: speedup mismatch")

        cache_dir = args.attempt / "cache" / f"seed{seed}_N{n_docs}_L{doc_length}"
        cache_files = list(cache_dir.glob("*.pt"))
        actual_storage = sum(path.stat().st_size for path in cache_files)
        if len(cache_files) != n_docs or actual_storage != row["storage_bytes"]:
            failures.append(f"seed={seed} N={n_docs} L={doc_length}: storage mismatch")
        fresh_tokens = {repeat["token"] for repeat in row["fresh"]["repeats"]}
        warm_tokens = {repeat["token"] for repeat in row["warm"]["repeats"]}
        token_mismatch_cells += fresh_tokens != warm_tokens
        failures.extend(
            audit_schema_fields(
                row,
                schema_version,
                config["repeats"],
                config.get("decode_length", 0),
                config.get("natural_decode_max_length", 0),
            )
        )

    audit = {
        "pass": not failures,
        "failures": failures,
        "cell_count": len(rows),
        "token_mismatch_cells": token_mismatch_cells,
        "cache_bytes": sum(row["storage_bytes"] for row in rows),
    }
    (args.attempt / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
