"""Aggregate V59 TTFT cells into paired geometry-level crossover estimates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def bootstrap_mean(values: list[float], *, seed: int) -> dict:
    samples = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for i in range(len(draws)):
        draws[i] = rng.choice(samples, size=len(samples), replace=True).mean()
    return {
        "values": values,
        "mean": float(samples.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def natural_decode_diagnostics(group: list[dict]) -> dict | None:
    """Keep generated-work timings separate from forced arm-comparable metrics."""
    arms = {}
    for arm in ("fresh", "cold", "warm"):
        repeats = [repeat for row in group for repeat in row[arm]["repeats"]]
        if not all("natural_decode" in repeat for repeat in repeats):
            return None
        natural = [repeat["natural_decode"] for repeat in repeats]
        arms[arm] = {
            name: statistics.median(item[name] for item in natural)
            for name in ("natural_decode_total_ms", "natural_request_total_ms")
        }
    return {"label": "natural_diagnostic_not_arm_comparable", "arms": arms}


def memory_diagnostics(group: list[dict]) -> dict | None:
    """Median per-arm peak CUDA memory when the artifact recorded memory blocks."""
    arms = {}
    for arm in ("fresh", "cold", "warm"):
        blocks = [repeat.get("memory") for row in group for repeat in row[arm]["repeats"]]
        if not all(isinstance(block, dict) for block in blocks):
            return None
        medians = {}
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            values = [block[name] for block in blocks if block[name] is not None]
            medians[name] = statistics.median(values) if values else None
        arms[arm] = medians
    return {"arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.attempt / "cells.jsonl").read_text().splitlines()]
    geometries = []
    for n_docs, doc_length in sorted({(row["N"], row["L"]) for row in rows}):
        group = [row for row in rows if row["N"] == n_docs and row["L"] == doc_length]
        speedups = [row["warm_speedup"] for row in group]
        hit_rates = [row["crossover"]["hit_rate"] for row in group]
        reuse_counts = [row["crossover"]["reuse_count"] for row in group]
        stages = [repeat["stages"] for row in group for repeat in row["warm"]["repeats"]]
        partial_pairs = [
            pair
            for row in group
            for schedule in row.get("partial_hit", [])
            for pair in schedule["pairs"]
        ]
        decode_metrics = {
            arm: {
                name: statistics.median(
                    repeat["metrics"][name] for row in group for repeat in row[arm]["repeats"]
                )
                for name in ("decode_total_ms", "request_total_ms", "tpot_ms")
            }
            for arm in ("fresh", "cold", "warm")
            if all(
                repeat.get("metrics", {}).get("decode_total_ms") is not None
                for row in group
                for repeat in row[arm]["repeats"]
            )
        }
        geometries.append(
            {
                "N": n_docs,
                "L": doc_length,
                "fresh_median_ms": statistics.median(
                    row["fresh"]["summary"]["median_ms"] for row in group
                ),
                "warm_median_ms": statistics.median(
                    row["warm"]["summary"]["median_ms"] for row in group
                ),
                "cold_median_ms": statistics.median(
                    row["cold"]["summary"]["median_ms"] for row in group
                ),
                "warm_speedup": bootstrap_mean(
                    speedups, seed=59 + n_docs * 100 + doc_length
                ),
                "hit_rate_crossover": (
                    bootstrap_mean(
                        hit_rates,  # type: ignore[arg-type]
                        seed=159 + n_docs * 100 + doc_length,
                    )
                    if all(value is not None for value in hit_rates)
                    else None
                ),
                "reuse_count_crossover": (
                    bootstrap_mean(
                        reuse_counts,  # type: ignore[arg-type]
                        seed=259 + n_docs * 100 + doc_length,
                    )
                    if all(value is not None for value in reuse_counts)
                    else None
                ),
                "warm_stage_mean_ms": {
                    name: statistics.mean(stage[name] for stage in stages)
                    for name in stages[0]
                },
                "bytes_per_doc_token": statistics.mean(
                    row["bytes_per_doc_token"] for row in group
                ),
                "partial_hit_overhead_ms": (
                    bootstrap_mean(
                        [pair["overhead"]["overhead_ms"] for pair in partial_pairs],
                        seed=359 + n_docs * 100 + doc_length,
                    )
                    if partial_pairs
                    else None
                ),
                "decode_metrics_ms": decode_metrics or None,
                "natural_decode_diagnostic_ms": natural_decode_diagnostics(group),
                "peak_memory_bytes": memory_diagnostics(group),
            }
        )
    analysis = {"geometries": geometries}
    (args.attempt / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
