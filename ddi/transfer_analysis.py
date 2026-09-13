"""Deterministic analysis for fixed-manifest, multi-seed transfer evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

MASKS = ("full", "ddi_qf", "blind")
PRIMARY_METRICS = ("em", "f1")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean(values: list[float], repeats: int, rng: random.Random) -> list[float]:
    return [statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_predictions(
    path: Path, expected_seed: int
) -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    checkpoint_revisions: set[str | None] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            provenance = row["provenance"]
            if provenance["seed"] != expected_seed:
                raise ValueError(f"{path}: expected seed {expected_seed}, got {provenance['seed']}")
            checkpoint_revisions.add(provenance.get("checkpoint_revision"))
            parts = row["run_id"].split(":")
            key = (parts[2], parts[3], row["record_id"])
            if key in rows:
                raise ValueError(f"{path}: duplicate prediction {key}")
            rows[key] = {metric: float(value) for metric, value in row["metrics"].items()}
    if len(checkpoint_revisions) != 1:
        raise ValueError(f"{path}: mixed checkpoint revisions {checkpoint_revisions}")
    return rows


def analyze(
    base_paths: dict[int, Path],
    tuned_paths: dict[int, Path],
    *,
    repeats: int = 10_000,
    bootstrap_seed: int = 46,
) -> dict[str, Any]:
    if set(base_paths) != set(tuned_paths) or len(base_paths) < 3:
        raise ValueError("base and tuned inputs must contain the same three or more seeds")

    loaded = {
        (arm, seed): _load_predictions(path, seed)
        for arm, paths in (("base", base_paths), ("tuned", tuned_paths))
        for seed, path in paths.items()
    }
    seeds = sorted(base_paths)
    datasets = sorted({key[0] for rows in loaded.values() for key in rows})
    rng = random.Random(bootstrap_seed)
    cells: list[dict[str, Any]] = []

    for dataset in datasets:
        for metric in PRIMARY_METRICS:
            seed_results = []
            for seed in seeds:
                base = loaded[("base", seed)]
                tuned = loaded[("tuned", seed)]
                id_sets = [
                    {
                        record_id
                        for ds, row_mask, record_id in rows
                        if ds == dataset and row_mask == mask
                    }
                    for rows in (base, tuned)
                    for mask in MASKS
                ]
                if not id_sets[0] or any(ids != id_sets[0] for ids in id_sets[1:]):
                    raise ValueError(f"dataset={dataset} seed={seed}: mask/arm record IDs differ")
                ids = sorted(id_sets[0])

                def values(
                    arm: str,
                    mask: str,
                    *,
                    current_seed: int = seed,
                    current_dataset: str = dataset,
                    current_metric: str = metric,
                    current_ids: list[str] = ids,
                ) -> list[float]:
                    rows = loaded[(arm, current_seed)]
                    return [
                        rows[(current_dataset, mask, record_id)][current_metric]
                        for record_id in current_ids
                    ]

                base_full = values("base", "full")
                contrasts = {
                    "headroom": [
                        a - b
                        for a, b in zip(base_full, values("base", "blind"), strict=True)
                    ],
                    "parity_gap": [
                        a - b
                        for a, b in zip(
                            values("tuned", "full"), values("tuned", "ddi_qf"), strict=True
                        )
                    ],
                    "preservation_gap": [
                        a - b for a, b in zip(base_full, values("tuned", "full"), strict=True)
                    ],
                    "blind_change": [
                        a - b
                        for a, b in zip(
                            values("tuned", "blind"), values("base", "blind"), strict=True
                        )
                    ],
                }
                contrast_results = {}
                for name, samples in contrasts.items():
                    draws = _bootstrap_mean(samples, repeats, rng)
                    contrast_results[name] = {
                        "mean": statistics.mean(samples),
                        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
                    }
                headroom = contrast_results["headroom"]["mean"]
                parity_gap = contrast_results["parity_gap"]["mean"]
                seed_results.append(
                    {
                        "seed": seed,
                        "records": len(ids),
                        "base_full": statistics.mean(base_full),
                        "normalized_closure": (
                            None if headroom == 0.0 else 1.0 - parity_gap / headroom
                        ),
                        "contrasts": contrast_results,
                    }
                )

            delta = 2.0 * statistics.stdev(row["base_full"] for row in seed_results)
            aggregate = {}
            for name in ("headroom", "parity_gap", "preservation_gap", "blind_change"):
                means = [row["contrasts"][name]["mean"] for row in seed_results]
                draws = _bootstrap_mean(means, repeats, rng)
                aggregate[name] = {
                    "seed_values": means,
                    "mean": statistics.mean(means),
                    "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
                }
            closures = [row["normalized_closure"] for row in seed_results]
            if all(value is not None for value in closures):
                closure_values = [float(value) for value in closures]
                closure_draws = _bootstrap_mean(closure_values, repeats, rng)
                aggregate["normalized_closure"] = {
                    "seed_values": closure_values,
                    "mean": statistics.mean(closure_values),
                    "ci95": [
                        _percentile(closure_draws, 0.025),
                        _percentile(closure_draws, 0.975),
                    ],
                }
            else:
                aggregate["normalized_closure"] = None

            threshold = max(0.05, 3.0 * delta)
            admissible = aggregate["headroom"]["mean"] > threshold
            parity = admissible and aggregate["parity_gap"]["ci95"][1] <= delta
            preservation = admissible and aggregate["preservation_gap"]["ci95"][1] <= delta
            blind_stable = admissible and abs(aggregate["blind_change"]["mean"]) <= delta
            if not admissible:
                verdict = "INADMISSIBLE"
            elif parity and preservation and blind_stable:
                verdict = "WITHIN_TOLERANCE"
            elif not blind_stable:
                verdict = "INVALID_BLIND_MOVED"
            elif not preservation:
                verdict = "FULL_PATH_REGRESSION"
            else:
                verdict = "PARITY_NOT_REACHED"
            cells.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "delta": delta,
                    "admissibility_threshold": threshold,
                    "admissible": admissible,
                    "parity_pass": parity,
                    "preservation_pass": preservation,
                    "blind_stability_pass": blind_stable,
                    "verdict": verdict,
                    "seeds": seed_results,
                    "aggregate": aggregate,
                }
            )

    return {
        "bootstrap": {
            "method": "paired percentile then equal-weight seed bootstrap",
            "repeats": repeats,
            "seed": bootstrap_seed,
        },
        "inputs": {
            arm: [
                {"seed": seed, "path": str(path), "sha256": _sha256(path)}
                for seed, path in sorted(paths.items())
            ]
            for arm, paths in (("base", base_paths), ("tuned", tuned_paths))
        },
        "cells": cells,
    }


def _parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed, path = value.split("=", 1)
        result[int(seed)] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--tuned", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=46)
    args = parser.parse_args()
    result = analyze(
        _parse_seed_paths(args.base),
        _parse_seed_paths(args.tuned),
        repeats=args.repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
