"""Post-hoc paired example-resampling sensitivity for transfer results.

This module deliberately does not alter ``transfer_analysis`` or its historical
decision output.  It reports additional estimands conditional on the retained
checkpoints and frozen evaluation records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from ddi.transfer_analysis import MASKS, _load_predictions, _percentile

METRICS = ("em", "f1", "presence", "loss")
ESTIMANDS = (
    "normalized_remaining_gap",
    "ddi_training_change",
    "full_path_change",
    "full_ddi_gap_change",
    "blind_change",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oriented(metric: str, value: float) -> float:
    return -value if metric == "loss" else value


def _summary(
    base_rows: dict[tuple[str, str, str], dict[str, float]],
    tuned_rows: dict[tuple[str, str, str], dict[str, float]],
    dataset: str,
    metric: str,
    ids: list[str],
) -> dict[str, float | None]:
    def arm(rows: dict[tuple[str, str, str], dict[str, float]], mask: str) -> float:
        return statistics.mean(
            _oriented(metric, rows[(dataset, mask, record_id)][metric]) for record_id in ids
        )

    base_full = arm(base_rows, "full")
    base_ddi = arm(base_rows, "ddi_qf")
    base_blind = arm(base_rows, "blind")
    tuned_full = arm(tuned_rows, "full")
    tuned_ddi = arm(tuned_rows, "ddi_qf")
    tuned_blind = arm(tuned_rows, "blind")
    headroom = base_full - base_blind
    tuned_gap = tuned_full - tuned_ddi
    return {
        "base_full": base_full,
        "base_ddi_qf": base_ddi,
        "base_blind": base_blind,
        "tuned_full": tuned_full,
        "tuned_ddi_qf": tuned_ddi,
        "tuned_blind": tuned_blind,
        "headroom": headroom,
        # Historical formula, relabelled for its actual numerator/denominator.
        "normalized_remaining_gap": None if headroom == 0.0 else 1.0 - tuned_gap / headroom,
        "ddi_training_change": tuned_ddi - base_ddi,
        "full_path_change": tuned_full - base_full,
        "full_ddi_gap_change": tuned_gap - (base_full - base_ddi),
        "blind_change": tuned_blind - base_blind,
    }


def _ci(draws: list[float]) -> list[float]:
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def analyze_sensitivity(
    architectures: dict[str, tuple[dict[int, Path], dict[int, Path]]],
    *,
    repeats: int = 10_000,
    bootstrap_seed: int = 54,
) -> dict[str, Any]:
    """Jointly resample paired records, keeping matching seeds aligned by ID.

    The intervals quantify only finite-record uncertainty conditional on the
    supplied three checkpoints and manifests.  Architecture contrasts use the
    record-ID intersection, so an incomplete overlap is explicit rather than
    silently treated as a matched population sample.
    """
    if len(architectures) < 2:
        raise ValueError("at least two architectures are required")
    seeds = None
    loaded: dict[tuple[str, str, int], dict[tuple[str, str, str], dict[str, float]]] = {}
    for architecture, (base_paths, tuned_paths) in architectures.items():
        if set(base_paths) != set(tuned_paths) or len(base_paths) < 3:
            raise ValueError(
                f"{architecture}: base and tuned inputs need the same three or more seeds"
            )
        if seeds is None:
            seeds = sorted(base_paths)
        elif seeds != sorted(base_paths):
            raise ValueError("architectures have different seed sets")
        for arm, paths in (("base", base_paths), ("tuned", tuned_paths)):
            for seed, path in paths.items():
                loaded[(architecture, arm, seed)] = _load_predictions(path, seed)
    assert seeds is not None

    datasets = sorted({dataset for rows in loaded.values() for dataset, _mask, _record_id in rows})
    cells: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    rng = random.Random(bootstrap_seed)
    names = sorted(architectures)

    for dataset in datasets:
        for metric in METRICS:
            ids_by_arch_seed: dict[tuple[str, int], set[str]] = {}
            for architecture in names:
                for seed in seeds:
                    id_sets = [
                        {
                            record_id
                            for ds, row_mask, record_id in loaded[(architecture, arm, seed)]
                            if ds == dataset and row_mask == mask
                        }
                        for arm in ("base", "tuned")
                        for mask in MASKS
                    ]
                    if not id_sets[0] or any(ids != id_sets[0] for ids in id_sets[1:]):
                        raise ValueError(
                            f"{architecture} dataset={dataset} seed={seed}: "
                            "mask/arm record IDs differ"
                        )
                    missing_metrics = [
                        record_id
                        for record_id in id_sets[0]
                        if metric
                        not in loaded[(architecture, "base", seed)][(dataset, "full", record_id)]
                    ]
                    if missing_metrics:
                        raise ValueError(f"{architecture} {dataset}/{metric}: metric is missing")
                    ids_by_arch_seed[(architecture, seed)] = id_sets[0]

            overlap = {
                str(seed): {
                    architecture: len(ids_by_arch_seed[(architecture, seed)])
                    for architecture in names
                }
                | {
                    "common": len(
                        set.intersection(
                            *(ids_by_arch_seed[(architecture, seed)] for architecture in names)
                        )
                    )
                }
                for seed in seeds
            }
            seed_values = {architecture: [] for architecture in names}
            for architecture in names:
                for seed in seeds:
                    seed_values[architecture].append(
                        _summary(
                            loaded[(architecture, "base", seed)],
                            loaded[(architecture, "tuned", seed)],
                            dataset,
                            metric,
                            sorted(ids_by_arch_seed[(architecture, seed)]),
                        )
                    )

            draws = {architecture: {name: [] for name in ESTIMANDS} for architecture in names}
            contrast_draws = {name: [] for name in ESTIMANDS}
            for _ in range(repeats):
                drawn = {architecture: {name: [] for name in ESTIMANDS} for architecture in names}
                for seed in seeds:
                    common_ids = sorted(
                        set.intersection(
                            *(ids_by_arch_seed[(architecture, seed)] for architecture in names)
                        )
                    )
                    sample = rng.choices(common_ids, k=len(common_ids))
                    for architecture in names:
                        summary = _summary(
                            loaded[(architecture, "base", seed)],
                            loaded[(architecture, "tuned", seed)],
                            dataset,
                            metric,
                            sample,
                        )
                        for name in ESTIMANDS:
                            value = summary[name]
                            if value is not None:
                                drawn[architecture][name].append(float(value))
                for architecture in names:
                    for name in ESTIMANDS:
                        values = drawn[architecture][name]
                        if len(values) == len(seeds):
                            draws[architecture][name].append(statistics.mean(values))
                for name in ESTIMANDS:
                    if all(draws[architecture][name] for architecture in names):
                        contrast_draws[name].append(
                            draws[names[1]][name][-1] - draws[names[0]][name][-1]
                        )

            for architecture in names:
                aggregate = {}
                for name in ESTIMANDS:
                    values = [seed_result[name] for seed_result in seed_values[architecture]]
                    aggregate[name] = {
                        "seed_values": values,
                        "mean": None
                        if any(value is None for value in values)
                        else statistics.mean(values),
                        "posthoc_paired_ci95": None
                        if not draws[architecture][name]
                        else _ci(draws[architecture][name]),
                    }
                cells.append(
                    {
                        "architecture": architecture,
                        "dataset": dataset,
                        "metric": metric,
                        "metric_orientation": "lower_is_better"
                        if metric == "loss"
                        else "higher_is_better",
                        "seed_values": seed_values[architecture],
                        "aggregate": aggregate,
                        "manifest_overlap": overlap,
                    }
                )
            for name in ESTIMANDS:
                values = [
                    float(seed_values[names[1]][index][name])
                    - float(seed_values[names[0]][index][name])
                    for index in range(len(seeds))
                    if seed_values[names[0]][index][name] is not None
                    and seed_values[names[1]][index][name] is not None
                ]
                contrasts.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "estimand": name,
                        "difference_direction": f"{names[1]} - {names[0]}",
                        "seed_values": values,
                        "mean": statistics.mean(values) if values else None,
                        "posthoc_paired_ci95": _ci(contrast_draws[name])
                        if contrast_draws[name]
                        else None,
                        "manifest_overlap": overlap,
                    }
                )

    return {
        "analysis": "post_hoc_paired_example_resampling_sensitivity",
        "interpretation": (
            "conditional on these fixed checkpoints and retained manifests; "
            "not a population confidence interval"
        ),
        "bootstrap": {
            "repeats": repeats,
            "seed": bootstrap_seed,
            "unit": "paired record within fixed seed",
        },
        "inputs": {
            architecture: {
                arm: [
                    {"seed": seed, "path": str(path), "sha256": _sha256(path)}
                    for seed, path in sorted(paths.items())
                ]
                for arm, paths in (("base", pair[0]), ("tuned", pair[1]))
            }
            for architecture, pair in sorted(architectures.items())
        },
        "cells": cells,
        "architecture_contrasts": contrasts,
    }


def _parse_inputs(path: Path) -> dict[str, tuple[dict[int, Path], dict[int, Path]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for architecture, arms in raw.items():
        result[architecture] = tuple(
            {int(seed): Path(prediction) for seed, prediction in arm.items()}
            for arm in (arms["base"], arms["tuned"])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON architecture -> base/tuned -> seed -> prediction path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=54)
    args = parser.parse_args()
    result = analyze_sensitivity(
        _parse_inputs(args.inputs), repeats=args.repeats, bootstrap_seed=args.bootstrap_seed
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
