"""Compare normalized transfer closure across two validated architecture results."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    repeats: int = 10_000,
    bootstrap_seed: int = 46,
    practical_threshold: float = 0.10,
) -> dict[str, Any]:
    left_cells = {(cell["dataset"], cell["metric"]): cell for cell in left["cells"]}
    right_cells = {(cell["dataset"], cell["metric"]): cell for cell in right["cells"]}
    if left_cells.keys() != right_cells.keys():
        raise ValueError("architecture result cells differ")

    rng = random.Random(bootstrap_seed)
    cells = []
    for key in sorted(left_cells):
        left_cell = left_cells[key]
        right_cell = right_cells[key]
        left_closure = left_cell["aggregate"]["normalized_closure"]
        right_closure = right_cell["aggregate"]["normalized_closure"]
        valid = bool(
            left_cell["admissible"]
            and right_cell["admissible"]
            and left_cell["blind_stability_pass"]
            and right_cell["blind_stability_pass"]
            and left_closure is not None
            and right_closure is not None
        )
        if left_closure is None or right_closure is None:
            differences = []
        else:
            left_values = left_closure["seed_values"]
            right_values = right_closure["seed_values"]
            if len(left_values) != len(right_values):
                raise ValueError(f"{key}: architecture seed counts differ")
            differences = [
                right_value - left_value
                for left_value, right_value in zip(left_values, right_values, strict=True)
            ]

        if differences:
            draws = [
                statistics.mean(rng.choices(differences, k=len(differences)))
                for _ in range(repeats)
            ]
            mean = statistics.mean(differences)
            interval = [_percentile(draws, 0.025), _percentile(draws, 0.975)]
        else:
            mean = None
            interval = None
        distinguished = bool(
            valid
            and mean is not None
            and interval is not None
            and abs(mean) >= practical_threshold
            and (interval[0] > 0.0 or interval[1] < 0.0)
        )
        if not valid:
            verdict = "UNAVAILABLE"
        elif distinguished:
            verdict = "PRACTICALLY_DISTINGUISHED"
        else:
            verdict = "NO_PRACTICAL_DIFFERENCE_ESTABLISHED"
        cells.append(
            {
                "dataset": key[0],
                "metric": key[1],
                "valid": valid,
                "difference_direction": f"{right_name} - {left_name}",
                "seed_values": differences,
                "mean": mean,
                "ci95": interval,
                "practical_threshold": practical_threshold,
                "practically_distinguished": distinguished,
                "verdict": verdict,
                "left_verdict": left_cell["verdict"],
                "right_verdict": right_cell["verdict"],
            }
        )
    return {
        "comparison": f"{right_name} - {left_name}",
        "bootstrap": {"repeats": repeats, "seed": bootstrap_seed},
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=46)
    args = parser.parse_args()
    result = compare(
        json.loads(args.left.read_text()),
        json.loads(args.right.read_text()),
        left_name=args.left_name,
        right_name=args.right_name,
        repeats=args.repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["inputs"] = {
        "left": {"path": str(args.left), "sha256": _sha256(args.left)},
        "right": {"path": str(args.right), "sha256": _sha256(args.right)},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
