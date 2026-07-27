"""Non-inferiority margin `Δ` for the DDI parity claim (issue #12, Phase 1.4).

Ground rule 9 (issue #1): before any DDI number can be called parity or non-parity, `Δ`
must be fixed from the `full` baseline's seed-to-seed variance alone.

`compute_delta` reads phase1_full_baseline rows from the store for a given
(model, dataset, metric), averages per seed, computes std across seed means, and
returns `Δ = k · std` with `k=2` by default. `record_delta` writes the per-seed means
and the derived Δ back into the store so downstream reports pull the whole
computation from the same store per ground rule 3.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ddi.provenance import Provenance
from ddi.store import DEFAULT_STORE, append

_MIN_SEEDS = 3
_BASELINE_METRICS = ("loss", "em", "f1", "presence")


@dataclass(frozen=True)
class Delta:
    """Result of a Δ computation. Every field is a number or a mapping of numbers."""

    model: str
    dataset: str
    metric: str
    k: float
    n_seeds: int
    per_seed_means: dict[int, float]
    std: float
    delta: float


def compute_delta(
    store: Path,
    model: str,
    dataset: str,
    metric: str,
    k: float = 2.0,
) -> Delta:
    """Compute `Δ = k · std` across per-seed means from the store.

    Filters to rows whose `run_id` starts with `phase1_baseline:{model}:{dataset}:` and
    whose `metric` matches. Groups by `provenance.seed`, averages each group, then
    computes the sample standard deviation (ddof=1) across those per-seed means.

    Requires `n_seeds >= 3` — with fewer seeds the std is not a meaningful noise floor.
    Raises `ValueError` if the filter yields fewer than 3 seeds or if `k <= 0`.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if metric not in _BASELINE_METRICS:
        raise ValueError(
            f"metric {metric!r} is not a baseline metric; expected one of {_BASELINE_METRICS}"
        )

    prefix = f"phase1_baseline:{model}:{dataset}:"

    by_seed: dict[int, list[float]] = {}
    with open(store, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id.startswith(prefix):
                continue
            if row.get("metric") != metric:
                continue
            prov = row.get("provenance") or {}
            seed = prov.get("seed")
            value = row.get("value")
            if seed is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            by_seed.setdefault(int(seed), []).append(float(value))

    if len(by_seed) < _MIN_SEEDS:
        raise ValueError(
            f"compute_delta requires >= {_MIN_SEEDS} seeds; got {len(by_seed)} "
            f"for model={model!r} dataset={dataset!r} metric={metric!r}"
        )

    per_seed_means = {seed: sum(vs) / len(vs) for seed, vs in by_seed.items()}
    means = list(per_seed_means.values())
    mean_of_means = sum(means) / len(means)
    variance = sum((m - mean_of_means) ** 2 for m in means) / (len(means) - 1)
    std = math.sqrt(variance)

    return Delta(
        model=model,
        dataset=dataset,
        metric=metric,
        k=float(k),
        n_seeds=len(per_seed_means),
        per_seed_means=dict(sorted(per_seed_means.items())),
        std=std,
        delta=float(k) * std,
    )


def record_delta(
    store: Path,
    delta: Delta,
    provenance: Provenance,
) -> None:
    """Append the per-seed means and the derived Δ to the store.

    Writes one `{metric}_per_seed_mean` row per seed (run_id encodes the seed) and one
    `delta_{metric}` row for Δ itself. Every row shares the given `provenance` so the
    whole computation is auditable from the store alone (ground rule 3).
    """
    base_run_id = (
        f"phase1_delta:{delta.model}:{delta.dataset}:metric={delta.metric}:k={delta.k:g}"
    )
    per_seed_metric = f"{delta.metric}_per_seed_mean"
    for seed, mean in delta.per_seed_means.items():
        append(
            f"{base_run_id}:seed={seed}",
            per_seed_metric,
            float(mean),
            provenance,
            store=store,
        )
    append(
        base_run_id,
        f"delta_{delta.metric}",
        float(delta.delta),
        provenance,
        store=store,
    )


def format_delta(delta: Delta) -> str:
    """Render a Delta as a small human-readable table for the CLI."""
    lines = [
        f"Δ for model={delta.model} dataset={delta.dataset} metric={delta.metric}",
        f"  k        = {delta.k:g}",
        f"  n_seeds  = {delta.n_seeds}",
        f"  std      = {delta.std:.6g}",
        f"  delta    = {delta.delta:.6g}",
        "  per-seed means:",
    ]
    for seed, mean in delta.per_seed_means.items():
        lines.append(f"    seed={seed}: {mean:.6g}")
    return "\n".join(lines)


__all__ = ["DEFAULT_STORE", "Delta", "compute_delta", "format_delta", "record_delta"]
