"""Non-inferiority margin `Δ` for the DDI parity claim (issue #12, Phase 1.4).

Ground rule 9 (issue #1): before any DDI number can be called parity or non-parity, `Δ`
must be fixed from the `full` baseline's seed-to-seed variance alone.

`compute_delta` reads the quality-eval rows the harness actually writes — the
`phase2_train_free:{model}:{dataset}:{variant}:seed=…` shape produced by
`ddi.run._quality_cell` — restricted to the `full` variant, for a given
(model, dataset, metric). It averages per seed, computes std across seed means, and
returns `Δ = k · std` with `k=2` by default. `record_delta` writes the per-seed means
and the derived Δ back into the store so downstream reports pull the whole
computation from the same store per ground rule 3.

**The `full` restriction is the ground rule, not a default worth changing.** Δ is the
noise floor of the measurement, and mixing `ddi_qf` or `blind` rows into it would fold
the very effect being tested into the margin used to judge that effect — the margin
would grow exactly when the mask matters most, which is backwards.

Historical note: this module used to filter on a `phase1_baseline:` prefix that nothing
in the harness has ever written, so `compute_delta` matched zero rows against a real
store and raised "requires >= 3 seeds" no matter how many seeds had been run. Its unit
tests constructed `phase1_baseline:` rows by hand and so agreed with the bug. Δ gates
every parity claim in VIBEQUEUE-2, which made this the quietest possible blocker.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ddi.provenance import Provenance
from ddi.runids import QualityRunId
from ddi.store import DEFAULT_STORE, append

_MIN_SEEDS = 3
_BASELINE_METRICS = ("loss", "em", "f1", "presence")

# Ground rule 9: Δ comes from the `full` baseline's seed-to-seed variance alone.
_BASELINE_VARIANT = "full"
_UNSET = object()


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
    variant: str = _BASELINE_VARIANT,
) -> Delta:
    """Compute `Δ = k · std` across per-seed means from the store.

    Filters to quality-eval rows — run_id shape `ddi.runids.QualityRunId`, the one parser
    `ddi.parity.read_means` and `ddi.aggregate`'s quality table also use — matching `model`,
    `dataset`, `metric` and `variant` (`full` by default, per ground rule 9). Groups by
    `provenance.seed`, averages each group, then computes the sample standard
    deviation (ddof=1) across those per-seed means.

    The seed is taken from `provenance.seed` rather than from the run_id's `seed=`
    field, so a row whose provenance disagrees with its id is skipped rather than
    silently counted under the wrong seed.

    Requires `n_seeds >= 3` — with fewer seeds the std is not a meaningful noise floor.
    Raises `ValueError` if the filter yields fewer than 3 seeds or if `k <= 0`.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if metric not in _BASELINE_METRICS:
        raise ValueError(
            f"metric {metric!r} is not a baseline metric; expected one of {_BASELINE_METRICS}"
        )

    by_seed: dict[int, list[float]] = {}
    checkpoint_revision: str | None | object = _UNSET
    with open(store, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = row.get("run_id")
            if not isinstance(run_id, str):
                continue
            parsed = QualityRunId.parse(run_id)
            if parsed is None:
                continue
            if (parsed.model, parsed.dataset, parsed.variant) != (model, dataset, variant):
                continue
            if row.get("metric") != metric:
                continue
            prov = row.get("provenance") or {}
            row_checkpoint = prov.get("checkpoint_revision")
            if row_checkpoint is not None and not isinstance(row_checkpoint, str):
                continue
            if checkpoint_revision is not _UNSET and checkpoint_revision != row_checkpoint:
                raise ValueError(
                    "compute_delta found multiple checkpoint identities for "
                    f"evaluation_model={model!r}"
                )
            checkpoint_revision = row_checkpoint
            seed = prov.get("seed")
            value = row.get("value")
            if seed is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            # The docstring's promise, enforced. `provenance.seed` is the seed of record,
            # so a row whose run_id names a different one is not merely mislabelled — its
            # identity is unresolvable, and counting it under either seed would move a
            # per-seed mean by an unknown amount. Δ gates every parity claim in the grid,
            # so an unresolvable row is dropped rather than guessed at. This was documented
            # here for a long time while the code only ever parsed the seed segment to
            # satisfy a length check and compared it to nothing.
            if parsed.seed != int(seed):
                continue
            by_seed.setdefault(int(seed), []).append(float(value))

    if len(by_seed) < _MIN_SEEDS:
        raise ValueError(
            f"compute_delta requires >= {_MIN_SEEDS} seeds; got {len(by_seed)} "
            f"for model={model!r} dataset={dataset!r} metric={metric!r} "
            f"variant={variant!r}"
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
