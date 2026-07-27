"""Phase 2 quality summary table (T2 — referenced by paper/sections/04_results_quality.tex).

One row per (model, variant); columns for EM / F1 / presence / loss, median across seeds of the
per-record mean. Filters to `phase2_train_free:` runs; a follow-up can add a `finetuned` column
for `phase2_finetuned` rows when those exist (issue #14). Report-only; no new runner.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

from ddi.report import filter_rows, iter_rows, register

_PREFIX = "phase2_train_free:"
_METRICS = ("em", "f1", "presence", "loss")


def _parse(run_id: str) -> tuple[str, str, str, int, str] | None:
    parts = run_id.split(":")
    if len(parts) < 7 or (parts[0] + ":") != _PREFIX:
        return None
    model = parts[1]
    dataset = parts[2]
    variant_tag = parts[3]
    seed_field = parts[4]
    record_id = ":".join(parts[6:])
    if not seed_field.startswith("seed="):
        return None
    try:
        seed = int(seed_field[len("seed="):])
    except ValueError:
        return None
    return model, dataset, variant_tag, seed, record_id


@register("T2")
def _t2_quality_summary(store: Path) -> str:
    rows = list(iter_rows(store))
    if not any(r.get("run_id", "").startswith(_PREFIX) for r in rows):
        raise ValueError(
            "T2: no phase2_train_free rows in store. "
            "Run `make sweep-train-free` first."
        )

    # (model, dataset, variant, metric, seed) -> [per-record values]
    per_cell: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for metric in _METRICS:
        for row in filter_rows(rows, metric=metric):
            parsed = _parse(row.get("run_id", ""))
            if parsed is None:
                continue
            model, dataset, variant, seed, _ = parsed
            per_cell[(model, dataset, variant, metric, seed)].append(float(row["value"]))

    # (model, dataset, variant, metric) -> median across seeds of the per-seed mean
    summary: dict[tuple[str, str, str, str], float] = {}
    for (model, dataset, variant, metric, _seed), values in per_cell.items():
        key = (model, dataset, variant, metric)
        summary.setdefault(key, [])
        summary[key].append(sum(values) / len(values))
    aggregated = {
        k: (
            statistics.median(v),
            statistics.stdev(v) if len(v) >= 2 else 0.0,
            len(v),
        )
        for k, v in summary.items()
    }

    lines = [
        "| model | dataset | variant | EM | F1 | presence | loss | seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups = sorted({(m, d, v) for (m, d, v, _) in aggregated})
    for (model, dataset, variant) in groups:
        cells = []
        max_seeds = 0
        for metric in _METRICS:
            entry = aggregated.get((model, dataset, variant, metric))
            if entry is None:
                cells.append("n/a")
            else:
                median, std, n = entry
                max_seeds = max(max_seeds, n)
                if math.isfinite(std) and std > 0.0:
                    cells.append(f"{median:.4f} ± {std:.4f}")
                else:
                    cells.append(f"{median:.4f}")
        lines.append(
            f"| {model} | {dataset} | {variant} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {max_seeds} |"
        )
    return "\n".join(lines) + "\n"
