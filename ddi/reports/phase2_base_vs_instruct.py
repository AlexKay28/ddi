"""Phase 2.4 base-vs-instruct mechanism probe (issue #16).

For each `(model family, dataset)` pair, computes the DDI-QF gap
(`gap = mean(full) − mean(ddi_qf)`) on the base and instruct variants of the same family, and
reports the effect (`gap_base − gap_instruct`). Hypothesis: instruction tuning already routes
context through the query region, so `ddi_qf` costs instruct models less than base models — a
negative effect means instruct is more DDI-friendly.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

from ddi.report import filter_rows, iter_rows, register
from ddi.reports._provenance import render_provenance_footer

_PHASE2_TF_PREFIX = "phase2_train_free:"


def _parse(run_id: str) -> tuple[str, str, str] | None:
    # phase2_train_free:{model}:{dataset}:{variant_tag}:seed=...:{sha}:{record_id}
    parts = run_id.split(":")
    if len(parts) < 7 or (parts[0] + ":") != _PHASE2_TF_PREFIX:
        return None
    return parts[1], parts[2], parts[3]


def _family_and_kind(model: str) -> tuple[str, str]:
    """Strip -Instruct/-instruct to get the family; the presence of the suffix determines kind."""
    for suffix in ("-Instruct", "-instruct"):
        if model.endswith(suffix):
            return model[: -len(suffix)], "instruct"
    return model, "base"


@register("T4_probe_instruct")
def _t4_base_vs_instruct(store: Path) -> str:
    """Markdown table of (family, dataset) → gap_base, gap_instruct, effect."""
    em_rows = list(filter_rows(iter_rows(store), metric="em"))

    # (model, dataset, variant, seed) -> [per-record values]
    per_cell: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    # Only rows that survived the (phase2_train_free, full|ddi_qf) filter feed the
    # gap calculation — those are what the footer should stamp.
    backing_rows: list[dict] = []
    for row in em_rows:
        parsed = _parse(row.get("run_id", ""))
        if parsed is None:
            continue
        model, dataset, variant = parsed
        if variant not in ("full", "ddi_qf"):
            continue
        seed = int(row.get("provenance", {}).get("seed", -1))
        per_cell[(model, dataset, variant, seed)].append(float(row["value"]))
        backing_rows.append(row)

    # per-seed means, then per-(model, dataset, variant) mean and stdev across seeds
    per_seed_mean: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (model, dataset, variant, _seed), values in per_cell.items():
        per_seed_mean[(model, dataset, variant)].append(sum(values) / len(values))

    def summarise(key: tuple[str, str, str]) -> tuple[float, float] | None:
        seeds = per_seed_mean.get(key)
        if not seeds:
            return None
        mean = sum(seeds) / len(seeds)
        std = statistics.stdev(seeds) if len(seeds) >= 2 else 0.0
        return mean, std

    # Pair by (family, dataset) across base and instruct.
    families: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for (model, dataset, variant), _ in per_seed_mean.items():
        family, kind = _family_and_kind(model)
        s = summarise((model, dataset, variant))
        if s is None:
            continue
        mean, std = s
        families.setdefault((family, dataset), {}).setdefault(kind, {})[variant] = mean
        families[(family, dataset)][kind][f"{variant}_std"] = std

    lines = [
        "| family | dataset | gap_base ± σ | gap_instruct ± σ | effect (base − instruct) |",
        "|---|---|---|---|---|",
    ]
    for (family, dataset) in sorted(families):
        cell = families[(family, dataset)]
        gap_base = _gap(cell.get("base", {}))
        gap_instr = _gap(cell.get("instruct", {}))
        effect = gap_base - gap_instr if _finite(gap_base) and _finite(gap_instr) else math.nan
        lines.append(
            f"| {family} | {dataset} | {_fmt(gap_base, cell.get('base', {}).get('full_std'))} "
            f"| {_fmt(gap_instr, cell.get('instruct', {}).get('full_std'))} "
            f"| {_fmt(effect, None)} |"
        )
    return "\n".join(lines) + "\n" + render_provenance_footer(store, backing_rows)


def _gap(kind_row: dict[str, float]) -> float:
    if "full" not in kind_row or "ddi_qf" not in kind_row:
        return math.nan
    return kind_row["full"] - kind_row["ddi_qf"]


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _fmt(value: float, std: float | None) -> str:
    if not _finite(value):
        return "n/a"
    if std is None or not _finite(std):
        return f"{value:+.4f}"
    return f"{value:+.4f} ± {std:.4f}"
