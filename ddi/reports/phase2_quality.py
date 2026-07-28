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
from ddi.reports._provenance import render_provenance_footer

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
    # Track the exact rows the table drew from so the provenance footer stamps only
    # what actually shaped a cell — off-metric / off-prefix rows in the store are
    # ignored on read and should not be counted in the footer.
    backing_rows: list[dict] = []
    for metric in _METRICS:
        for row in filter_rows(rows, metric=metric):
            parsed = _parse(row.get("run_id", ""))
            if parsed is None:
                continue
            model, dataset, variant, seed, _ = parsed
            per_cell[(model, dataset, variant, metric, seed)].append(float(row["value"]))
            backing_rows.append(row)

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
    return "\n".join(lines) + "\n" + render_provenance_footer(store, backing_rows)


# ---- Per-metric renderers (T2_em / T2_f1 / T2_presence / T2_loss) --------------------------
#
# EXP-3 (#73) needs one table per metric with rows=mask and cols=10 fixed datasets, ready to
# paste into the template comment on #72. Ordering is fixed — do NOT alphabetise — because the
# template's story flows composition -> synthetic_qa -> hotpot -> musique -> nq -> stress
# families (fact_chain, fact_agg, contra, coref, needle). Missing cells render as em dash
# rather than n/a to keep the table visually clean when a mask/dataset was skipped.

_DATASET_COLUMNS: tuple[str, ...] = (
    "composition",
    "synthetic_qa",
    "hotpot_qa",
    "musique",
    "kilt_nq",
    "ds/fact_chain",
    "ds/fact_agg",
    "ds/contra",
    "ds/coref",
    "ds/needle",
)

# Map raw ddi_stress family names to the ds/<short> column labels.
_STRESS_FAMILY_TO_COL: dict[str, str] = {
    "fact_chain": "ds/fact_chain",
    "fact_aggregation": "ds/fact_agg",
    "contradiction": "ds/contra",
    "coreference": "ds/coref",
    "needle": "ds/needle",
}

# nq is degenerate (no docs) and is excluded from the EXP-3 templates; the store still holds
# EXP-01 nq rows so we must drop them explicitly.
_DROPPED_DATASETS: frozenset[str] = frozenset({"nq"})

_METRIC_HEADERS: dict[str, str] = {
    "em": "EM",
    "f1": "F1",
    "presence": "presence",
    "loss": "loss",
}

_EM_DASH = "—"


def _dataset_to_column(dataset: str) -> str | None:
    """Return the fixed column label for a store dataset name, or None to drop the row.

    WHY: EXP-01 wrote `ddi_stress` without a family suffix; those older rows are ambiguous
    (fact_chain default per the loader, but not recorded in run_id) so we drop them from the
    per-metric tables rather than silently misclassify.
    """
    if dataset in _DROPPED_DATASETS:
        return None
    if dataset.startswith("ddi_stress__"):
        family = dataset[len("ddi_stress__"):]
        return _STRESS_FAMILY_TO_COL.get(family)
    if dataset == "ddi_stress":
        return None
    return dataset if dataset in _DATASET_COLUMNS else None


def _fmt_metric_cell(value: float | None) -> str:
    if value is None:
        return _EM_DASH
    return f"{value:.4f}"


def _render_metric_table(store: Path, metric: str) -> str:
    """Render one metric's table: rows=mask, cols=10 fixed datasets. One block per model."""
    rows = list(iter_rows(store))
    if not any(r.get("run_id", "").startswith(_PREFIX) for r in rows):
        raise ValueError(
            f"T2_{metric}: no phase2_train_free rows in store. "
            "Run `make sweep-train-free` first."
        )

    # (model, mask, column) -> list of per-record values (across seeds)
    per_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    # Track rows that actually contributed a numeric cell (post-filter, post-parse) so
    # the footer counts match what the reader sees on the page.
    backing_rows: list[dict] = []
    for row in filter_rows(rows, metric=metric):
        parsed = _parse(row.get("run_id", ""))
        if parsed is None:
            continue
        model, dataset, variant, _seed, _rid = parsed
        col = _dataset_to_column(dataset)
        if col is None:
            continue
        try:
            per_cell[(model, variant, col)].append(float(row["value"]))
        except (TypeError, ValueError):
            continue
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"T2_{metric}: no rows for metric={metric!r} in phase2_train_free store."
        )

    models = sorted({m for (m, _v, _c) in per_cell})
    masks_seen: dict[str, list[str]] = {}
    for (model, mask, _col) in per_cell:
        masks_seen.setdefault(model, []).append(mask)
    # Preferred mask ordering (paper canon), then any others that appear alphabetically.
    _preferred = ("full", "ddi_qf", "blind")

    def _mask_order(masks: list[str]) -> list[str]:
        uniq = list(dict.fromkeys(masks))
        ordered = [m for m in _preferred if m in uniq]
        extras = sorted(m for m in uniq if m not in _preferred)
        return ordered + extras

    header_cols = " | ".join(_DATASET_COLUMNS)
    # +1 for the leading `mask` column; wrapper pipes are added at write time.
    header_sep = "|".join(["---"] * (len(_DATASET_COLUMNS) + 1))
    metric_label = _METRIC_HEADERS[metric]

    blocks: list[str] = []
    for model in models:
        lines = [
            f"### {model} — {metric_label}",
            "",
            f"| mask | {header_cols} |",
            f"|{header_sep}|",
        ]
        for mask in _mask_order(masks_seen.get(model, [])):
            cells = []
            for col in _DATASET_COLUMNS:
                values = per_cell.get((model, mask, col))
                if not values:
                    cells.append(_EM_DASH)
                else:
                    cells.append(_fmt_metric_cell(sum(values) / len(values)))
            lines.append(f"| {mask} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    # One footer covers the union of sub-tables (one `###` block per model). The store
    # + row-set are identical across blocks, so per-block stamping would only add noise.
    return "\n\n".join(blocks) + "\n" + render_provenance_footer(store, backing_rows)


@register("T2_em")
def _t2_em(store: Path) -> str:
    return _render_metric_table(store, "em")


@register("T2_f1")
def _t2_f1(store: Path) -> str:
    return _render_metric_table(store, "f1")


@register("T2_presence")
def _t2_presence(store: Path) -> str:
    return _render_metric_table(store, "presence")


@register("T2_loss")
def _t2_loss(store: Path) -> str:
    return _render_metric_table(store, "loss")
