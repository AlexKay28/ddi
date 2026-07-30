"""YAML-driven table builder — JSON output edition.

One CLI reads the ``report.tables:`` block from an experiment YAML and emits a
single JSON file: one object per entry in ``tables[]``, wrapped in an envelope
that carries the store hash, git SHA, and model filter used at collection time.

Dispatch on ``kind``:

- ``quality``       → phase2_train_free rows, ``{model: {mask: {dataset: value}}}``.
- ``timing``        → phase3_timing rows, ``{model: {L: {backend: {mask: {N: value}}}}}``.
                      ``realized`` and ``realization`` are DERIVED by pairing variant
                      rows with matching ``full`` rows at ``(model, N, L, backend, seed)``.
- ``throughput``    → phase4_throughput rows, ``{model: {mode: {N: value}}}``.
- ``restart_quality``→ phase4_restart rows, ``{model: {layout: {dataset: value}}}``.
- ``correctness``   → phase4_cache rows, ``{model: {mode: {L: {N: value}}}}``
                      (worst-case max across queries).

Design notes / why:

- **Delegates row access** to ``iter_rows`` / ``filter_rows`` so the store format
  stays owned by one module.
- **JSON output** — colorization, delta annotations, and the provenance-footer
  blockquote are gone. Consumers who need Δ vs a baseline compute it from the raw
  values.
- **Missing cells omit the key entirely** (no ``null``, no ``"—"``). Absence is
  detectable by the consumer.
- **Insertion order is meaningful** — the ordered mask/N/L axes are baked into
  dict insertion order (Python 3.7+). ``json.dumps(..., sort_keys=False)``.
- **Provenance envelope** reuses ``ddi.reports._provenance`` helpers for the git
  SHA and store hash so the format stays identical to the pre-refactor footer.
- **Open/closed for new kinds**: dispatcher ``_COLLECTORS`` is a plain dict —
  a new kind adds one entry and one ``_collect_<kind>`` function.

CLI:

    python -m ddi.aggregate <exp.yaml> --out <summary.json>
                            [--store results/runs.jsonl] [--model REPO_ID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---- Store readers (inlined from the deleted ddi.report in #89) ----------------------------
#
# The append-only JSONL store has one row per (run_id, metric) — see ddi.store.append. These
# iterators used to live in ddi/report.py; they were moved here in #89 when that module was
# deleted because ddi.aggregate is now the only in-tree consumer.


def iter_rows(store: Path) -> Iterator[dict]:
    """Yield rows from the JSONL store in file order.

    The store is append-only so file order is deterministic; every downstream
    aggregation that needs a stable tie-break relies on this ordering.
    """
    with open(store, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_rows(
    rows: Iterable[dict],
    *,
    metric: str | None = None,
    run_id: str | None = None,
    provenance: dict | None = None,
) -> Iterator[dict]:
    """Filter rows by metric, run_id, and/or a subset match on nested provenance."""
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if provenance is not None:
            prov = row.get("provenance") or {}
            if not all(prov.get(k) == v for k, v in provenance.items()):
                continue
        yield row


def _row_model(row: dict) -> str | None:
    """Extract the model from a row's run_id.

    Every producer in `ddi/run.py` writes run_ids of shape ``<prefix>:<model>:...``
    (phase2_train_free, phase3_timing, phase4_cache, phase4_throughput, phase4_restart).
    Returns ``None`` for rows whose shape doesn't match — those rows can't be filtered by
    model and will be dropped by any model-filtered load.
    """
    run_id = str(row.get("run_id", ""))
    parts = run_id.split(":", 2)  # only need up to the model field
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[1]


def load_rows(store: Path, model_filter: str | None = None) -> list[dict]:
    """Read the JSONL store into a list, optionally filtered to one model.

    Every collector calls this at the top instead of ``iter_rows(store)`` so that a
    ``--model`` CLI override applied to ``aggregate`` propagates uniformly. The store
    itself is append-only and shared across every run — the filter lets a caller who
    only cares about one model get a clean summary from a store that has many.
    """
    rows = list(iter_rows(store))
    if model_filter is None:
        return rows
    return [r for r in rows if _row_model(r) == model_filter]


# ---- Quality kind ---------------------------------------------------------------------------
#
# Dataset whitelist is lifted from the pre-JSON renderer; historical rows for pruned
# datasets (`nq`, `composition`, `ddi_stress__*`) are silently dropped rather than surfacing
# as extra keys in the JSON.

_QUALITY_PREFIX = "phase2_train_free:"
_QUALITY_METRICS: frozenset[str] = frozenset({"em", "f1", "presence", "loss"})

_QUALITY_DATASET_COLUMNS: tuple[str, ...] = (
    "synthetic_qa",
    "hotpot_qa",
    "musique",
    "kilt_nq",
)

# ``nq`` is degenerate (no docs); older EXP-01 rows still exist in the store — drop them
# explicitly. ``composition`` / ``ddi_stress__*`` also stay in the drop list: their loaders /
# runners were deleted in #89, but historical rows may still live in results/runs.jsonl and
# must be silently skipped rather than surfacing as noise keys in the JSON.
_QUALITY_DROPPED_DATASETS: frozenset[str] = frozenset({"nq"})

_QUALITY_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")


def _quality_parse_run_id(run_id: str) -> tuple[str, str, str, int] | None:
    """Return ``(model, dataset, variant, seed)`` from a phase2_train_free run_id.

    Shape: ``phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha}:{record_id}``.
    Returns None if the shape does not match — a malformed row is silently skipped so a
    single bad line in the store never breaks a collection.
    """
    parts = run_id.split(":")
    if len(parts) < 7 or (parts[0] + ":") != _QUALITY_PREFIX:
        return None
    model = parts[1]
    dataset = parts[2]
    variant = parts[3]
    seed_field = parts[4]
    if not seed_field.startswith("seed="):
        return None
    try:
        seed = int(seed_field[len("seed="):])
    except ValueError:
        return None
    return model, dataset, variant, seed


def _quality_dataset_to_column(dataset: str) -> str | None:
    """Map a store dataset name to a column label, or None to drop the row."""
    if dataset in _QUALITY_DROPPED_DATASETS:
        return None
    if dataset == "ddi_stress" or dataset.startswith("ddi_stress__") or dataset == "composition":
        return None
    return dataset if dataset in _QUALITY_DATASET_COLUMNS else None


def _quality_mask_order(masks: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(masks))
    ordered = [m for m in _QUALITY_PREFERRED_MASKS if m in uniq]
    extras = sorted(m for m in uniq if m not in _QUALITY_PREFERRED_MASKS)
    return ordered + extras


def _collect_quality(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: quality)."""
    metric = spec.get("metric")
    if metric not in _QUALITY_METRICS:
        raise ValueError(
            f"quality: unknown metric {metric!r}; expected one of "
            f"{sorted(_QUALITY_METRICS)}"
        )

    all_rows = load_rows(store, model_filter)
    if not any(str(r.get("run_id", "")).startswith(_QUALITY_PREFIX) for r in all_rows):
        raise ValueError(
            f"quality[{metric}]: no phase2_train_free rows in store {store}. "
            "Run the phase2 sweep first."
        )

    # (model, mask, column) -> [per-record values across seeds]
    per_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    backing_rows: list[dict] = []
    for row in filter_rows(all_rows, metric=metric):
        parsed = _quality_parse_run_id(str(row.get("run_id", "")))
        if parsed is None:
            continue
        model, dataset, variant, _seed = parsed
        col = _quality_dataset_to_column(dataset)
        if col is None:
            continue
        try:
            per_cell[(model, variant, col)].append(float(row["value"]))
        except (TypeError, ValueError, KeyError):
            continue
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"quality[{metric}]: no phase2_train_free rows carried metric={metric!r}."
        )

    models = sorted({m for (m, _v, _c) in per_cell})
    masks_seen: dict[str, list[str]] = defaultdict(list)
    for (model, mask, _col) in per_cell:
        masks_seen[model].append(mask)

    data: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        model_body: dict[str, dict[str, float]] = {}
        for mask in _quality_mask_order(masks_seen[model]):
            mask_body: dict[str, float] = {}
            for col in _QUALITY_DATASET_COLUMNS:
                values = per_cell.get((model, mask, col))
                if not values:
                    continue  # missing cell → key omitted entirely
                mask_body[col] = sum(values) / len(values)
            if mask_body:
                model_body[mask] = mask_body
        if model_body:
            data[model] = model_body

    return {
        "kind": "quality",
        "metric": metric,
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        "models": models,
        "data": data,
    }


# ---- Restart-quality kind (C4 restart-position layout parity) ------------------------------
#
# `sequential` is the baseline layout; `restart` is the variant. The JSON body uses the
# same dataset whitelist as ``quality``; deltas vs `sequential` are a consumer concern.

_RESTART_PREFIX = "phase4_restart:"
_RESTART_METRICS: frozenset[str] = frozenset({"em", "f1", "presence", "loss"})
_RESTART_PREFERRED_LAYOUTS: tuple[str, ...] = ("sequential", "restart")


def _restart_parse_run_id(run_id: str) -> tuple[str, str, str, str, int] | None:
    """Return ``(model, dataset, tag, layout, seed)`` from a phase4_restart run_id.

    Shape: ``phase4_restart:{model}:{dataset}:{tag}:{layout}:seed={seed}:{sha}:{record_id}``.
    Returns None if the shape does not match — a malformed row is silently skipped.
    """
    parts = run_id.split(":")
    if len(parts) < 8 or (parts[0] + ":") != _RESTART_PREFIX:
        return None
    model, dataset, tag, layout, seed_field = parts[1], parts[2], parts[3], parts[4], parts[5]
    if not seed_field.startswith("seed="):
        return None
    try:
        seed = int(seed_field[len("seed="):])
    except ValueError:
        return None
    return model, dataset, tag, layout, seed


def _restart_layout_order(layouts: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(layouts))
    ordered = [lay for lay in _RESTART_PREFERRED_LAYOUTS if lay in uniq]
    extras = sorted(lay for lay in uniq if lay not in _RESTART_PREFERRED_LAYOUTS)
    return ordered + extras


def _collect_restart_quality(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: restart_quality)."""
    metric = spec.get("metric")
    if metric not in _RESTART_METRICS:
        raise ValueError(
            f"restart_quality: unknown metric {metric!r}; expected one of "
            f"{sorted(_RESTART_METRICS)}"
        )

    all_rows = load_rows(store, model_filter)
    if not any(str(r.get("run_id", "")).startswith(_RESTART_PREFIX) for r in all_rows):
        raise ValueError(
            f"restart_quality[{metric}]: no phase4_restart rows in store {store}. "
            "Run the cache/restart_quality task first."
        )

    per_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    backing_rows: list[dict] = []
    for row in filter_rows(all_rows, metric=metric):
        parsed = _restart_parse_run_id(str(row.get("run_id", "")))
        if parsed is None:
            continue
        model, dataset, _tag, layout, _seed = parsed
        col = _quality_dataset_to_column(dataset)
        if col is None:
            continue
        try:
            per_cell[(model, layout, col)].append(float(row["value"]))
        except (TypeError, ValueError, KeyError):
            continue
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"restart_quality[{metric}]: no phase4_restart rows carried metric={metric!r}."
        )

    models = sorted({m for (m, _lay, _c) in per_cell})
    layouts_seen: dict[str, list[str]] = defaultdict(list)
    for (model, layout, _col) in per_cell:
        layouts_seen[model].append(layout)

    data: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        model_body: dict[str, dict[str, float]] = {}
        for layout in _restart_layout_order(layouts_seen[model]):
            layout_body: dict[str, float] = {}
            for col in _QUALITY_DATASET_COLUMNS:
                values = per_cell.get((model, layout, col))
                if not values:
                    continue
                layout_body[col] = sum(values) / len(values)
            if layout_body:
                model_body[layout] = layout_body
        if model_body:
            data[model] = model_body

    return {
        "kind": "restart_quality",
        "metric": metric,
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        "models": models,
        "data": data,
    }


# ---- Timing kind ---------------------------------------------------------------------------

_TIMING_PREFIX = "phase3_timing:"
_TIMING_FULL_VARIANT = "full"

_TIMING_N_COLUMNS: tuple[int, ...] = (1, 3, 5, 9, 13, 25)
_TIMING_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")

_TIMING_METRICS: frozenset[str] = frozenset(
    {"ttft", "tpot", "sparsity", "analytical", "realized", "realization"}
)

# Map short metric name from YAML → store metric name it aggregates. `realized` and
# `realization` are DERIVED so they need TTFT rows on both the variant and `full` arms.
_TIMING_METRIC_ALIAS: dict[str, str] = {
    "ttft": "ftt_median_ms",
    "tpot": "tpot_median_ms",
    "sparsity": "mask_sparsity",
    "analytical": "analytical_speedup",
    "realized": "ftt_median_ms",  # derived from full vs variant TTFT
    "realization": "ftt_median_ms",  # derived from realized / analytical
}


def _timing_parse_run_id(run_id: str) -> dict[str, Any] | None:
    """Parse a phase3_timing run_id into a dict of shape parameters.

    Shape: ``phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant}:{backend}:seed={seed}:{sha}``.
    """
    if not run_id.startswith(_TIMING_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 9:
        return None
    n_seg, l_seg, q_seg = parts[2], parts[3], parts[4]
    if not (n_seg.startswith("N=") and l_seg.startswith("L=") and q_seg.startswith("Q=")):
        return None
    try:
        n = int(n_seg[2:])
        l_ = int(l_seg[2:])
        q = int(q_seg[2:])
    except ValueError:
        return None
    return {
        "model": parts[1],
        "N": n,
        "L": l_,
        "Q": q,
        "variant": parts[5],
        "backend": parts[6],
    }


def _timing_mask_order(masks: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(masks))
    ordered = [m for m in _TIMING_PREFERRED_MASKS if m in uniq]
    extras = sorted(m for m in uniq if m not in _TIMING_PREFERRED_MASKS)
    return ordered + extras


def _timing_collect(
    store: Path,
    model_filter: str | None = None,
) -> tuple[
    dict[tuple[str, str, int, int, str, int], dict[str, float]],
    list[dict],
]:
    """Group all phase3_timing rows by (model, variant, N, L, backend, seed).

    Returns ``(per_cell, backing_rows)``. ``per_cell[key][short]`` is the raw metric value;
    the derived ``realized`` / ``realization`` are filled by ``_timing_apply_derived``.
    """
    per_cell: dict[tuple[str, str, int, int, str, int], dict[str, float]] = {}
    backing_rows: list[dict] = []
    for row in load_rows(store, model_filter):
        parsed = _timing_parse_run_id(str(row.get("run_id", "")))
        if parsed is None:
            continue
        metric = row.get("metric")
        if metric == "ftt_median_ms":
            short = "ttft"
        elif metric == "tpot_median_ms":
            short = "tpot"
        elif metric == "mask_sparsity":
            short = "sparsity"
        elif metric == "analytical_speedup":
            short = "analytical"
        else:
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        seed = int((row.get("provenance") or {}).get("seed", 0))
        key = (
            str(parsed["model"]),
            str(parsed["variant"]),
            int(parsed["N"]),
            int(parsed["L"]),
            str(parsed["backend"]),
            seed,
        )
        per_cell.setdefault(key, {})[short] = value
        backing_rows.append(row)
    return per_cell, backing_rows


def _timing_apply_derived(
    per_cell: dict[tuple[str, str, int, int, str, int], dict[str, float]],
) -> None:
    """Fill ``realized`` / ``realization`` in-place by pairing variant rows with same-seed full.

    Per-seed pairing keeps hardware constant (kernel timing noise varies run-to-run).
    """
    full_ttft: dict[tuple[str, int, int, str, int], float] = {}
    for (model, variant, n, l_, backend, seed), metrics in per_cell.items():
        if variant == _TIMING_FULL_VARIANT and "ttft" in metrics:
            full_ttft[(model, n, l_, backend, seed)] = metrics["ttft"]

    for (model, variant, n, l_, backend, seed), metrics in per_cell.items():
        if variant == _TIMING_FULL_VARIANT:
            # By definition full realizes exactly 1.0× and its realization ratio is 1.0.
            metrics["realized"] = 1.0
            metrics["realization"] = 1.0
            metrics.setdefault("sparsity", 0.0)
            metrics.setdefault("analytical", 1.0)
            continue
        base = full_ttft.get((model, n, l_, backend, seed))
        variant_ttft = metrics.get("ttft")
        if base is not None and variant_ttft is not None and variant_ttft > 0:
            realized = base / variant_ttft
            metrics["realized"] = realized
            analytical = metrics.get("analytical")
            if analytical is not None and analytical > 0:
                metrics["realization"] = realized / analytical


def _collect_timing(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: timing)."""
    metric = spec.get("metric")
    if metric not in _TIMING_METRICS:
        raise ValueError(
            f"timing: unknown metric {metric!r}; expected one of "
            f"{sorted(_TIMING_METRICS)}"
        )

    if not any(
        str(r.get("run_id", "")).startswith(_TIMING_PREFIX)
        for r in load_rows(store, model_filter)
    ):
        raise ValueError(
            f"timing[{metric}]: no phase3_timing rows in store {store}. "
            "Run the phase3 sweep first."
        )

    per_cell, backing_rows = _timing_collect(store, model_filter)
    _timing_apply_derived(per_cell)

    # Aggregate values per (model, L, backend, mask, N) by mean across seeds. Track masks
    # separately so a variant with only derived metrics missing still shows up (with an
    # empty-inner dict pruned rather than surfaced as null).
    agg: dict[tuple[str, int, str, str, int], list[float]] = defaultdict(list)
    all_masks: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for (model, variant, n, l_, backend, _seed), metrics in per_cell.items():
        if n not in _TIMING_N_COLUMNS:
            continue
        all_masks[(model, l_, backend)].add(variant)
        val = metrics.get(metric)
        if val is None:
            continue
        agg[(model, l_, backend, variant, n)].append(val)

    # Build the nested dict: {model: {L: {backend: {mask: {N: value}}}}}.
    # Outer axes sorted ascendingly; masks follow _timing_mask_order; N follows the
    # fixed column order.
    triples = sorted(all_masks)
    data: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for (model, l_, backend) in triples:
        model_body = data.setdefault(model, {})
        l_body = model_body.setdefault(str(l_), {})
        backend_body = l_body.setdefault(backend, {})
        for mask in _timing_mask_order(all_masks[(model, l_, backend)]):
            mask_body: dict[str, float] = {}
            for n in _TIMING_N_COLUMNS:
                vals = agg.get((model, l_, backend, mask, n))
                if not vals:
                    continue
                mask_body[str(n)] = sum(vals) / len(vals)
            if mask_body:
                backend_body[mask] = mask_body
        if not backend_body:
            l_body.pop(backend, None)
        if not l_body:
            model_body.pop(str(l_), None)
    # Drop any model that ended up empty after pruning.
    data = {m: body for m, body in data.items() if body}

    return {
        "kind": "timing",
        "metric": metric,
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        "models": sorted(data),
        "data": data,
    }


# ---- Throughput kind ------------------------------------------------------------------------

_THROUGHPUT_PREFIX = "phase4_throughput:"
_THROUGHPUT_PREFERRED_MODES: tuple[str, ...] = ("ddi_cache", "ddi_recompute", "full_causal")


def _throughput_parse_run_id(run_id: str) -> tuple[str, str, str] | None:
    """Return ``(model, mode, N_str_or_"n/a")`` from a phase4_throughput run_id.

    Shape: ``phase4_throughput:{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha}``
    or with an optional ``N={N}`` segment inserted before ``reuse=``. A run without ``N=``
    maps to the sentinel N key ``"n/a"``.
    """
    if not run_id.startswith(_THROUGHPUT_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 7:
        return None
    model = parts[1]
    mode = parts[2]
    n_str: str | None = None
    for seg in parts[3:]:
        if seg.startswith("N="):
            n_str = seg[2:]
            break
    return model, mode, (n_str if n_str is not None else "n/a")


def _throughput_mode_order(modes: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(modes))
    ordered = [m for m in _THROUGHPUT_PREFERRED_MODES if m in uniq]
    extras = sorted(m for m in uniq if m not in _THROUGHPUT_PREFERRED_MODES)
    return ordered + extras


def _collect_throughput(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: throughput)."""
    metric = spec.get("metric") or "qps"

    all_rows = load_rows(store, model_filter)
    if not any(str(r.get("run_id", "")).startswith(_THROUGHPUT_PREFIX) for r in all_rows):
        raise ValueError(
            f"throughput[{metric}]: no phase4_throughput rows in store {store}. "
            "Run the phase4 sweep first."
        )

    per_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    all_ns_per_model: dict[str, set[str]] = defaultdict(set)
    all_modes_per_model: dict[str, set[str]] = defaultdict(set)
    backing_rows: list[dict] = []
    for row in filter_rows(all_rows, metric=metric):
        parsed = _throughput_parse_run_id(str(row.get("run_id", "")))
        if parsed is None:
            continue
        model, mode, n_str = parsed
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        per_cell[(model, mode, n_str)].append(val)
        all_ns_per_model[model].add(n_str)
        all_modes_per_model[model].add(mode)
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"throughput[{metric}]: no phase4_throughput rows carried metric={metric!r}."
        )

    # Sort N keys numerically when possible, else lexicographic (keeps the "n/a" sentinel
    # at the end deterministically).
    def _n_sort_key(n_str: str) -> tuple[int, int | str]:
        try:
            return (0, int(n_str))
        except ValueError:
            return (1, n_str)

    models = sorted(all_modes_per_model)
    data: dict[str, dict[str, dict[str, float]]] = {}
    for model in models:
        model_body: dict[str, dict[str, float]] = {}
        n_keys = sorted(all_ns_per_model[model], key=_n_sort_key)
        for mode in _throughput_mode_order(all_modes_per_model[model]):
            mode_body: dict[str, float] = {}
            for n in n_keys:
                vals = per_cell.get((model, mode, n))
                if not vals:
                    continue
                mode_body[n] = sum(vals) / len(vals)
            if mode_body:
                model_body[mode] = mode_body
        if model_body:
            data[model] = model_body

    return {
        "kind": "throughput",
        "metric": metric,
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        "models": models,
        "data": data,
    }


# ---- Correctness kind ----------------------------------------------------------------------
#
# Cell = ``max(max_abs_logit_diff)`` across the ``n_queries_per_doc`` queries at each
# (model, mode, N, L). ``max`` (not mean) is the worst-case gate — a single bad query
# would otherwise be washed out.

_CORRECTNESS_PREFIX = "phase4_cache:"
_CORRECTNESS_BASELINE_MODE = "full_causal"


def _correctness_parse_run_id(run_id: str) -> tuple[str, str, int, int, int] | None:
    """Return ``(model, mode, N, L, query_i)`` from a phase4_cache correctness run_id.

    Shape: ``phase4_cache:{model}:{mode}:N={N}:L={L}:seed={seed}:{sha}:q{query_i}``.
    Returns None if the shape doesn't match — malformed rows are silently skipped.
    """
    if not run_id.startswith(_CORRECTNESS_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 8:
        return None
    model, mode = parts[1], parts[2]
    n_seg, l_seg = parts[3], parts[4]
    if not (n_seg.startswith("N=") and l_seg.startswith("L=")):
        return None
    try:
        n = int(n_seg[2:])
        l_ = int(l_seg[2:])
    except ValueError:
        return None
    q_seg = parts[-1]
    if not q_seg.startswith("q"):
        return None
    try:
        query_i = int(q_seg[1:])
    except ValueError:
        return None
    return model, mode, n, l_, query_i


def _correctness_mode_order(modes: Iterable[str]) -> list[str]:
    """Baseline (``full_causal``) first, then remaining modes alphabetically."""
    uniq = sorted(set(modes))
    ordered: list[str] = []
    if _CORRECTNESS_BASELINE_MODE in uniq:
        ordered.append(_CORRECTNESS_BASELINE_MODE)
    for m in uniq:
        if m != _CORRECTNESS_BASELINE_MODE:
            ordered.append(m)
    return ordered


def _collect_correctness(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: correctness)."""
    metric = spec.get("metric") or "max_abs_logit_diff"

    all_rows = load_rows(store, model_filter)
    if not any(str(r.get("run_id", "")).startswith(_CORRECTNESS_PREFIX) for r in all_rows):
        raise ValueError(
            f"correctness[{metric}]: no phase4_cache rows in store {store}. "
            "Run the cache/correctness task first."
        )

    per_cell: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    all_ls_per_model: dict[str, set[int]] = defaultdict(set)
    all_ns_per_ml: dict[tuple[str, int], set[int]] = defaultdict(set)
    all_modes_per_ml: dict[tuple[str, int], set[str]] = defaultdict(set)
    backing_rows: list[dict] = []
    for row in filter_rows(all_rows, metric=metric):
        parsed = _correctness_parse_run_id(str(row.get("run_id", "")))
        if parsed is None:
            continue
        model, mode, n, l_, _q = parsed
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        per_cell[(model, mode, l_, n)].append(value)
        all_ls_per_model[model].add(l_)
        all_ns_per_ml[(model, l_)].add(n)
        all_modes_per_ml[(model, l_)].add(mode)
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"correctness[{metric}]: no phase4_cache rows carried metric={metric!r}."
        )

    models = sorted(all_ls_per_model)
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model in models:
        model_body: dict[str, dict[str, dict[str, float]]] = {}
        for l_ in sorted(all_ls_per_model[model]):
            ns = sorted(all_ns_per_ml[(model, l_)])
            modes = _correctness_mode_order(all_modes_per_ml[(model, l_)])
            # {mode: {L: {N: value}}} — mode is the outer axis under model per the spec.
            for mode in modes:
                mode_body = model_body.setdefault(mode, {})
                l_body: dict[str, float] = {}
                for n in ns:
                    vals = per_cell.get((model, mode, l_, n))
                    if not vals:
                        continue
                    # Worst-case gate: max, not mean.
                    l_body[str(n)] = max(vals)
                if l_body:
                    mode_body[str(l_)] = l_body
            # Prune modes that ended up with no L blocks (shouldn't happen given the loop
            # structure, but keeps the missing-cell contract clean).
            model_body = {mode: body for mode, body in model_body.items() if body}
        if model_body:
            data[model] = model_body

    return {
        "kind": "correctness",
        "metric": metric,
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        "models": models,
        "data": data,
    }


# ---- Dispatcher + public API ---------------------------------------------------------------

_COLLECTORS: dict[str, Callable[[dict, Path, str | None], dict]] = {
    "quality": _collect_quality,
    "timing": _collect_timing,
    "throughput": _collect_throughput,
    "restart_quality": _collect_restart_quality,
    "correctness": _collect_correctness,
}


def _log(msg: str) -> None:
    """Print a progress line to stderr with the [ddi.aggregate] prefix (flushed)."""
    print(f"[ddi.aggregate] {msg}", file=sys.stderr, flush=True)


def collect_table(spec: dict, store: Path, model_filter: str | None = None) -> dict:
    """Collect one ``report.tables[]`` entry into a JSON-serializable dict.

    ``spec`` keys:
      - ``kind``: ``quality`` | ``timing`` | ``throughput`` | ``restart_quality`` |
        ``correctness``.
      - ``rows`` / ``cols``: axis names (informational; the emitted shape is fixed per kind).
      - ``metric``: which metric to render.

    ``model_filter`` scopes every collector to a single model — the store is shared and
    append-only, so a caller that just ran one model wants to see only that model in the
    summary.
    """
    kind = spec.get("kind")
    if kind not in _COLLECTORS:
        raise ValueError(
            f"unknown table kind {kind!r}; expected one of {sorted(_COLLECTORS)}"
        )
    return _COLLECTORS[kind](spec, store, model_filter)


# ---- Provenance envelope helpers -----------------------------------------------------------
#
# Stdlib-only replicas of the (git SHA, store-hash) helpers that used to live in
# ddi/reports/_provenance.py; kept local rather than imported so this module has no
# markdown-formatting dependency at all. Behaviour matches:
#   - git SHA falls back to "not-a-git-repo" when git is missing / not a repo
#   - store hash is the first 8 hex chars of sha256(store bytes), or "missing" when absent


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-repo"
    sha = result.stdout.strip()
    return sha if sha else "not-a-git-repo"


def _store_sha256_8(store: Path) -> str:
    try:
        data = Path(store).read_bytes()
    except (FileNotFoundError, OSError):
        return "missing"
    return hashlib.sha256(data).hexdigest()[:8]


def _generated_utc() -> str:
    """UTC timestamp in ISO8601 with a ``Z`` suffix (seconds precision)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if ts.endswith("+00:00"):
        ts = ts[: -len("+00:00")] + "Z"
    return ts


def _envelope(
    exp_yaml: Path,
    store: Path,
    model_filter: str | None,
    tables: list[dict],
) -> dict:
    """Build the top-level JSON envelope stamped with git + store + timestamp."""
    return {
        "yaml": str(exp_yaml),
        "generated_utc": _generated_utc(),
        "git_sha": _git_sha(),
        "store": str(store),
        "store_sha256_8": _store_sha256_8(store),
        "model_filter": model_filter,
        "tables": tables,
    }


def aggregate(
    exp_yaml: Path,
    out: Path,
    store: Path = Path("results/runs.jsonl"),
    model_filter: str | None = None,
) -> None:
    """Read ``report.tables:`` from ``exp_yaml``, emit each table into ``out`` as JSON.

    ``out`` is fully overwritten each call. Multi-run aggregation is a post-processing
    concern — the top-level envelope carries ``generated_utc`` / ``store_sha256_8`` /
    ``git_sha`` that would be ambiguous if multiple runs were merged into one file.

    ``model_filter`` scopes the collected tables to rows whose ``run_id`` names that model.
    The store itself is left untouched.
    """
    exp_yaml = Path(exp_yaml)
    out = Path(out)
    store = Path(store)

    with exp_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    report_block = cfg.get("report") or {}
    tables_spec = report_block.get("tables") or []
    if not isinstance(tables_spec, list):
        raise ValueError(
            f"{exp_yaml}: report.tables must be a list, got {type(tables_spec).__name__}"
        )
    if not tables_spec:
        raise ValueError(f"{exp_yaml}: report.tables is empty — nothing to render.")

    scope = f" (model={model_filter})" if model_filter else ""
    _log(
        f"loading YAML {exp_yaml}; {len(tables_spec)} table(s) to collect from store "
        f"{store}{scope}"
    )
    t0 = time.perf_counter()
    tables: list[dict] = []
    for i, spec in enumerate(tables_spec, 1):
        kind = spec.get("kind", "?")
        metric = spec.get("metric", "?")
        t = time.perf_counter()
        try:
            tables.append(collect_table(spec, store, model_filter))
            _log(
                f"  table {i}/{len(tables_spec)}: kind={kind} metric={metric} → OK "
                f"({time.perf_counter() - t:.2f}s)"
            )
        except Exception as exc:
            _log(
                f"  table {i}/{len(tables_spec)}: kind={kind} metric={metric} → "
                f"FAILED: {exc!r}"
            )
            raise

    envelope = _envelope(exp_yaml, store, model_filter, tables)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        # Insertion order is meaningful (ordered mask/N/L axes), so sort_keys=False.
        json.dump(envelope, f, indent=2, sort_keys=False)
        f.write("\n")
    _log(
        f"wrote {len(tables)} table(s) to {out} "
        f"({time.perf_counter() - t0:.2f}s total)"
    )


# ---- CLI -----------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ddi.aggregate",
        description=(
            "Read report.tables from an experiment YAML and emit a JSON summary "
            "(one object per table, wrapped in an envelope stamped with git SHA, "
            "store hash, and model filter)."
        ),
    )
    p.add_argument("exp_yaml", type=Path, help="Path to experiment YAML.")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the composed summary.json.",
    )
    p.add_argument(
        "--store",
        type=Path,
        default=Path("results/runs.jsonl"),
        help="Path to the JSONL store (default: results/runs.jsonl).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Filter collected rows to a single model (HF repo id, matches what --model of "
            "ddi.run wrote). Store is unchanged."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    aggregate(
        args.exp_yaml,
        args.out,
        store=args.store,
        model_filter=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
