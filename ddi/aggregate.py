"""YAML-driven table builder — JSON output edition.

One CLI reads the ``report.tables:`` block from an experiment YAML and emits a
single JSON file: one object per entry in ``tables[]``, wrapped in an envelope
that carries the store hash, git SHA, and model filter used at collection time.

Dispatch on ``kind``:

- ``quality``       → phase2_train_free rows, ``{model: {mask: {dataset: value}}}``.
- ``timing``        → phase3_timing rows, ``{model: {L: {backend: {mask: {N: value}}}}}``.
                      ``realized`` and ``realization`` are DERIVED by pairing variant
                      rows with matching ``full`` rows at ``(model, N, L, backend, seed)``.
- ``throughput``    → phase4_throughput rows, ``{model: {mode: {arm: value}}}`` where
                      ``arm`` is ``{layout}/reuse={reuse}`` — the only axis the
                      phase4_throughput run_id actually carries besides the mode.
- ``restart_quality``→ phase4_restart rows, ``{model: {layout: {dataset: value}}}``.
- ``correctness``   → phase4_cache rows, ``{model: {mode: {L: {N: value}}}}``
                      (worst-case max across queries).

Design notes / why:

- **Delegates row access** to ``iter_rows`` / ``filter_rows`` so the store format
  stays owned by one module, and the ``run_id`` grammar to ``ddi.runids`` so the shape
  of a row's identity is owned by one module too. No prefix string or segment index
  appears here.
- **One driver, five kinds.** ``quality`` / ``restart_quality`` / ``throughput`` /
  ``correctness`` / ``timing`` are the same parse → bucket → reduce → nest pipeline;
  ``_collect`` runs it and each kind supplies only what differs (see ``_Axis``).
- **The store is read once per YAML**, in ``aggregate()``, and the row list is passed to
  every collector. Each collector used to call ``load_rows`` itself and ``timing`` called
  it twice, so ``performance.yaml``'s six tables parsed the whole JSONL seven times.
- **JSON output** — colorization, delta annotations, and the provenance-footer
  blockquote are gone. Consumers who need Δ vs a baseline compute it from the raw
  values.
- **Missing cells omit the key entirely** (no ``null``, no ``"—"``). Absence is
  detectable by the consumer.
- **Insertion order is meaningful** — the ordered mask/N/L axes are baked into
  dict insertion order (Python 3.7+). ``json.dumps(..., sort_keys=False)``.
- **Every axis is derived from the rows present**, ordered by a per-axis
  "preferred" tuple with anything unexpected appended (see ``_ordered_axis``). A
  hardcoded axis whitelist silently empties the table the moment a config names a
  value outside it, and the "no rows in store" guards only check the run_id prefix.
- **Provenance envelope** does NOT import ``ddi.reports._provenance``; it carries
  stdlib-only replicas of those helpers (see ``_git_sha`` / ``_store_sha256_8``)
  so this module has no markdown-formatting dependency at all. ``ddi/reports/``
  is not a package and is not shipped in ``pyproject.toml``.
- **Open/closed for new kinds**: dispatcher ``_COLLECTORS`` is a plain dict —
  a new kind adds one entry and one ``_collect_<kind>`` function, plus its allowed
  ``report.tables[]`` keys in ``_TABLE_KEYS``.

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

from ddi.runids import (
    CorrectnessRunId,
    FinetuneRunId,
    QualityRunId,
    RestartRunId,
    ThroughputRunId,
    TimingRunId,
)

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


# ---- Axis ordering --------------------------------------------------------------------------


def _ordered_axis(values: Iterable[Any], preferred: tuple[Any, ...]) -> list[Any]:
    """Order an axis: ``preferred`` entries first in their declared order, then the rest.

    The axis itself is always DERIVED from ``values`` — the rows actually present — so a
    value nobody anticipated still gets a column instead of vanishing. ``preferred`` only
    fixes the reading order of the values it names; extras are appended sorted so the
    output stays deterministic.

    This shape used to be written out five times (masks, layouts, timing masks, throughput
    modes, correctness modes) and twice more as a hardcoded tuple that was *filtered
    against* rather than merely sorted by — ``_TIMING_N_COLUMNS`` and the old
    ``_QUALITY_DATASET_COLUMNS``. Those two dropped every row whose N / dataset was not in
    the tuple, so changing ``Ns:`` in performance.yaml emptied the timing table with no
    error anywhere (the "no rows in store" guard only checks the run_id prefix, which was
    still matching). One helper, one behaviour, one place to fix it.
    """
    uniq = list(dict.fromkeys(values))
    ordered = [v for v in preferred if v in uniq]
    extras = sorted(v for v in uniq if v not in preferred)
    return ordered + extras


# ---- The collector driver -------------------------------------------------------------------
#
# `quality`, `restart_quality`, `throughput` and `correctness` were four copies of the same
# body: guard that the store holds rows of this prefix → parse each row's run_id into a cell
# key → bucket values into a defaultdict → reduce each bucket → nest into ordered axes,
# pruning empties → wrap in the shared envelope. `timing` was the same with a derived-metric
# pass spliced between bucketing and reducing. Five copies of a five-phase pipeline is five
# places for the axis-ordering and missing-cell rules to drift apart, and two of them already
# had (see `_ordered_axis`'s docstring for the hardcoded-axis incident).
#
# What varies between kinds is exactly: which rows are mine (`prefix`), how a row becomes a
# cell key (`bucket`), how a bucket becomes a number (`reduce` — `max` for correctness, mean
# for the rest), and how the key's components are ordered and nested (`axes` / `nest_order`).
# Everything else is shared.


# One level of a collected table, expressed as its ordering rule: given every value seen at
# that level *under the parent cell being expanded* — never a hardcoded whitelist — return
# them in reading order. Cell values become JSON object keys via `str`, which is what turns
# the int N / L axes into `"13"` / `"8"`.
_Axis = Callable[[Iterable[Any]], list[Any]]


def _sorted_axis(values: Iterable[Any]) -> list[Any]:
    """Ascending, deduplicated. The default for axes with no preferred reading order."""
    return sorted(set(values))


def _preferred_axis(preferred: tuple[Any, ...]) -> _Axis:
    """An axis ordered by ``preferred`` first, then anything else sorted (`_ordered_axis`)."""
    return lambda values: _ordered_axis(values, preferred)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _bucket_by_metric(
    rows: Iterable[dict],
    metric: str,
    key_of: Callable[[str], tuple | None],
) -> tuple[dict[tuple, list[float]], list[dict]]:
    """Group ``metric`` rows into ``{cell key: [values]}`` plus the rows that backed them.

    A row is dropped — silently, so one bad line in an append-only store never breaks a
    whole collection — when its run_id does not parse into a cell key, or when its value is
    not a number.
    """
    per_cell: dict[tuple, list[float]] = defaultdict(list)
    backing_rows: list[dict] = []
    for row in filter_rows(rows, metric=metric):
        key = key_of(str(row.get("run_id", "")))
        if key is None:
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        per_cell[key].append(value)
        backing_rows.append(row)
    return per_cell, backing_rows


def _nest(
    per_cell: dict[tuple, list[float]],
    reduce: Callable[[list[float]], float],
    axes: tuple[_Axis, ...],
    nest_order: tuple[int, ...] | None = None,
) -> dict:
    """Reduce each cell and nest it under its ordered axes.

    ``axes`` is in *traversal* order and each axis's values are collected per parent cell, so
    a value nobody anticipated still gets a key instead of vanishing. ``nest_order`` permutes
    the key components on the way into the output when the reading order and the nesting
    order differ (only ``correctness`` needs this: it orders modes within each L but nests L
    inside mode).

    Only cells that exist are walked, so a missing cell omits its key entirely and no empty
    container is ever created — "absence is detectable by the consumer" without any pruning
    pass.
    """
    order = tuple(range(len(axes))) if nest_order is None else nest_order

    # scope[(v0, ..., v(d-1))] = the values seen at depth d under that parent, in row order.
    scope: dict[tuple, list[Any]] = defaultdict(list)
    for key in per_cell:
        for depth in range(len(axes)):
            scope[key[:depth]].append(key[depth])

    ordered_keys: list[tuple] = []

    def walk(parent: tuple, depth: int) -> None:
        if depth == len(axes):
            ordered_keys.append(parent)
            return
        for value in axes[depth](scope[parent]):
            walk((*parent, value), depth + 1)

    walk((), 0)

    data: dict = {}
    for key in ordered_keys:
        node = data
        for i in order[:-1]:
            node = node.setdefault(str(key[i]), {})
        node[str(key[order[-1]])] = reduce(per_cell[key])
    return data


def _table(kind: str, metric: str | None, backing_rows: list[dict], data: dict) -> dict:
    """The envelope every collected table shares.

    ``seeds`` comes from ``provenance.seed`` rather than the run_id's ``seed=`` field: the
    provenance is the seed of record (see the cross-check in `ddi.delta.compute_delta`).
    """
    body = {"kind": kind}
    if metric is not None:
        body["metric"] = metric
    body.update(
        backing_rows=len(backing_rows),
        distinct_run_ids=len({r.get("run_id") for r in backing_rows}),
        seeds=sorted({int((r.get("provenance") or {}).get("seed", 0)) for r in backing_rows}),
        models=sorted(data),
        data=data,
    )
    return body


def _collect(
    *,
    kind: str,
    metric: str,
    rows: list[dict],
    store: Path,
    prefix: str,
    no_rows_hint: str,
    bucket: Callable[[list[dict]], tuple[dict[tuple, list[float]], list[dict]]],
    reduce: Callable[[list[float]], float],
    axes: tuple[_Axis, ...],
    nest_order: tuple[int, ...] | None = None,
    require_cells: bool = True,
) -> dict:
    """Run the five-phase collection for one ``report.tables[]`` entry.

    The two failure modes are deliberately distinct. "No rows of this prefix at all" means
    the sweep was never run and names the task to run (``no_rows_hint``). "Rows exist but
    none carried this metric" means the sweep ran and the metric selector is wrong. Both
    raise rather than emitting an empty table, because an empty table renders identically to
    a real result of zero. ``require_cells=False`` opts out of the second one for the single
    kind (``timing``) whose metrics can legitimately be absent from every cell.
    """
    if not any(str(r.get("run_id", "")).startswith(prefix) for r in rows):
        raise ValueError(
            f"{kind}[{metric}]: no {prefix.rstrip(':')} rows in store {store}. {no_rows_hint}"
        )
    per_cell, backing_rows = bucket(rows)
    if require_cells and not per_cell:
        raise ValueError(
            f"{kind}[{metric}]: no {prefix.rstrip(':')} rows carried metric={metric!r}."
        )
    return _table(kind, metric, backing_rows, _nest(per_cell, reduce, axes, nest_order))


# ---- Quality kind ---------------------------------------------------------------------------
#
# The dataset axis is derived from the rows present and ordered by
# `_QUALITY_PREFERRED_DATASETS`. Historical rows for pruned datasets (`nq`, `kilt_nq`,
# `composition`, `ddi_stress__*`) are silently dropped rather than surfacing as extra
# keys in the JSON.

_QUALITY_METRICS: frozenset[str] = frozenset({"em", "f1", "presence", "loss"})

# Reading order only — NOT a filter. The datasets `experiments/quality.yaml` runs, first.
_QUALITY_PREFERRED_DATASETS: tuple[str, ...] = (
    "synthetic_qa",
    "hotpot_qa",
    "musique",
)

# Datasets whose rows must never reach a table, however many of them a store holds.
#
# ``nq`` is degenerate (no docs), so every DDI mask variant coerces to `full` and the cell
# measures nothing; its loader was deleted in VIBEQUEUE-16 but older EXP-01 rows still
# exist in the store. ``kilt_nq`` is worse than degenerate: its "documents" were bare
# Wikipedia titles rather than passage bodies (VIBEQUEUE-2 review, issue R1), so `full`
# and `blind` saw the same information and the whole column is void — and a store carrying
# those rows has already been reported once (see `nirvana/common/constant.py` on rows
# 22/23, which averaged 41,304 stale rows into a published table). ``composition`` /
# ``ddi_stress__*`` are here because their loaders / runners were deleted in #89.
#
# This list is load-bearing, not belt-and-braces: the dataset axis is derived from the
# rows present, so anything not dropped HERE becomes a column. Restoring `kilt_nq` means
# doing the KILT knowledge-source join first, then removing it from this set.
_QUALITY_DROPPED_DATASETS: frozenset[str] = frozenset({"nq", "kilt_nq"})

_QUALITY_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")


def _quality_dataset_to_column(dataset: str) -> str | None:
    """Map a store dataset name to a column label, or None to drop the row.

    Anything not explicitly dropped becomes a column: the axis is derived from the rows,
    so a dataset added to `quality.yaml` shows up without editing this module. The
    corollary is that `_QUALITY_DROPPED_DATASETS` (and the two prefix checks below) are
    the only thing standing between a stale store and a void column in a report.
    """
    if dataset in _QUALITY_DROPPED_DATASETS:
        return None
    if dataset == "ddi_stress" or dataset.startswith("ddi_stress__") or dataset == "composition":
        return None
    return dataset or None


def _quality_cell_key(run_id: str) -> tuple[str, str, str] | None:
    """``(model, mask, column)`` for a phase2_train_free row, or None to drop it."""
    parsed = QualityRunId.parse(run_id)
    if parsed is None:
        return None
    col = _quality_dataset_to_column(parsed.dataset)
    if col is None:
        return None
    return parsed.model, parsed.variant, col


def _collect_quality(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: quality)."""
    metric = spec.get("metric")
    if metric not in _QUALITY_METRICS:
        raise ValueError(
            f"quality: unknown metric {metric!r}; expected one of "
            f"{sorted(_QUALITY_METRICS)}"
        )
    return _collect(
        kind="quality",
        metric=metric,
        rows=rows,
        store=store,
        prefix=QualityRunId.PREFIX,
        no_rows_hint="Run the phase2 sweep first.",
        bucket=lambda rs: _bucket_by_metric(rs, metric, _quality_cell_key),
        reduce=_mean,
        axes=(
            _sorted_axis,                                   # model
            _preferred_axis(_QUALITY_PREFERRED_MASKS),      # mask
            _preferred_axis(_QUALITY_PREFERRED_DATASETS),   # dataset column
        ),
    )


# ---- Restart-quality kind (C4 restart-position layout parity) ------------------------------
#
# `sequential` is the baseline layout; `restart` is the variant. The JSON body uses the
# same dataset whitelist as ``quality``; deltas vs `sequential` are a consumer concern.

_RESTART_METRICS: frozenset[str] = frozenset({"em", "f1", "presence", "loss"})
_RESTART_PREFERRED_LAYOUTS: tuple[str, ...] = ("sequential", "restart")


def _restart_cell_key(run_id: str) -> tuple[str, str, str] | None:
    """``(model, layout, column)`` for a phase4_restart row, or None to drop it.

    The mask `tag` is deliberately not part of the key: this table's axis is the layout, and
    a restart sweep runs one mask variant.
    """
    parsed = RestartRunId.parse(run_id)
    if parsed is None:
        return None
    col = _quality_dataset_to_column(parsed.dataset)
    if col is None:
        return None
    return parsed.model, parsed.layout, col


def _collect_restart_quality(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: restart_quality)."""
    metric = spec.get("metric")
    if metric not in _RESTART_METRICS:
        raise ValueError(
            f"restart_quality: unknown metric {metric!r}; expected one of "
            f"{sorted(_RESTART_METRICS)}"
        )
    return _collect(
        kind="restart_quality",
        metric=metric,
        rows=rows,
        store=store,
        prefix=RestartRunId.PREFIX,
        no_rows_hint="Run the cache/restart_quality task first.",
        bucket=lambda rs: _bucket_by_metric(rs, metric, _restart_cell_key),
        reduce=_mean,
        axes=(
            _sorted_axis,                                   # model
            _preferred_axis(_RESTART_PREFERRED_LAYOUTS),    # layout
            _preferred_axis(_QUALITY_PREFERRED_DATASETS),   # dataset column
        ),
    )


# ---- Timing kind ---------------------------------------------------------------------------

_TIMING_FULL_VARIANT = "full"

# Reading order for the N axis — NOT a filter. These are the Ns `performance.yaml` sweeps;
# any other N present in the store is appended after them rather than dropped.
_TIMING_PREFERRED_NS: tuple[int, ...] = (1, 3, 5, 9, 13, 25)
_TIMING_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")

# Store metric name → the short name `report.tables[].metric` uses. Single source of
# truth: this mapping used to exist twice — once here in the short→store direction as an
# unreferenced `_TIMING_METRIC_ALIAS`, and once as an if/elif chain inside
# `_timing_collect` — so nothing would have caught the two drifting apart.
_TIMING_RAW_METRICS: dict[str, str] = {
    "ftt_median_ms": "ttft",
    "tpot_median_ms": "tpot",
    "mask_sparsity": "sparsity",
    "analytical_speedup": "analytical",
}

# These two have no store rows of their own. `_timing_apply_derived` computes them from
# TTFT on BOTH the variant and the `full` arm at the same (model, N, L, backend, seed) —
# `realized` = full_ttft / variant_ttft, `realization` = realized / analytical_speedup.
_TIMING_DERIVED_METRICS: frozenset[str] = frozenset({"realized", "realization"})

_TIMING_METRICS: frozenset[str] = (
    frozenset(_TIMING_RAW_METRICS.values()) | _TIMING_DERIVED_METRICS
)


def _timing_collect(
    rows: Iterable[dict],
) -> tuple[
    dict[tuple[str, str, int, int, str, int], dict[str, float]],
    list[dict],
]:
    """Group all phase3_timing rows by (model, variant, N, L, backend, seed).

    Returns ``(per_cell, backing_rows)``. ``per_cell[key][short]`` is the raw metric value;
    the derived ``realized`` / ``realization`` are filled by ``_timing_apply_derived``.

    The seed here is ``provenance.seed``, not the run_id's — the derived metrics pair a
    variant with the `full` arm measured on the same hardware pass, and the provenance is
    what records that.
    """
    per_cell: dict[tuple[str, str, int, int, str, int], dict[str, float]] = {}
    backing_rows: list[dict] = []
    for row in rows:
        parsed = TimingRunId.parse(str(row.get("run_id", "")))
        if parsed is None:
            continue
        short = _TIMING_RAW_METRICS.get(str(row.get("metric")))
        if short is None:
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        seed = int((row.get("provenance") or {}).get("seed", 0))
        key = (parsed.model, parsed.variant, parsed.N, parsed.L, parsed.backend, seed)
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


def _timing_bucket(
    rows: Iterable[dict], metric: str
) -> tuple[dict[tuple, list[float]], list[dict]]:
    """Bucket timing cells by (model, L, backend, mask, N) for one metric.

    This is the one collector that cannot use `_bucket_by_metric`: a timing cell is assembled
    from SEVERAL store rows (one per raw metric) before the derived `realized` / `realization`
    can be computed from them, so the metric selector applies after grouping rather than as a
    row filter. That also makes `backing_rows` here count every row that fed a cell, not only
    the rows carrying the selected metric — which is what the seeds list needs, since a seed
    is present in the sweep whether or not this particular metric landed.
    """
    per_seed_cell, backing_rows = _timing_collect(rows)
    _timing_apply_derived(per_seed_cell)

    per_cell: dict[tuple, list[float]] = defaultdict(list)
    for (model, variant, n, l_, backend, _seed), metrics in per_seed_cell.items():
        value = metrics.get(metric)
        if value is None:
            continue  # missing cell → key omitted entirely
        per_cell[(model, l_, backend, variant, n)].append(value)
    return per_cell, backing_rows


def _collect_timing(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: timing).

    Emits ``{model: {L: {backend: {mask: {N: value}}}}}``, meaned across seeds. Masks and Ns
    are both derived from the rows present in each (model, L, backend) block and ordered by
    their preferred tuples.

    The N axis used to be the fixed `_TIMING_N_COLUMNS` tuple, ENFORCED as a filter
    (`if n not in _TIMING_N_COLUMNS: continue`). It happened to match
    `experiments/performance.yaml`, so no published timing table was wrong — but any edit to
    `Ns:` would have emptied the table with no error, because the "no phase3_timing rows in
    store" guard only checks the run_id prefix, which keeps matching perfectly well at an
    unlisted N.

    Unlike the other kinds this one does NOT raise when no cell carries the metric: a timing
    sweep legitimately produces raw metrics for arms whose derived `realized` /
    `realization` could not be paired, and an empty body is the honest answer there.
    """
    metric = spec.get("metric")
    if metric not in _TIMING_METRICS:
        raise ValueError(
            f"timing: unknown metric {metric!r}; expected one of "
            f"{sorted(_TIMING_METRICS)}"
        )
    return _collect(
        kind="timing",
        metric=metric,
        rows=rows,
        store=store,
        prefix=TimingRunId.PREFIX,
        no_rows_hint="Run the phase3 sweep first.",
        bucket=lambda rs: _timing_bucket(rs, metric),
        reduce=_mean,
        axes=(
            _sorted_axis,                                   # model
            _sorted_axis,                                   # L  (int, labelled str)
            _sorted_axis,                                   # backend
            _preferred_axis(_TIMING_PREFERRED_MASKS),       # mask
            _preferred_axis(_TIMING_PREFERRED_NS),          # N  (int, labelled str)
        ),
        require_cells=False,
    )


# ---- Throughput kind ------------------------------------------------------------------------

_THROUGHPUT_PREFERRED_MODES: tuple[str, ...] = ("ddi_cache", "ddi_recompute", "full_causal")

# Used when a run_id carries neither a layout nor a reuse segment. Such a row still gets a
# cell rather than being dropped, but it is not silently merged with the shaped ones.
# The grammar itself lives in `ddi.runids.ThroughputRunId`; aliased here because the arm axis
# has to sort this value last.
_THROUGHPUT_UNKNOWN_ARM = ThroughputRunId.UNKNOWN_ARM


def _throughput_cell_key(run_id: str) -> tuple[str, str, str] | None:
    """``(model, mode, arm)`` for a phase4_throughput row, or None to drop it.

    ``arm`` is ``"{layout}/reuse={reuse}"`` — the only axis the run_id carries besides the
    mode. It used to be an ``N={N}`` scan with a ``"n/a"`` sentinel fallback. No producer has
    ever emitted ``N=`` here (the throughput runner takes a scalar ``docs_per_request``, not
    an ``Ns:`` sweep), so the sentinel was the only value the axis ever took —
    ``cache_throughput.yaml``'s ``cols: N`` rendered as one unnamed column. Meanwhile
    ``layout`` and ``reuse=``, which ARE in every run_id and DO name different experiments,
    were parsed and then dropped from the cell key: a store holding both a `sequential` and a
    `restart` throughput run averaged them into a single number with nothing in the output
    saying so. The sentinel is gone and the key now carries the axis that actually exists.

    Still not carried by the run_id, and therefore still collidable: ``top_docs``,
    ``docs_per_request`` and ``doc_token_len``. Two runs differing only in those merge into
    one cell. Fixing that means changing the run_id in ``ddi/run.py``.
    """
    parsed = ThroughputRunId.parse(run_id)
    if parsed is None:
        return None
    return parsed.model, parsed.mode, parsed.arm


def _throughput_arm_axis(values: Iterable[str]) -> list[str]:
    """Arms sort alphabetically; the unshaped sentinel goes last so it never leads a row."""
    arms = set(values)
    ordered = sorted(arms - {_THROUGHPUT_UNKNOWN_ARM})
    if _THROUGHPUT_UNKNOWN_ARM in arms:
        ordered.append(_THROUGHPUT_UNKNOWN_ARM)
    return ordered


def _collect_throughput(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: throughput)."""
    metric = spec.get("metric") or "qps"
    return _collect(
        kind="throughput",
        metric=metric,
        rows=rows,
        store=store,
        prefix=ThroughputRunId.PREFIX,
        no_rows_hint="Run the phase4 sweep first.",
        bucket=lambda rs: _bucket_by_metric(rs, metric, _throughput_cell_key),
        reduce=_mean,
        axes=(
            _sorted_axis,                                   # model
            _preferred_axis(_THROUGHPUT_PREFERRED_MODES),   # mode
            _throughput_arm_axis,                           # arm
        ),
    )


# ---- Correctness kind ----------------------------------------------------------------------
#
# Cell = ``max(max_abs_logit_diff)`` across the ``n_queries_per_doc`` queries at each
# (model, mode, N, L). ``max`` (not mean) is the worst-case gate — a single bad query
# would otherwise be washed out.

_CORRECTNESS_BASELINE_MODE = "full_causal"

# Baseline (``full_causal``) first, then remaining modes alphabetically — the reference
# arm has to lead the block for the table to read as a comparison against it.
_CORRECTNESS_PREFERRED_MODES: tuple[str, ...] = (_CORRECTNESS_BASELINE_MODE,)


def _correctness_cell_key(run_id: str) -> tuple[str, int, str, int] | None:
    """``(model, L, mode, N)`` for a phase4_cache row, or None to drop it.

    Note the component order: L before mode. That is the order the axes are *read* in — the
    mode axis is ordered within each L — while the emitted JSON nests mode outside L (see
    `_CORRECTNESS_NEST_ORDER`). The two differ, and this collector is the only one where
    they do.
    """
    parsed = CorrectnessRunId.parse(run_id)
    if parsed is None:
        return None
    return parsed.model, parsed.L, parsed.mode, parsed.N


# {model: {mode: {L: {N: value}}}} out of a (model, L, mode, N) traversal — mode is the outer
# axis under model per the spec, but its reading order is settled per-L, first L first.
_CORRECTNESS_NEST_ORDER: tuple[int, ...] = (0, 2, 1, 3)


def _collect_correctness(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Return the JSON body for one report.tables[] entry (kind: correctness)."""
    metric = spec.get("metric") or "max_abs_logit_diff"
    return _collect(
        kind="correctness",
        metric=metric,
        rows=rows,
        store=store,
        prefix=CorrectnessRunId.PREFIX,
        no_rows_hint="Run the cache/correctness task first.",
        bucket=lambda rs: _bucket_by_metric(rs, metric, _correctness_cell_key),
        # Worst-case gate: max, not mean. A single bad query would otherwise be washed out.
        reduce=max,
        axes=(
            _sorted_axis,                                   # model
            _sorted_axis,                                   # L (int, labelled str)
            _preferred_axis(_CORRECTNESS_PREFERRED_MODES),  # mode
            _sorted_axis,                                   # N (int, labelled str)
        ),
        nest_order=_CORRECTNESS_NEST_ORDER,
    )


# ---- Finetune / perplexity kinds (parity with ddi/finetune.py rows, #120) ------------------
#
# These are intentionally thin — dump matching store rows keyed by (model, step) or
# (model, epoch). We don't have a paper-shape for these yet; the goal is only that
# `make exp-finetune` completes end-to-end and surfaces the training trajectory + pplx
# trajectory in the summary JSON for eyeballing. Fancier renderers (loss curves,
# pplx-vs-teacher deltas) are #120 follow-ups.

def _collect_finetune_loss(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Dump per-step training rows (kl_loss + lr) keyed by (model, step)."""
    if not any(str(r.get("run_id", "")).startswith(FinetuneRunId.PREFIX) for r in rows):
        raise ValueError(
            f"finetune_loss: no finetune rows in store {store}. "
            "Run the finetune YAML first."
        )
    # (model, step) -> {metric: value}
    per_step: dict[tuple[str, int], dict[str, float]] = {}
    backing_rows: list[dict] = []
    seeds: set[int] = set()
    for row in rows:
        parsed = FinetuneRunId.parse(str(row.get("run_id", "")))
        if parsed is None or parsed.step is None:
            continue
        metric = row.get("metric")
        if metric not in ("kl_loss", "lr"):
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (parsed.model, parsed.step)
        per_step.setdefault(key, {})[metric] = value
        seeds.add(parsed.seed)
        backing_rows.append(row)

    # {model: {step: {kl_loss, lr}}}
    data: dict[str, dict[str, dict[str, float]]] = {}
    for (model, step), metrics in per_step.items():
        data.setdefault(model, {})[str(step)] = metrics
    # Sort step keys numerically within each model.
    for model in data:
        data[model] = {
            str(k): data[model][str(k)] for k in sorted(int(s) for s in data[model])
        }

    out = {
        "kind": "finetune_loss",
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted(seeds),
        "models": sorted(data),
        "data": data,
    }

    # An empty table here is not the same as "training produced no loss", but it renders
    # identically to a reader — VIBEQUEUE-2 row 8 finished a full 8-GPU finetune and its
    # report showed `backing_rows: 0, data: {}` with nothing anywhere saying why.
    #
    # Deliberately a note rather than a raise: the run above it trained correctly and
    # uploaded a usable checkpoint, so failing the whole aggregation — and with it the
    # graph's post_grid block — over a missing curve would be a worse outcome than a
    # loud, self-explaining hole in the report. The `no finetune rows at all` case above
    # still raises, because that one means the finetune YAML was never run.
    if not backing_rows:
        out["warning"] = (
            "no per-step rows: the store holds finetune rows (epoch-level perplexity) but "
            "no kl_loss/lr at any step. Cause is almost always a run shorter than "
            "`log_every` (default 50) on a build predating the first/last-step fix — "
            "training was fine, only the curve is missing."
        )
    return out


def _collect_perplexity(spec: dict, store: Path, rows: list[dict]) -> dict:
    """Dump per-epoch perplexity rows keyed by (model, epoch).

    Emits every ``perplexity_*`` metric name unchanged (pre/post student, teacher)
    so a reader can compare trajectories without hunting through the store.
    """
    if not any(str(r.get("run_id", "")).startswith(FinetuneRunId.PREFIX) for r in rows):
        raise ValueError(
            f"perplexity: no finetune rows in store {store}. "
            "Run the finetune YAML first."
        )
    # (model, epoch) -> {metric: value}
    per_epoch: dict[tuple[str, int], dict[str, float]] = {}
    backing_rows: list[dict] = []
    seeds: set[int] = set()
    for row in rows:
        parsed = FinetuneRunId.parse(str(row.get("run_id", "")))
        if parsed is None or parsed.epoch is None:
            continue
        metric = row.get("metric")
        if not isinstance(metric, str) or not metric.startswith("perplexity_"):
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (parsed.model, parsed.epoch)
        per_epoch.setdefault(key, {})[metric] = value
        seeds.add(parsed.seed)
        backing_rows.append(row)

    # {model: {epoch: {perplexity_*: value}}}
    data: dict[str, dict[str, dict[str, float]]] = {}
    for (model, epoch), metrics in per_epoch.items():
        data.setdefault(model, {})[str(epoch)] = metrics
    for model in data:
        data[model] = {
            str(k): data[model][str(k)] for k in sorted(int(e) for e in data[model])
        }

    return {
        "kind": "perplexity",
        "backing_rows": len(backing_rows),
        "distinct_run_ids": len({r.get("run_id") for r in backing_rows}),
        "seeds": sorted(seeds),
        "models": sorted(data),
        "data": data,
    }


# ---- Dispatcher + public API ---------------------------------------------------------------

_COLLECTORS: dict[str, Callable[[dict, Path, list[dict]], dict]] = {
    "quality": _collect_quality,
    "timing": _collect_timing,
    "throughput": _collect_throughput,
    "restart_quality": _collect_restart_quality,
    "correctness": _collect_correctness,
    "finetune_loss": _collect_finetune_loss,
    "perplexity": _collect_perplexity,
}


def _log(msg: str) -> None:
    """Print a progress line to stderr with the [ddi.aggregate] prefix (flushed)."""
    print(f"[ddi.aggregate] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# What a `report.tables[]` entry is allowed to say, per kind.
#
# Every collector reads its spec through `spec.get(...)`, so an unrecognised key is not an
# error — it is simply never looked at, while sitting in a checked-in YAML teaching every
# subsequent reader that the knob works. This mirrors `ddi/run.py::_validate_set_keys`,
# which exists because a `--set` override was accepted, folded into the config hash so the
# store row looked distinct, and then never read — it cost an 8-GPU allocation.
#
# The same class of key had accumulated here: `split_by: [L, backend]` on all six
# performance.yaml tables, `rows:`/`cols:` on every table in every YAML. They described a
# markdown renderer that no longer exists; the JSON body's shape is fixed per kind (and
# for timing already nests by L then backend, which is what `split_by` was asking for).
# Adding a spec key means adding it here, or aggregation refuses it.
# ---------------------------------------------------------------------------

_TABLE_KEYS_COMMON: frozenset[str] = frozenset({"kind"})

_TABLE_KEYS: dict[str, frozenset[str]] = {
    "quality": _TABLE_KEYS_COMMON | {"metric"},
    "timing": _TABLE_KEYS_COMMON | {"metric"},
    "throughput": _TABLE_KEYS_COMMON | {"metric"},
    "restart_quality": _TABLE_KEYS_COMMON | {"metric"},
    "correctness": _TABLE_KEYS_COMMON | {"metric"},
    # These two dump every matching store row; they have no metric selector.
    "finetune_loss": _TABLE_KEYS_COMMON,
    "perplexity": _TABLE_KEYS_COMMON,
}


def _validate_table_keys(kind: str, spec: dict) -> None:
    """Reject a ``report.tables[]`` key this kind's collector will never read."""
    allowed = _TABLE_KEYS[kind]
    for key in spec:
        if key in allowed:
            continue
        raise ValueError(
            f"report.tables[]: kind {kind!r} has no table key {str(key)!r}, so the "
            "setting would be merged into the spec, never read, and the rendered table "
            "would be exactly the same as if it were absent — a config that documents a "
            "knob nobody implements. "
            f"Valid keys for kind={kind!r}: {', '.join(sorted(allowed))}. "
            "The emitted JSON shape is fixed per kind (see this module's docstring); if "
            "you need a different shape, that is a code change, not a config key."
        )


def collect_table(
    spec: dict,
    store: Path,
    model_filter: str | None = None,
    rows: list[dict] | None = None,
) -> dict:
    """Collect one ``report.tables[]`` entry into a JSON-serializable dict.

    ``spec`` keys:
      - ``kind``: ``quality`` | ``timing`` | ``throughput`` | ``restart_quality`` |
        ``correctness`` | ``finetune_loss`` | ``perplexity``.
      - ``metric``: which metric to render (not accepted by ``finetune_loss`` /
        ``perplexity``, which dump every matching row).

    Any other key raises — see ``_TABLE_KEYS``.

    ``model_filter`` scopes every collector to a single model — the store is shared and
    append-only, so a caller that just ran one model wants to see only that model in the
    summary.

    ``rows`` is the already-loaded, already-model-filtered store. Every collector used to
    call ``load_rows`` itself (and ``timing`` called it twice), so rendering
    `performance.yaml`'s six timing tables re-read and re-JSON-parsed the whole store seven
    times. `aggregate` now loads once and passes the list down; ``store`` is still taken
    because the "no rows in store" errors name it. Left optional so the public single-table
    entry point keeps working on its own.
    """
    kind = spec.get("kind")
    if kind not in _COLLECTORS:
        raise ValueError(
            f"unknown table kind {kind!r}; expected one of {sorted(_COLLECTORS)}"
        )
    _validate_table_keys(kind, spec)
    if rows is None:
        rows = load_rows(store, model_filter)
    return _COLLECTORS[kind](spec, store, rows)


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
    # Read and parse the store ONCE for the whole YAML rather than once per collector.
    rows = load_rows(store, model_filter)
    _log(f"loaded {len(rows)} store row(s) ({time.perf_counter() - t0:.2f}s)")
    tables: list[dict] = []
    for i, spec in enumerate(tables_spec, 1):
        kind = spec.get("kind", "?")
        metric = spec.get("metric", "?")
        t = time.perf_counter()
        try:
            tables.append(collect_table(spec, store, model_filter, rows=rows))
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
