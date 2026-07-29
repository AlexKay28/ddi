"""YAML-driven table builder (issue #86, sub-issue of #81).

One CLI reads the ``report.tables:`` block from an experiment YAML and emits
one Markdown table per entry (with the #79 provenance footer), concatenated
into a single ``summary.md``. Replaces every ``ddi/reports/*.py`` renderer.

Dispatch on ``kind``:

- ``quality``   → phase2_train_free rows, rows=mask × cols=dataset, cells with
                  the #72 colorization contract (``value (Δ vs full)``, 🟢 when
                  |Δ| < 5% of ``full`` and both values have signal).
- ``timing``    → phase3_timing rows, rows=mask × cols=N. Optional
                  ``split_by: [L, backend]`` emits one sub-table per group.
                  ``realized`` and ``realization`` derived by pairing each
                  variant row with the matching ``full`` row at the same
                  ``(model, N, L, backend, seed)``.
- ``throughput``→ phase4_throughput rows, rows=mode × cols=N.

Design notes / why:

- **Delegates row access** to ``ddi.report.iter_rows`` / ``filter_rows`` so the
  store format stays owned by one module.
- **Reuses** ``ddi.reports._provenance.render_provenance_footer`` — every
  emitted table ends with the same footer #79 stamps.
- **Deterministic** by design: sorted iteration everywhere, no timestamps in
  the table body (the footer is the only volatile field, marked stable by the
  ``<!-- provenance -->`` line so tests can grep for it).
- **Open/closed for new kinds**: dispatcher ``_RENDERERS`` is a plain dict —
  a new kind adds one entry and one ``_render_<kind>`` function.

CLI:

    python -m ddi.aggregate <exp.yaml> --out <summary.md>
                            [--store results/runs.jsonl] [--append]

The YAML block shape (excerpt from #81):

    report:
      tables:
        - kind: quality
          rows: mask
          cols: dataset
          metric: f1
        - kind: timing
          rows: mask
          cols: N
          metric: realized
          split_by: [L, backend]
        - kind: throughput
          rows: mode
          cols: N
          metric: qps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from ddi.reports._provenance import render_provenance_footer


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

# ---- Colorization contract (from color_tables.py, #72 Qwen comment) ------------------------
#
# The rules that govern ``render_cell`` are load-bearing for how the paper reads: 🟢 only
# means "quality parity" when the metric can actually discriminate. The floor / ceiling
# suppression rules exist so a degenerate metric (F1 = 0.00 under blind AND under full on
# ``ds/contra``) never reads as a "win" for a variant that simply happens to also be at floor.

_GREEN_THRESHOLD = 0.05  # |Δ| below 5% of |full| → 🟢
_SMALL_BASELINE = 0.05  # |full| below this → show absolute Δ (percentage would explode)
_LOSS_CEILING = 10.0  # loss above this ≈ uniform-over-vocab; "parity" isn't meaningful
_SIGNAL_FLOOR = 0.01  # accuracy metrics below this on BOTH arms → no green (both at noise)


def _render_cell(
    variant: float,
    full: float,
    *,
    is_loss: bool = False,
    allow_green: bool = True,
) -> str:
    """Return the cell string ``value (Δ vs full)`` with optional 🟢 prefix.

    See module docstring for the contract; keep behavior byte-identical to
    ``/tmp/color_tables.py::render_cell`` — the paper's tables were
    reviewed with those exact rules.
    """
    val = f"{variant:.4f}" if not is_loss else f"{variant:.2f}"
    abs_diff = variant - full
    sign = "+" if abs_diff >= 0 else "−"
    if abs(full) >= _SMALL_BASELINE:
        pct = abs_diff / abs(full) * 100.0
        diff_str = f"Δ{sign}{abs(pct):.1f}%"
        within_threshold = abs(pct) / 100.0 < _GREEN_THRESHOLD
    else:
        # Baseline too small for a meaningful percentage — fall back to absolute Δ.
        abs_disp = f"{abs(abs_diff):.4f}" if not is_loss else f"{abs(abs_diff):.2f}"
        diff_str = f"Δ{sign}{abs_disp}"
        # A drop from 0.03 to 0.00 is a 100% relative collapse, not "parity" — key off
        # |full| as the threshold, not a fixed absolute number.
        within_threshold = (
            abs(abs_diff) <= abs(full) * _GREEN_THRESHOLD if full != 0 else abs_diff == 0
        )
    # Suppress green when both sides are at floor / ceiling — parity there is degenerate.
    if is_loss:
        has_signal = not (full >= _LOSS_CEILING and variant >= _LOSS_CEILING)
    else:
        has_signal = not (full < _SIGNAL_FLOOR and variant < _SIGNAL_FLOOR)
    prefix = "🟢 " if (allow_green and within_threshold and has_signal) else ""
    return f"{prefix}{val} ({diff_str})"


def _fmt(v: float, *, is_loss: bool = False) -> str:
    return f"{v:.4f}" if not is_loss else f"{v:.2f}"


# ---- Quality kind (parity with T2_em / T2_f1 / T2_presence / T2_loss) -----------------------
#
# Column order and dataset mapping are lifted verbatim from
# ``ddi/reports/phase2_quality.py`` so the tables read identically in the tracking issues.

_QUALITY_PREFIX = "phase2_train_free:"

_QUALITY_DATASET_COLUMNS: tuple[str, ...] = (
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

_QUALITY_STRESS_TO_COL: dict[str, str] = {
    "fact_chain": "ds/fact_chain",
    "fact_aggregation": "ds/fact_agg",
    "contradiction": "ds/contra",
    "coreference": "ds/coref",
    "needle": "ds/needle",
}

# ``nq`` is degenerate (no docs); older EXP-01 rows still exist in the store — drop them
# explicitly so the aggregator matches the #72 templates.
_QUALITY_DROPPED_DATASETS: frozenset[str] = frozenset({"nq"})

_QUALITY_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")

_QUALITY_METRIC_HEADERS: dict[str, str] = {
    "em": "EM",
    "f1": "F1",
    "presence": "presence",
    "loss": "loss",
}

_EM_DASH = "—"


def _quality_parse_run_id(run_id: str) -> tuple[str, str, str, int] | None:
    """Return ``(model, dataset, variant, seed)`` from a phase2_train_free run_id.

    Shape: ``phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha}:{record_id}``.
    Returns None if the shape does not match — a malformed row is silently skipped so a
    single bad line in the store never breaks a render.
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
    if dataset.startswith("ddi_stress__"):
        family = dataset[len("ddi_stress__"):]
        return _QUALITY_STRESS_TO_COL.get(family)
    if dataset == "ddi_stress":
        # Bare `ddi_stress` (EXP-01 legacy) has no family suffix — ambiguous; drop.
        return None
    return dataset if dataset in _QUALITY_DATASET_COLUMNS else None


def _quality_mask_order(masks: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(masks))
    ordered = [m for m in _QUALITY_PREFERRED_MASKS if m in uniq]
    extras = sorted(m for m in uniq if m not in _QUALITY_PREFERRED_MASKS)
    return ordered + extras


def _render_quality(spec: dict, store: Path) -> str:
    metric = spec.get("metric")
    if metric not in _QUALITY_METRIC_HEADERS:
        raise ValueError(
            f"quality: unknown metric {metric!r}; expected one of "
            f"{sorted(_QUALITY_METRIC_HEADERS)}"
        )
    is_loss = metric == "loss"

    all_rows = list(iter_rows(store))
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

    header_cols = " | ".join(_QUALITY_DATASET_COLUMNS)
    header_sep = "|".join(["---"] * (len(_QUALITY_DATASET_COLUMNS) + 1))
    metric_label = _QUALITY_METRIC_HEADERS[metric]

    blocks: list[str] = []
    for model in models:
        lines = [
            f"### {model} — {metric_label}",
            "",
            f"| mask | {header_cols} |",
            f"|{header_sep}|",
        ]
        masks = _quality_mask_order(masks_seen[model])
        # Full baseline per column (needed for Δ on the variant rows).
        full_vals: dict[str, float | None] = {}
        for col in _QUALITY_DATASET_COLUMNS:
            values = per_cell.get((model, "full", col))
            full_vals[col] = (sum(values) / len(values)) if values else None
        for mask in masks:
            cells: list[str] = []
            for col in _QUALITY_DATASET_COLUMNS:
                values = per_cell.get((model, mask, col))
                if not values:
                    cells.append(_EM_DASH)
                    continue
                val = sum(values) / len(values)
                if mask == "full":
                    cells.append(_fmt(val, is_loss=is_loss))
                    continue
                full = full_vals.get(col)
                if full is None:
                    # No baseline to Δ against — fall back to raw formatting.
                    cells.append(_fmt(val, is_loss=is_loss))
                    continue
                # `blind` row is *never* coloured — parity there is a ground-rule-7 failure,
                # not a win (see color_tables.py comment and #72 Qwen comment).
                allow_green = mask != "blind"
                cells.append(
                    _render_cell(val, full, is_loss=is_loss, allow_green=allow_green)
                )
            lines.append(f"| {mask} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    # One footer covers the union of sub-tables — same store slice, per-block would duplicate.
    return "\n\n".join(blocks) + "\n" + render_provenance_footer(store, backing_rows)


# ---- Timing kind (parity with T_perf_ttft / tpot / sparsity / analytical / realized / ratio)
#
# Shape mirrors ``ddi/reports/phase3_realization.py``: N columns fixed to
# ``(1, 3, 5, 9, 13, 25)``, one sub-table per (model, L, backend) triple, ``realized`` and
# ``realization`` are DERIVED at render time by pairing variant rows with matching ``full``.

_TIMING_PREFIX = "phase3_timing:"
_TIMING_FULL_VARIANT = "full"

_TIMING_N_COLUMNS: tuple[int, ...] = (1, 3, 5, 9, 13, 25)
_TIMING_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")

_TIMING_METRIC_HEADERS: dict[str, str] = {
    "ttft": "TTFT (median ms)",
    "tpot": "TPOT (median ms)",
    "sparsity": "mask sparsity",
    "analytical": "analytical speedup",
    "realized": "realized speedup",
    "realization": "realization ratio",
}

# Map short metric name from YAML → store metric name it aggregates. `realized` and
# `realization` are DERIVED so they need TTFT rows on both the variant and `full` arms.
_TIMING_SHORT_TO_STORE: dict[str, str] = {
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
    for row in iter_rows(store):
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


def _timing_fmt_cell(metric: str, value: float | None) -> str:
    if value is None:
        return _EM_DASH
    if metric in ("ttft", "tpot"):
        return f"{value:.2f}"
    if metric in ("analytical", "realized"):
        return f"{value:.2f}x"
    if metric == "realization":
        return f"{value:.3f}"
    if metric == "sparsity":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _render_timing(spec: dict, store: Path) -> str:
    metric = spec.get("metric")
    if metric not in _TIMING_METRIC_HEADERS:
        raise ValueError(
            f"timing: unknown metric {metric!r}; expected one of "
            f"{sorted(_TIMING_METRIC_HEADERS)}"
        )
    split_by = spec.get("split_by") or []
    if not isinstance(split_by, (list, tuple)):
        raise ValueError(f"timing: split_by must be a list, got {type(split_by).__name__}")
    unknown = [s for s in split_by if s not in ("L", "backend")]
    if unknown:
        raise ValueError(
            f"timing: split_by keys must be a subset of ['L', 'backend'], got extras {unknown}"
        )

    if not any(str(r.get("run_id", "")).startswith(_TIMING_PREFIX) for r in iter_rows(store)):
        raise ValueError(
            f"timing[{metric}]: no phase3_timing rows in store {store}. "
            "Run the phase3 sweep first."
        )

    per_cell, backing_rows = _timing_collect(store)
    _timing_apply_derived(per_cell)

    # Aggregate values per (model, L, backend, mask, N) by mean across seeds. Track masks
    # separately so a variant with only derived metrics missing still renders as em-dashes
    # rather than vanishing from the table.
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

    metric_label = _TIMING_METRIC_HEADERS[metric]
    header_cols = " | ".join(f"N={n}" for n in _TIMING_N_COLUMNS)
    header_sep = "|".join(["---"] * (len(_TIMING_N_COLUMNS) + 1))

    blocks: list[str] = [f"# timing — {metric_label}"]

    triples = sorted(all_masks)
    for (model, l_, backend) in triples:
        # Honor split_by: if L is absent from split_by, we roll up all L's — but since our
        # store keys carry L, "roll up" would require a re-aggregation pass. The current
        # renderer semantics (see phase3_realization._render_perf_metric) always split on
        # (model, L, backend); split_by is documented as an OPTIONAL hint. Preserve that.
        header_parts = [f"## {model}"]
        if "L" in split_by or not split_by:
            header_parts.append(f"L={l_}")
        if "backend" in split_by or not split_by:
            header_parts.append(f"backend={backend}")
        heading = " — ".join(header_parts) if len(header_parts) > 1 else header_parts[0]
        lines = [heading, "", f"| mask | {header_cols} |", f"|{header_sep}|"]
        masks = _timing_mask_order(all_masks[(model, l_, backend)])
        for mask in masks:
            cells: list[str] = []
            for n in _TIMING_N_COLUMNS:
                vals = agg.get((model, l_, backend, mask, n))
                if not vals:
                    cells.append(_EM_DASH)
                else:
                    cells.append(_timing_fmt_cell(metric, sum(vals) / len(vals)))
            lines.append(f"| {mask} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n" + render_provenance_footer(store, backing_rows)


# ---- Throughput kind (parity with F5 — phase4_throughput) ----------------------------------
#
# F5 in ``ddi/reports/phase4_throughput.py`` renders a PLOT; the ``kind: throughput`` table
# entry in the YAML spec (#81) is the *tabular* form of the same data — rows=mode × cols=N,
# cells = mean qps at that mode/N.

_THROUGHPUT_PREFIX = "phase4_throughput:"
_THROUGHPUT_PREFERRED_MODES: tuple[str, ...] = ("ddi_cache", "ddi_recompute", "full_causal")


def _throughput_parse_run_id(run_id: str) -> tuple[str, str] | None:
    """Return ``(mode, N_str_or_none)`` from a phase4_throughput run_id.

    Shape: ``phase4_throughput:{model}:{mode}:{layout}:reuse={reuse}:seed={seed}:{sha}``
    or with an optional ``N={N}`` segment inserted before ``reuse=`` when the runner
    encodes the doc count. We tolerate both: a run without ``N=`` maps to the sentinel
    N column ``"n/a"``, which the aggregator surfaces as a single unnamed N column.
    """
    if not run_id.startswith(_THROUGHPUT_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 7:
        return None
    mode = parts[2]
    n_str = None
    for seg in parts[3:]:
        if seg.startswith("N="):
            n_str = seg[2:]
            break
    return mode, (n_str if n_str is not None else "n/a")


def _throughput_mode_order(modes: Iterable[str]) -> list[str]:
    uniq = list(dict.fromkeys(modes))
    ordered = [m for m in _THROUGHPUT_PREFERRED_MODES if m in uniq]
    extras = sorted(m for m in uniq if m not in _THROUGHPUT_PREFERRED_MODES)
    return ordered + extras


def _render_throughput(spec: dict, store: Path) -> str:
    metric = spec.get("metric") or "qps"

    all_rows = list(iter_rows(store))
    if not any(str(r.get("run_id", "")).startswith(_THROUGHPUT_PREFIX) for r in all_rows):
        raise ValueError(
            f"throughput[{metric}]: no phase4_throughput rows in store {store}. "
            "Run the phase4 sweep first."
        )

    # (mode, N) -> [values]
    per_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_ns: set[str] = set()
    backing_rows: list[dict] = []
    for row in filter_rows(all_rows, metric=metric):
        rid = str(row.get("run_id", ""))
        parsed = _throughput_parse_run_id(rid)
        if parsed is None:
            continue
        mode, n_str = parsed
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        per_cell[(mode, n_str)].append(val)
        all_ns.add(n_str)
        backing_rows.append(row)

    if not per_cell:
        raise ValueError(
            f"throughput[{metric}]: no phase4_throughput rows carried metric={metric!r}."
        )

    # Sort N columns numerically when possible, else lexicographic (keeps the "n/a" sentinel
    # at the end deterministically).
    def _n_sort_key(n_str: str) -> tuple[int, int | str]:
        try:
            return (0, int(n_str))
        except ValueError:
            return (1, n_str)

    n_cols = sorted(all_ns, key=_n_sort_key)
    modes = _throughput_mode_order({m for (m, _n) in per_cell})

    header_cols = " | ".join(f"N={n}" for n in n_cols)
    header_sep = "|".join(["---"] * (len(n_cols) + 1))
    lines = [
        f"# throughput — {metric}",
        "",
        f"| mode | {header_cols} |",
        f"|{header_sep}|",
    ]
    for mode in modes:
        cells: list[str] = []
        for n in n_cols:
            vals = per_cell.get((mode, n))
            cells.append(_EM_DASH if not vals else f"{sum(vals) / len(vals):.4f}")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n" + render_provenance_footer(store, backing_rows)


# ---- Dispatcher + public API ---------------------------------------------------------------

_RENDERERS: dict[str, Callable[[dict, Path], str]] = {
    "quality": _render_quality,
    "timing": _render_timing,
    "throughput": _render_throughput,
}


def render_table(spec: dict, store: Path) -> str:
    """Render one ``report.tables[]`` entry to a Markdown string with provenance footer.

    ``spec`` keys:
      - ``kind``: ``quality`` | ``timing`` | ``throughput``.
      - ``rows`` / ``cols``: axis names (informational; the emitted shape is fixed per kind).
      - ``metric``: which metric to render.
      - ``split_by`` (optional, timing only): list of axes to split sub-tables on.
    """
    kind = spec.get("kind")
    if kind not in _RENDERERS:
        raise ValueError(
            f"unknown table kind {kind!r}; expected one of {sorted(_RENDERERS)}"
        )
    return _RENDERERS[kind](spec, store)


def aggregate(
    exp_yaml: Path,
    out: Path,
    store: Path = Path("results/runs.jsonl"),
    append: bool = False,
) -> None:
    """Read ``report.tables:`` from ``exp_yaml``, emit each table into ``out``.

    Each ``report.tables[]`` entry becomes one Markdown block ending with a provenance
    footer, separated by blank lines. If ``append`` is True, blocks are appended to an
    existing ``out``; otherwise ``out`` is overwritten.
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

    rendered: list[str] = [render_table(spec, store) for spec in tables_spec]
    body = "\n\n".join(rendered)
    if not body.endswith("\n"):
        body += "\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as f:
        if append and out.exists() and out.stat().st_size > 0:
            f.write("\n")
        f.write(body)


# ---- CLI -----------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ddi.aggregate",
        description=(
            "Read report.tables from an experiment YAML and emit Markdown tables "
            "(with provenance footer) into summary.md."
        ),
    )
    p.add_argument("exp_yaml", type=Path, help="Path to experiment YAML.")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the composed summary.md.",
    )
    p.add_argument(
        "--store",
        type=Path,
        default=Path("results/runs.jsonl"),
        help="Path to the JSONL store (default: results/runs.jsonl).",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to --out instead of overwriting.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    aggregate(args.exp_yaml, args.out, store=args.store, append=args.append)
    return 0


if __name__ == "__main__":
    sys.exit(main())
