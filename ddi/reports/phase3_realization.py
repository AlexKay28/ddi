"""Phase 3.5 realization-ratio renderers (issue #21).

T3 tabulates analytical vs realized vs realization ratio per `N`; F3 plots realization ratio vs
`N` with one line per `L`. Both operate over already-collected `phase3_timing` rows — Phase 3.5's
contribution is analysis, not a new timing sweep. The 9x gap between the analytical 10.59x
(RESEARCH_PLAN §7) and the archaeology-measured 1.20x is the paper's most interesting datapoint;
these renderers are the visualization for that story (§10, F3/T3 spec).
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from ddi.report import filter_rows, iter_rows, register
from ddi.reports._provenance import render_provenance_footer

_PHASE3_PREFIX = "phase3_timing:"
_TARGET_VARIANT = "ddi_qf"
_FULL_VARIANT = "full"


def _parse(run_id: str) -> dict[str, str | int] | None:
    """Parse a phase3_timing run_id.

    Shape: `phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant_tag}:{backend}:seed=...:{sha}`.

    WHY the segment order is fixed: phase3_timing._run is the sole writer of these ids and
    encodes the shape parameters positionally so downstream readers (this file, F3) do not
    need to re-scan provenance. Parsing rejects anything that does not match — the runner is
    the source of truth for the shape.
    """
    if not run_id.startswith(_PHASE3_PREFIX):
        return None
    parts = run_id.split(":")
    # Model may contain "/" so it does not create extra colon segments; layout is stable.
    # Expect: [phase3_timing, model, N=x, L=x, Q=x, variant_tag, backend, seed=x, sha]
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
        "variant_tag": parts[5],
        "backend": parts[6],
    }


def _cell_key(parsed: dict[str, str | int], seed: int) -> tuple:
    """One row per (model, N, L, Q, backend, seed) — the unit at which ftt is measured."""
    return (parsed["model"], parsed["N"], parsed["L"], parsed["Q"], parsed["backend"], seed)


def _collect_ftt_by_cell(
    store: Path,
    *,
    target_variant: str,
) -> dict[tuple, dict[str, float]]:
    """Group ftt_median_ms rows into per-cell {variant: ftt_ms} dicts.

    Cells with only one arm (full or target) are kept — the caller decides what to do with a
    missing arm (T3 prints n/a; F3 skips the point).
    """
    rows = list(filter_rows(iter_rows(store), metric="ftt_median_ms"))
    per_cell: dict[tuple, dict[str, float]] = {}
    for row in rows:
        parsed = _parse(row.get("run_id", ""))
        if parsed is None:
            continue
        variant_tag = parsed["variant_tag"]
        # Only pair the specific target variant against `full`; other tags (e.g. bridge_k) do
        # not belong in the T3/F3 story and are consumed by F6.
        if variant_tag not in (target_variant, _FULL_VARIANT):
            continue
        seed = int((row.get("provenance") or {}).get("seed", 0))
        key = _cell_key(parsed, seed)
        per_cell.setdefault(key, {})[variant_tag] = float(row["value"])
    return per_cell


def _collect_scalar_by_cell(
    store: Path,
    *,
    metric: str,
    variant_tag: str,
) -> dict[tuple, float]:
    """Latest value per cell for a scalar metric (analytical_speedup / mask_sparsity)."""
    out: dict[tuple, float] = {}
    for row in filter_rows(iter_rows(store), metric=metric):
        parsed = _parse(row.get("run_id", ""))
        if parsed is None or parsed["variant_tag"] != variant_tag:
            continue
        seed = int((row.get("provenance") or {}).get("seed", 0))
        key = _cell_key(parsed, seed)
        try:
            out[key] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return out


def _any_phase3_rows(store: Path) -> bool:
    for row in iter_rows(store):
        if str(row.get("run_id", "")).startswith(_PHASE3_PREFIX):
            return True
    return False


def _phase3_backing_rows(store: Path) -> list[dict]:
    """Materialize every phase3_timing row in the store for the provenance footer.

    We stamp on the phase3_timing prefix rather than the specific metric per renderer
    because a T_perf_* table's realized/realization cells are derived from *pairs*
    (variant vs full arm) — the "backing evidence" is the whole phase3 slice at render
    time, not just the metric column that named this renderer.
    """
    return [
        row
        for row in iter_rows(store)
        if str(row.get("run_id", "")).startswith(_PHASE3_PREFIX)
    ]


@register("T3")
def _t3_realization_table(store: Path) -> str:
    """Markdown table: per N, analytical vs realized (median) vs realization ratio vs sparsity."""
    if not _any_phase3_rows(store):
        raise ValueError(
            "T3: store has no phase3_timing rows. Run `make timing-N` (or the equivalent Phase 3 "
            "sweep) first."
        )

    per_cell = _collect_ftt_by_cell(store, target_variant=_TARGET_VARIANT)
    analytical = _collect_scalar_by_cell(
        store, metric="analytical_speedup", variant_tag=_TARGET_VARIANT
    )
    sparsity = _collect_scalar_by_cell(
        store, metric="mask_sparsity", variant_tag=_TARGET_VARIANT
    )

    # Group per-cell entries by N alone — T3 rolls up L, Q, backend, seed into a single row per N.
    per_n_realized: dict[int, list[float]] = {}
    per_n_analytical: dict[int, list[float]] = {}
    per_n_sparsity: dict[int, list[float]] = {}
    all_ns: set[int] = set()

    for key, arms in per_cell.items():
        n = int(key[1])
        all_ns.add(n)
        if _FULL_VARIANT in arms and _TARGET_VARIANT in arms and arms[_TARGET_VARIANT] > 0:
            per_n_realized.setdefault(n, []).append(arms[_FULL_VARIANT] / arms[_TARGET_VARIANT])
    for key, val in analytical.items():
        n = int(key[1])
        all_ns.add(n)
        per_n_analytical.setdefault(n, []).append(val)
    for key, val in sparsity.items():
        n = int(key[1])
        all_ns.add(n)
        per_n_sparsity.setdefault(n, []).append(val)

    lines = [
        "| N | analytical | realized (median) | realization ratio | sparsity |",
        "|---|---|---|---|---|",
    ]
    for n in sorted(all_ns):
        analytical_str = _fmt_speedup(_maybe_median(per_n_analytical.get(n)))
        realized_val = _maybe_median(per_n_realized.get(n))
        realized_str = _fmt_speedup(realized_val)
        analytical_val = _maybe_median(per_n_analytical.get(n))
        if realized_val is not None and analytical_val is not None and analytical_val > 0:
            ratio_str = f"{realized_val / analytical_val:.3f}"
        else:
            ratio_str = "n/a"
        sparsity_str = _fmt_frac(_maybe_median(per_n_sparsity.get(n)))
        lines.append(
            f"| {n} | {analytical_str} | {realized_str} | {ratio_str} | {sparsity_str} |"
        )
    return "\n".join(lines) + "\n" + render_provenance_footer(
        store, _phase3_backing_rows(store)
    )


def _maybe_median(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(median(values))


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _fmt_frac(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


# ---- T_perf_* renderers (EXP-3 / #73) -------------------------------------------------------
#
# Each renderer emits one markdown document with a `## L=<L>, backend=<backend>` sub-block for
# every (L, backend) pair present in the store. Within each sub-block, rows are masks (full /
# ddi_qf / blind, plus anything else that appears, sorted) and columns are the fixed N order
# 1, 3, 5, 9, 13, 25. Cells are the metric's mean over the seed(s) at that (model, N, L, backend).
# Two derived metrics — realized_speedup and realization_ratio — are computed at report time by
# pairing each variant row with the matching `full` row at the same (model, N, L, backend, seed).

_PERF_N_COLUMNS: tuple[int, ...] = (1, 3, 5, 9, 13, 25)
_PERF_PREFERRED_MASKS: tuple[str, ...] = ("full", "ddi_qf", "blind")
_PERF_EM_DASH = "—"


def _perf_mask_order(masks: set[str]) -> list[str]:
    ordered = [m for m in _PERF_PREFERRED_MASKS if m in masks]
    extras = sorted(m for m in masks if m not in _PERF_PREFERRED_MASKS)
    return ordered + extras


def _collect_perf_rows(
    store: Path,
) -> dict[tuple[str, str, int, int, str, int], dict[str, float]]:
    """Group all phase3_timing values by (model, variant, N, L, backend, seed).

    Returns a mapping cell -> {metric_short: value}. Metric shortnames are the six the T_perf
    family renders: ttft, tpot, sparsity, analytical_speedup, realized_speedup,
    realization_ratio. The last two are derived (see _apply_derived_metrics).
    """
    per_cell: dict[tuple[str, str, int, int, str, int], dict[str, float]] = {}
    for row in iter_rows(store):
        parsed = _parse(row.get("run_id", ""))
        if parsed is None:
            continue
        metric = row.get("metric")
        # Map store metrics to the short names we render. `ftt_median_ms` in the store is what
        # the paper calls TTFT (time-to-first-token) — kept for backwards compatibility.
        short: str | None = None
        if metric == "ftt_median_ms":
            short = "ttft"
        elif metric == "tpot_median_ms":
            short = "tpot"
        elif metric == "mask_sparsity":
            short = "sparsity"
        elif metric == "analytical_speedup":
            short = "analytical_speedup"
        else:
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        seed = int((row.get("provenance") or {}).get("seed", 0))
        key = (
            str(parsed["model"]),
            str(parsed["variant_tag"]),
            int(parsed["N"]),
            int(parsed["L"]),
            str(parsed["backend"]),
            seed,
        )
        per_cell.setdefault(key, {})[short] = value
    return per_cell


def _apply_derived_metrics(
    per_cell: dict[tuple[str, str, int, int, str, int], dict[str, float]],
) -> None:
    """Fill realized_speedup / realization_ratio in-place by pairing variants with `full`.

    WHY the pairing is per-seed: full baseline TTFT varies run-to-run (kernel timing noise) and
    the analytical ratio only makes sense against the same-seed full to hold hardware constant.
    """
    # (model, N, L, backend, seed) -> full TTFT
    full_ttft: dict[tuple[str, int, int, str, int], float] = {}
    for (model, variant, n, l_, backend, seed), metrics in per_cell.items():
        if variant == _FULL_VARIANT and "ttft" in metrics:
            full_ttft[(model, n, l_, backend, seed)] = metrics["ttft"]

    for (model, variant, n, l_, backend, seed), metrics in per_cell.items():
        if variant == _FULL_VARIANT:
            # By definition full realizes exactly analytical=1.0, so both derived metrics are 1.0.
            metrics["realized_speedup"] = 1.0
            metrics["realization_ratio"] = 1.0
            # `full` also has sparsity 0 by definition; store it so cells render as 0.000 even
            # when the runner did not emit a mask_sparsity row for the full variant.
            metrics.setdefault("sparsity", 0.0)
            metrics.setdefault("analytical_speedup", 1.0)
            continue
        base = full_ttft.get((model, n, l_, backend, seed))
        variant_ttft = metrics.get("ttft")
        if base is not None and variant_ttft is not None and variant_ttft > 0:
            realized = base / variant_ttft
            metrics["realized_speedup"] = realized
            analytical = metrics.get("analytical_speedup")
            if analytical is not None and analytical > 0:
                metrics["realization_ratio"] = realized / analytical


def _fmt_perf_cell(metric: str, value: float | None) -> str:
    if value is None:
        return _PERF_EM_DASH
    if metric in ("ttft", "tpot"):
        return f"{value:.2f}"
    if metric in ("analytical_speedup", "realized_speedup"):
        return f"{value:.2f}x"
    if metric == "realization_ratio":
        return f"{value:.3f}"
    if metric == "sparsity":
        return f"{value:.3f}"
    return f"{value:.4f}"


_PERF_METRIC_LABEL: dict[str, str] = {
    "ttft": "TTFT (median ms)",
    "tpot": "TPOT (median ms)",
    "sparsity": "mask sparsity",
    "analytical_speedup": "analytical speedup",
    "realized_speedup": "realized speedup",
    "realization_ratio": "realization ratio",
}


def _render_perf_metric(store: Path, metric_short: str) -> str:
    """Emit one markdown document for a single perf metric across all (model, L, backend)."""
    if not _any_phase3_rows(store):
        raise ValueError(
            f"T_perf_{metric_short}: store has no phase3_timing rows. "
            "Run `make timing-N` (or the equivalent Phase 3 sweep) first."
        )

    per_cell = _collect_perf_rows(store)
    _apply_derived_metrics(per_cell)

    # Aggregate values per (model, L, backend, mask, N) by mean across seeds. Track masks
    # separately so a variant with only derived metrics missing (e.g. no `full` to pair with)
    # still gets a row of em-dashes rather than vanishing from the table.
    agg: dict[tuple[str, int, str, str, int], list[float]] = {}
    all_masks: dict[tuple[str, int, str], set[str]] = {}
    for (model, variant, n, l_, backend, _seed), metrics in per_cell.items():
        if n not in _PERF_N_COLUMNS:
            # Ns outside the fixed column set are not surfaced in these tables.
            continue
        all_masks.setdefault((model, l_, backend), set()).add(variant)
        val = metrics.get(metric_short)
        if val is None:
            continue
        agg.setdefault((model, l_, backend, variant, n), []).append(val)

    metric_label = _PERF_METRIC_LABEL[metric_short]
    header_cols = " | ".join(f"N={n}" for n in _PERF_N_COLUMNS)
    # +1 for the leading `mask` column; wrapper pipes are added at write time.
    header_sep = "|".join(["---"] * (len(_PERF_N_COLUMNS) + 1))

    blocks: list[str] = [f"# T_perf_{metric_short} — {metric_label}"]

    triples = sorted(all_masks)
    for (model, l_, backend) in triples:
        masks = all_masks.get((model, l_, backend), set())
        lines = [
            f"## {model} — L={l_}, backend={backend}",
            "",
            f"| mask | {header_cols} |",
            f"|{header_sep}|",
        ]
        for mask in _perf_mask_order(masks):
            cells = []
            for n in _PERF_N_COLUMNS:
                vals = agg.get((model, l_, backend, mask, n))
                if not vals:
                    cells.append(_PERF_EM_DASH)
                else:
                    cells.append(_fmt_perf_cell(metric_short, sum(vals) / len(vals)))
            lines.append(f"| {mask} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))

    # Single footer at the bottom — the (model, L, backend) sub-blocks all draw from the
    # same phase3_timing slice, so per-block stamping would duplicate identical metadata.
    return "\n\n".join(blocks) + "\n" + render_provenance_footer(
        store, _phase3_backing_rows(store)
    )


@register("T_perf_ttft")
def _t_perf_ttft(store: Path) -> str:
    return _render_perf_metric(store, "ttft")


@register("T_perf_tpot")
def _t_perf_tpot(store: Path) -> str:
    return _render_perf_metric(store, "tpot")


@register("T_perf_sparsity")
def _t_perf_sparsity(store: Path) -> str:
    return _render_perf_metric(store, "sparsity")


@register("T_perf_analytical")
def _t_perf_analytical(store: Path) -> str:
    return _render_perf_metric(store, "analytical_speedup")


@register("T_perf_realized")
def _t_perf_realized(store: Path) -> str:
    return _render_perf_metric(store, "realized_speedup")


@register("T_perf_realization")
def _t_perf_realization(store: Path) -> str:
    return _render_perf_metric(store, "realization_ratio")


@register("F3")
def _f3_realization_curve(store: Path, output: Path) -> None:
    """Realization ratio (realized / analytical) vs N, one line per L."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    if not _any_phase3_rows(store):
        raise ValueError(
            "F3: store has no phase3_timing rows. Run `make timing-N` (or the equivalent Phase 3 "
            "sweep) first."
        )

    per_cell = _collect_ftt_by_cell(store, target_variant=_TARGET_VARIANT)
    analytical = _collect_scalar_by_cell(
        store, metric="analytical_speedup", variant_tag=_TARGET_VARIANT
    )

    # (L, N) -> list of realization ratios (one per seed/backend cell).
    ln_ratios: dict[tuple[int, int], list[float]] = {}
    for key, arms in per_cell.items():
        _model, n, l_, _q, _backend, _seed = key
        if _FULL_VARIANT not in arms or _TARGET_VARIANT not in arms:
            continue
        if arms[_TARGET_VARIANT] <= 0:
            continue
        realized = arms[_FULL_VARIANT] / arms[_TARGET_VARIANT]
        analytical_val = analytical.get(key)
        if analytical_val is None or analytical_val <= 0:
            continue
        ln_ratios.setdefault((int(l_), int(n)), []).append(realized / analytical_val)

    fig, ax = plt.subplots(figsize=(6, 4))
    ls_sorted = sorted({l_ for (l_, _n) in ln_ratios})
    for l_ in ls_sorted:
        xy = sorted(
            (n, float(median(vs))) for (ll, n), vs in ln_ratios.items() if ll == l_
        )
        if not xy:
            continue
        xs = [x for x, _ in xy]
        ys = [y for _, y in xy]
        ax.plot(xs, ys, marker="o", label=f"L={l_}")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (document count)")
    ax.set_ylabel("realization ratio (realized / analytical)")
    ax.set_title("Phase 3.5 realization ratio vs N")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if ls_sorted:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
