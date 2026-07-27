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
    return "\n".join(lines) + "\n"


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
