"""Phase 2.3 x Phase 3.2 Pareto frontier (F6) — the paper centrepiece (issue #21).

F6 pairs Phase 2.3 bridge-width EM rows with Phase 3.2 timing rows at the same `bridge_k`:
x-axis is realized speedup (full vs bridge_k prefill), y-axis is median EM across seeds. The
`full`-baseline EM and the `full − Δ` parity band (from `ddi.delta`) are drawn as dashed guides.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from ddi.report import filter_rows, iter_rows, register

_PHASE2_TF_PREFIX = "phase2_train_free:"
_PHASE3_PREFIX = "phase3_timing:"
_BRIDGE_TAG_PREFIX = "ddi_bridge_k_k"
_FULL_TAG = "full"


def _extract_bridge_k(variant_tag: str) -> int | None:
    """`ddi_bridge_k_k4` -> 4; anything else -> None."""
    if not variant_tag.startswith(_BRIDGE_TAG_PREFIX):
        return None
    try:
        return int(variant_tag[len(_BRIDGE_TAG_PREFIX):])
    except ValueError:
        return None


def _parse_phase2(run_id: str) -> dict[str, str] | None:
    # phase2_train_free:{model}:{dataset}:{variant_tag}:seed=...:{sha}:{record_id}
    if not run_id.startswith(_PHASE2_TF_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 7:
        return None
    return {"model": parts[1], "dataset": parts[2], "variant_tag": parts[3]}


def _parse_phase3(run_id: str) -> dict[str, str] | None:
    # phase3_timing:{model}:N={N}:L={L}:Q={Q}:{variant_tag}:{backend}:seed=...:{sha}
    if not run_id.startswith(_PHASE3_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 9:
        return None
    return {"model": parts[1], "variant_tag": parts[5]}


@register("F6")
def _f6_pareto(store: Path, output: Path) -> None:
    """Quality/speed Pareto frontier at matching bridge_k."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    all_rows = list(iter_rows(store))

    # Phase 2.3: median EM per K across seeds, plus full-baseline EM for the horizontal line.
    em_rows = [r for r in all_rows if r.get("metric") == "em"]
    per_k_em: dict[int, list[float]] = {}
    full_em: list[float] = []
    for row in em_rows:
        parsed = _parse_phase2(row.get("run_id", ""))
        if parsed is None:
            continue
        tag = parsed["variant_tag"]
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if tag == _FULL_TAG:
            full_em.append(val)
            continue
        k = _extract_bridge_k(tag)
        if k is None:
            continue
        per_k_em.setdefault(k, []).append(val)

    # Phase 3.2: ftt_median_ms rows for `full` and each bridge_k, at the sweep shape.
    ftt_rows = [r for r in all_rows if r.get("metric") == "ftt_median_ms"]
    per_k_ftt: dict[int, list[float]] = {}
    full_ftt: list[float] = []
    for row in ftt_rows:
        parsed = _parse_phase3(row.get("run_id", ""))
        if parsed is None:
            continue
        tag = parsed["variant_tag"]
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if tag == _FULL_TAG:
            full_ftt.append(val)
            continue
        k = _extract_bridge_k(tag)
        if k is None:
            continue
        per_k_ftt.setdefault(k, []).append(val)

    # Realized speedup per K needs a full-arm baseline. Without it there is no x coordinate.
    if not full_ftt:
        raise ValueError(
            "F6: no (bridge, timing) pairs found — run sweep-bridge and timing-N first"
        )
    full_ftt_med = float(median(full_ftt))

    points: list[tuple[int, float, float]] = []  # (k, realized_speedup, median_em)
    for k in sorted(set(per_k_em) & set(per_k_ftt)):
        em_med = float(median(per_k_em[k]))
        ftt_med = float(median(per_k_ftt[k]))
        if ftt_med <= 0:
            continue
        realized = full_ftt_med / ftt_med
        points.append((k, realized, em_med))

    if not points:
        raise ValueError(
            "F6: no (bridge, timing) pairs found — run sweep-bridge and timing-N first"
        )

    # Delta comes from ddi.delta writes: rows carry metric=delta_em; pick the median if several.
    delta_values = [
        float(r["value"]) for r in all_rows if r.get("metric") == "delta_em"
    ]
    delta_em: float | None = float(median(delta_values)) if delta_values else None

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.plot(xs, ys, marker="o", linestyle="-")
    for k, x, y in points:
        ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(5, 5))

    if full_em:
        base_em = float(median(full_em))
        ax.axhline(base_em, linestyle="--", color="grey", label="full baseline EM")
        if delta_em is not None:
            ax.axhline(
                base_em - delta_em,
                linestyle="--",
                color="red",
                label=f"full − Δ ({delta_em:.4f})",
            )
        else:
            # Note the missing Δ inside the figure so a reader knows it was not silently omitted.
            ax.text(
                0.02,
                0.02,
                "Δ band absent: no delta_em in store",
                transform=ax.transAxes,
                fontsize=8,
                color="grey",
            )

    ax.set_xlabel("realized speedup (full FTT / bridge_k FTT)")
    ax.set_ylabel("EM (median across seeds)")
    ax.set_title("F6: quality/speed Pareto (Phase 2.3 x Phase 3.2)")
    ax.grid(True, ls="--", alpha=0.3)
    if full_em:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
