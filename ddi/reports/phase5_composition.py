"""Phase 5.3 composition-probe renderers — T6 (table) and F7 (curve). Issue #28.

Both consume `phase5_composition:*` rows written by `ddi.runners.phase5_composition`. Run IDs
encode the probe knobs positionally so parsing does not need to reach into provenance:

    phase5_composition:{model}:{dataset}:{variant_tag}:{layout}:d{distance}:p{pair_position}:seed=...:{sha}:{record_id}

T6 tabulates EM (mean, median, IQR) grouped by (variant, layout, distance). F7 plots
median EM vs distance, one line per (variant, layout) — the picture that shows where
`ddi_qf` starts to fail as the gold pair pulls apart.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from ddi.report import filter_rows, iter_rows, median_with_iqr, register

_PHASE5_PREFIX = "phase5_composition:"


def _parse_run_id(run_id: str) -> dict[str, str | int] | None:
    """Parse a phase5_composition run_id.

    WHY strict positional parsing: the runner is the sole writer of these IDs and encodes
    knobs positionally so downstream readers do not need to open provenance. Any deviation
    is a runner bug — reject rather than silently mis-slice.
    """
    if not run_id.startswith(_PHASE5_PREFIX):
        return None
    parts = run_id.split(":")
    # Shape: [phase5_composition, model, dataset, variant_tag, layout, d{N}, p{pos}, seed=..., sha, record_id]
    if len(parts) < 10:
        return None
    layout = parts[4]
    d_seg = parts[5]
    p_seg = parts[6]
    if not d_seg.startswith("d") or not p_seg.startswith("p"):
        return None
    try:
        distance = int(d_seg[1:])
    except ValueError:
        return None
    pair_position = p_seg[1:]
    return {
        "model": parts[1],
        "dataset": parts[2],
        "variant_tag": parts[3],
        "layout": layout,
        "distance": distance,
        "pair_position": pair_position,
    }


def _collect_em_rows(store: Path) -> list[dict]:
    """Return the parsed subset of `em` rows tagged with phase5_composition knobs."""
    out: list[dict] = []
    for row in filter_rows(iter_rows(store), metric="em"):
        parsed = _parse_run_id(row.get("run_id", ""))
        if parsed is None:
            continue
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({**parsed, "value": value})
    return out


@register("T6")
def _t6_composition_table(store: Path) -> str:
    """Markdown: EM (mean/median/IQR) per (variant, layout, distance)."""
    rows = _collect_em_rows(store)
    if not rows:
        raise ValueError(
            "T6: no phase5_composition EM rows in store. Run `phase5_composition` first."
        )

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["variant_tag"]), str(row["layout"]), int(row["distance"]))
        grouped[key].append(float(row["value"]))

    lines = [
        "| variant | layout | distance | EM (mean) | EM (median) | EM (IQR low) | EM (IQR high) | n_records |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(grouped):
        vals = grouped[key]
        m = float(mean(vals))
        med, q1, q3 = median_with_iqr(vals)
        variant, layout, distance = key
        lines.append(
            f"| {variant} | {layout} | {distance} | {m:.4f} | {med:.4f} | "
            f"{q1:.4f} | {q3:.4f} | {len(vals)} |"
        )
    return "\n".join(lines) + "\n"


@register("F7")
def _f7_composition_curve(store: Path, output: Path) -> None:
    """Median EM vs distance; one line per (variant, layout) combo."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    rows = _collect_em_rows(store)
    if not rows:
        raise ValueError(
            "F7: no phase5_composition EM rows in store. Run `phase5_composition` first."
        )

    # (variant, layout) -> distance -> list of EM values
    per_series: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        series = (str(row["variant_tag"]), str(row["layout"]))
        per_series[series][int(row["distance"])].append(float(row["value"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    for series in sorted(per_series):
        by_distance = per_series[series]
        xs = sorted(by_distance)
        ys = [float(median(by_distance[d])) for d in xs]
        variant, layout = series
        ax.plot(xs, ys, marker="o", label=f"{variant}/{layout}")

    ax.set_xlabel("distance (distractors between doc_A and doc_B)")
    ax.set_ylabel("EM (median)")
    ax.set_title("Phase 5.3 — composition boundary: EM vs pair distance")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
