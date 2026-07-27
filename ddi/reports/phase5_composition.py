"""Phase 5.3 composition-probe renderers — T6 (table) and F7 (curve). Issue #28.

Both consume `phase5_composition:*` rows written by `ddi.runners.phase5_composition`. Run IDs
encode the probe knobs positionally so parsing does not need to reach into provenance:

    phase5_composition:{model}:{dataset}:{variant_tag}:{layout}:d{distance}:p{pair_position}:seed=...:{sha}:{record_id}

T6 tabulates EM/F1/presence (mean, median, IQR) grouped by (variant, layout, distance). F7 plots
median EM/F1/presence vs distance, one panel per metric and one line per (variant, layout) — the
picture that shows where `ddi_qf` starts to fail as the gold pair pulls apart.

EM saturates at (or near) 0 on `composition` regardless of variant/layout/distance (see #35, #36)
— F1/presence carry the signal there, so both renderers report all three metrics that are present
in the store rather than EM alone understating (or hiding) the boundary.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from ddi.report import filter_rows, iter_rows, median_with_iqr, register

_PHASE5_PREFIX = "phase5_composition:"
_METRICS = ("em", "f1", "presence")
_METRIC_LABEL = {"em": "EM", "f1": "F1", "presence": "presence"}
# Fixed categorical order (see dataviz skill palette slots 1-3); linestyle carries layout so two
# variants stay within the all-pairs-validated first three slots even with layout as a 2nd factor.
_VARIANT_COLOR_ORDER = ("full", "ddi_qf", "ddi_strict", "blind")
_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_LAYOUT_STYLE = {"sequential": "-", "restart": "--"}


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


def _collect_metric_rows(store: Path, metric: str) -> list[dict]:
    """Return the parsed subset of `metric` rows tagged with phase5_composition knobs."""
    out: list[dict] = []
    for row in filter_rows(iter_rows(store), metric=metric):
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
    """Markdown: EM/F1/presence (mean/median/IQR) per (variant, layout, distance)."""
    by_metric = {m: _collect_metric_rows(store, m) for m in _METRICS}
    if not any(by_metric.values()):
        raise ValueError(
            "T6: no phase5_composition EM/F1/presence rows in store. Run `phase5_composition` first."
        )
    present = [m for m in _METRICS if by_metric[m]]

    grouped: dict[str, dict[tuple[str, str, int], list[float]]] = {}
    keys: set[tuple[str, str, int]] = set()
    for m in present:
        g: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in by_metric[m]:
            key = (str(row["variant_tag"]), str(row["layout"]), int(row["distance"]))
            g[key].append(float(row["value"]))
        grouped[m] = g
        keys.update(g.keys())

    header_cols = ["variant", "layout", "distance"]
    for m in present:
        label = _METRIC_LABEL[m]
        header_cols += [f"{label} (mean)", f"{label} (median)", f"{label} (IQR low)", f"{label} (IQR high)"]
    header_cols.append("n_records")
    lines = ["| " + " | ".join(header_cols) + " |", "|" + "---|" * len(header_cols)]

    for key in sorted(keys):
        variant, layout, distance = key
        cells = [variant, layout, str(distance)]
        n_records = 0
        for m in present:
            vals = grouped[m].get(key, [])
            if vals:
                n_records = max(n_records, len(vals))
                mn = float(mean(vals))
                med, q1, q3 = median_with_iqr(vals)
                cells += [f"{mn:.4f}", f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}"]
            else:
                cells += ["n/a", "n/a", "n/a", "n/a"]
        cells.append(str(n_records))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@register("F7")
def _f7_composition_curve(store: Path, output: Path) -> None:
    """Median EM/F1/presence vs distance; one panel per metric, one line per (variant, layout)."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    by_metric = {m: _collect_metric_rows(store, m) for m in _METRICS}
    if not any(by_metric.values()):
        raise ValueError(
            "F7: no phase5_composition EM/F1/presence rows in store. Run `phase5_composition` first."
        )
    present = [m for m in _METRICS if by_metric[m]]

    # metric -> (variant, layout) -> distance -> list of values
    per_metric_series: dict[str, dict[tuple[str, str], dict[int, list[float]]]] = {}
    for m in present:
        per_series: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in by_metric[m]:
            series = (str(row["variant_tag"]), str(row["layout"]))
            per_series[series][int(row["distance"])].append(float(row["value"]))
        per_metric_series[m] = per_series

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)
    axes = axes[0]
    variant_color = {
        v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(_VARIANT_COLOR_ORDER)
    }
    seen_labels: set[str] = set()
    for ax, m in zip(axes, present):
        per_series = per_metric_series[m]
        for series in sorted(per_series):
            variant, layout = series
            by_distance = per_series[series]
            xs = sorted(by_distance)
            ys = [float(median(by_distance[d])) for d in xs]
            label = f"{variant}/{layout}"
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5,
                linewidth=1.6,
                linestyle=_LAYOUT_STYLE.get(layout, "-"),
                color=variant_color.get(variant, "#52514e"),
                label=label if label not in seen_labels else None,
            )
            seen_labels.add(label)
        ax.set_xlabel("distance (distractors between doc_A and doc_B)")
        ax.set_ylabel(f"{_METRIC_LABEL[m]} (median)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(_METRIC_LABEL[m])
        ax.grid(True, ls="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if len(present) > 1:
        for ax in axes[1:]:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Phase 5.3 — composition boundary: score vs pair distance")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
