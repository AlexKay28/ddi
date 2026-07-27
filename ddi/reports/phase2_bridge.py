"""Phase 2.3 bridge-width curve renderers (F2 quality; F6 Pareto pairs with Phase 3.2)."""

from __future__ import annotations

from pathlib import Path

from ddi.report import filter_rows, group_by, iter_rows, median_with_iqr, register

_PHASE2_TF_PREFIX = "phase2_train_free:"
_VARIANT_TAG = "ddi_bridge_k"


def _extract_bridge_k(run_id: str) -> int | None:
    # run_id shape from phase2_train_free: "phase2_train_free:{model}:{dataset}:{variant_tag}:seed=...:{sha}:{record_id}"
    parts = run_id.split(":")
    if len(parts) < 4 or not parts[0] + ":" == _PHASE2_TF_PREFIX:
        return None
    tag = parts[3]
    prefix = f"{_VARIANT_TAG}_k"
    if not tag.startswith(prefix):
        return None
    try:
        return int(tag[len(prefix):])
    except ValueError:
        return None


# Plotted in this fixed order so the categorical hues stay CVD-safe (see palette slots 1-3).
# EM alone saturates at (or near) 0 on some sweeps (composition/#36) — F1/presence carry the
# signal there, so all three are drawn together rather than EM alone understating the curve.
_METRICS = ("em", "f1", "presence")
_METRIC_COLOR = {"em": "#2a78d6", "f1": "#eb6834", "presence": "#1baf7a"}
_METRIC_LABEL = {"em": "EM", "f1": "F1", "presence": "presence"}
_MUTED_INK = "#52514e"


@register("F2")
def _f2_bridge_quality(store: Path, output: Path) -> None:
    """Median EM/F1/presence (with IQR) vs bridge width k for the ddi_bridge_k sweep."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    by_metric: dict[str, list[dict]] = {m: [] for m in _METRICS}
    for metric in _METRICS:
        for row in filter_rows(iter_rows(store), metric=metric):
            k = _extract_bridge_k(row.get("run_id", ""))
            if k is None:
                continue
            by_metric[metric].append({**row, "k": k})

    if not any(by_metric.values()):
        # Fail loudly: rendering with no data is a bug at the call site.
        raise ValueError(
            "F2: no phase2_train_free rows with variant=ddi_bridge_k found in store. "
            "Run `make sweep-bridge` (with bridge_k overrides) first."
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    for metric in _METRICS:
        tagged = by_metric[metric]
        if not tagged:
            continue
        grouped = group_by(tagged, ["k"])
        xs = sorted(k for (k,) in grouped)
        ys, lo, hi = [], [], []
        for k in xs:
            vals = [r["value"] for r in grouped[(k,)]]
            m, q1, q3 = median_with_iqr(vals)
            ys.append(m)
            lo.append(m - q1)
            hi.append(q3 - m)
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            marker="o",
            markersize=6,
            linewidth=1.6,
            capsize=3,
            color=_METRIC_COLOR[metric],
            label=_METRIC_LABEL[metric],
        )
        # Direct end-label in neutral ink — color already carries series identity via the line.
        ax.annotate(
            f"{ys[-1]:.2f}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=8,
            color=_MUTED_INK,
            va="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("bridge width k (tokens)")
    ax.set_ylabel("score (median ± IQR)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Phase 2.3 bridge curve — quality vs bridge width")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
