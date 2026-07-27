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


@register("F2")
def _f2_bridge_quality(store: Path, output: Path) -> None:
    """Median EM (with IQR) vs bridge width k for the phase2_train_free ddi_bridge_k sweep."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    rows = list(filter_rows(iter_rows(store), metric="em"))
    tagged = []
    for row in rows:
        k = _extract_bridge_k(row.get("run_id", ""))
        if k is None:
            continue
        tagged.append({**row, "k": k})

    if not tagged:
        # Fail loudly: rendering with no data is a bug at the call site.
        raise ValueError(
            "F2: no phase2_train_free rows with variant=ddi_bridge_k found in store. "
            "Run `make sweep-bridge` (with bridge_k overrides) first."
        )

    grouped = group_by(tagged, ["k"])
    xs = sorted(k for (k,) in grouped)
    ys, lo, hi = [], [], []
    for k in xs:
        vals = [r["value"] for r in grouped[(k,)]]
        m, q1, q3 = median_with_iqr(vals)
        ys.append(m)
        lo.append(m - q1)
        hi.append(q3 - m)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bridge width k (tokens)")
    ax.set_ylabel("EM (median ± IQR)")
    ax.set_title("Phase 2.3 bridge curve — quality vs bridge width")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
