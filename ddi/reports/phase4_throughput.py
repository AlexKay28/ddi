"""Phase 4.4 throughput-under-reuse renderer — F5 (issue #25).

Reads `phase4_throughput:*` rows from the store, joins each run's `qps` with its
`hit_rate`, and plots QPS vs hit rate as one series per mode (ddi_cache, ddi_recompute,
full_causal). Both non-cache modes have hit_rate == 0.0 by construction, so they land as
single points on the y-axis — the contrast with ddi_cache's curve is the co-headline.

See RESEARCH_PLAN §10 F5; issue #25.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ddi.report import filter_rows, iter_rows, register

_PHASE4_TP_PREFIX = "phase4_throughput:"
_MODES = ("ddi_cache", "ddi_recompute", "full_causal")


def _parse_run_id(run_id: str) -> tuple[str, str] | None:
    """Return (mode, reuse_tag) from a phase4_throughput run_id, or None if not parseable.

    run_id shape: `phase4_throughput:{model}:{mode}:{layout}:reuse={reuse_tag}:seed=...:{sha}`
    Per-request rows append `:req{i}` — we strip that first.
    """
    if not run_id.startswith(_PHASE4_TP_PREFIX):
        return None
    parts = run_id.split(":")
    # Base run_id has 7 colon-parts (prefix, model, mode, layout, reuse=..., seed=..., sha).
    # Per-request rows have 8 (extra :req{i} suffix); we only care about the base for F5.
    if len(parts) < 7:
        return None
    mode = parts[2]
    reuse = parts[4]
    if mode not in _MODES:
        return None
    return mode, reuse


def _base_run_id(run_id: str) -> str:
    """Strip `:req{i}` suffix so per-request rows collapse to their parent run.

    Never called for base rows (we only feed it strings we know carry the suffix), but
    accepting either shape keeps the caller's join loop simple.
    """
    if ":req" in run_id:
        return run_id.rsplit(":req", 1)[0]
    return run_id


@register("F5")
def _f5_throughput_vs_reuse(store: Path, output: Path) -> None:
    """Plot QPS vs hit rate, one series per mode. Log-y so the ddi_cache curve reads."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    # Collect per-run qps and hit_rate. We index by the *base* run_id (per-request rows
    # exist but do not carry qps/hit_rate themselves).
    qps_by_run: dict[str, float] = {}
    hit_by_run: dict[str, float] = {}
    mode_by_run: dict[str, str] = {}

    for row in filter_rows(iter_rows(store), metric="qps"):
        rid = row.get("run_id", "")
        parsed = _parse_run_id(rid)
        if parsed is None:
            continue
        mode_by_run[rid] = parsed[0]
        qps_by_run[rid] = float(row["value"])

    for row in filter_rows(iter_rows(store), metric="hit_rate"):
        rid = row.get("run_id", "")
        parsed = _parse_run_id(rid)
        if parsed is None:
            continue
        hit_by_run[rid] = float(row["value"])

    joined: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rid, qps in qps_by_run.items():
        if rid not in hit_by_run:
            continue
        mode = mode_by_run[rid]
        joined[mode].append((hit_by_run[rid], qps))

    if not joined:
        # Fail loudly: rendering with no data is a bug at the call site.
        raise ValueError(
            "F5: no phase4_throughput rows with both `qps` and `hit_rate` found in store. "
            "Run `phase4_throughput` under all three modes first."
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"ddi_cache": "o", "ddi_recompute": "s", "full_causal": "^"}
    for mode in _MODES:
        pts = sorted(joined.get(mode, []))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=markers.get(mode, "o"), label=mode, linewidth=1.5)
    ax.set_xlabel("cache hit rate")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")
    ax.set_title("Phase 4.4 — throughput vs document reuse rate")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
