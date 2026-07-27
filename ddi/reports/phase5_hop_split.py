"""Phase 5.1 hop-split renderers — T5 (markdown) and F4 (figure) (issue #26).

DDI predicts a specific failure mode: without cross-document attention, any question that
composes information from two documents must resolve inside the dialog region alone. `ddi_qf`
should therefore degrade specifically on multi-hop examples relative to `full`, while single-hop
examples should be near-parity. T5/F4 measure that split by joining each Phase 2 quality row's
`record_id` with the record's `hop_type` label from `ddi.data.load`.

Report-only: no new runner and no new store rows. Both artifacts read `phase2_train_free:*` EM
rows already written by the Phase 2.1 sweep. See RESEARCH_PLAN §8 (Phase 5) and §10 (F4).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Literal

from ddi.data import load
from ddi.hop_labels import hop_type as _hop_type_of
from ddi.report import filter_rows, iter_rows, median_with_iqr, register

_PHASE2_TF_PREFIX = "phase2_train_free:"
_SPLIT: Literal["train", "val", "test"] = "val"
_HOP_SINGLE = "single"
_HOP_MULTI = "multi"
_HOP_UNKNOWN = "unknown"


def _parse(run_id: str) -> tuple[str, str, str, int, str] | None:
    """(model, dataset, variant_tag, seed, record_id) or None if not phase2_train_free.

    run_id shape: `phase2_train_free:{model}:{dataset}:{variant_tag}:seed={seed}:{sha}:{record_id}`.
    HF model names may contain `/` but never `:`, so positional colon splits are safe.
    """
    if not run_id.startswith(_PHASE2_TF_PREFIX):
        return None
    parts = run_id.split(":")
    if len(parts) < 7:
        return None
    seed_seg = parts[4]
    if not seed_seg.startswith("seed="):
        return None
    try:
        seed = int(seed_seg[len("seed="):])
    except ValueError:
        return None
    model, dataset, variant_tag = parts[1], parts[2], parts[3]
    # Rejoin remaining segments in case a record_id itself contains a colon.
    record_id = ":".join(parts[6:])
    return model, dataset, variant_tag, seed, record_id


def _hop_label_lookup(
    dataset: str, seed: int, cache: dict[tuple[str, int], dict[str, str]]
) -> dict[str, str]:
    """Return {record_id -> hop_type} for (dataset, seed), lazily populated.

    WHY the cache: `ddi.data.load` re-materializes the full split on each call (deterministic
    per seed but non-trivial to build). We may see thousands of Phase 2 EM rows for the same
    (dataset, seed) — reloading the dataset once per row would dominate the renderer's cost.
    """
    key = (dataset, seed)
    if key not in cache:
        try:
            records = list(load(dataset, _SPLIT, seed))
        except KeyError:
            # Unknown dataset name in a stored run_id: fall back to unknown for every record.
            cache[key] = {}
        else:
            cache[key] = {rec["id"]: _hop_type_of(rec) for rec in records}
    return cache[key]


def _any_phase2_rows(store: Path) -> bool:
    for row in iter_rows(store):
        if str(row.get("run_id", "")).startswith(_PHASE2_TF_PREFIX):
            return True
    return False


def _collect(
    store: Path,
) -> tuple[
    dict[tuple[str, str, int, str], list[float]],
    dict[str, int],
]:
    """Group EM rows into {(dataset, variant, seed, hop) -> [em_values]} plus a hop tally.

    The tally reports how many rows fell into each hop bucket across all cells, so the caller
    can note if any dropped to "unknown".
    """
    lookup_cache: dict[tuple[str, int], dict[str, str]] = {}
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    hop_counts: dict[str, int] = defaultdict(int)

    for row in filter_rows(iter_rows(store), metric="em"):
        parsed = _parse(row.get("run_id", ""))
        if parsed is None:
            continue
        _model, dataset, variant, seed, record_id = parsed
        labels = _hop_label_lookup(dataset, seed, lookup_cache)
        hop = labels.get(record_id, _HOP_UNKNOWN)
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[(dataset, variant, seed, hop)].append(val)
        hop_counts[hop] += 1

    return grouped, hop_counts


def _per_seed_means(
    grouped: dict[tuple[str, str, int, str], list[float]],
) -> dict[tuple[str, str, str], list[float]]:
    """Collapse per-record values into per-seed means keyed by (dataset, variant, hop)."""
    per_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (dataset, variant, _seed, hop), values in grouped.items():
        if not values:
            continue
        per_seed[(dataset, variant, hop)].append(sum(values) / len(values))
    return per_seed


def _median_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    return float(median(values))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{a - b:+.4f}"


@register("T5")
def _t5_hop_split(store: Path) -> str:
    """Markdown table: per (dataset, variant) row, EM split by hop_type plus the Δ column."""
    if not _any_phase2_rows(store):
        raise ValueError(
            "T5: store has no phase2_train_free rows. Run the Phase 2.1 train-free sweep first."
        )

    grouped, hop_counts = _collect(store)
    per_seed = _per_seed_means(grouped)

    # A cell is (dataset, variant); columns are hop buckets plus the "all" and Δ derived cells.
    cells: set[tuple[str, str]] = {(dataset, variant) for (dataset, variant, _hop) in per_seed}

    # Compute the "all" bucket by pooling every row across hops back to per-seed means.
    all_per_seed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (dataset, variant, seed, _hop), values in grouped.items():
        all_per_seed[(dataset, variant, seed)].extend(values)
    all_medians: dict[tuple[str, str], float | None] = {}
    for (dataset, variant) in cells:
        seed_means = [
            sum(vals) / len(vals)
            for (d, v, _seed), vals in all_per_seed.items()
            if d == dataset and v == variant and vals
        ]
        all_medians[(dataset, variant)] = _median_or_none(seed_means)

    include_unknown = hop_counts.get(_HOP_UNKNOWN, 0) > 0

    header = ["dataset", "variant", "EM (single)", "EM (multi)", "EM (all)", "Δ EM (multi − single)"]
    if include_unknown:
        header.insert(5, "EM (unknown)")
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for (dataset, variant) in sorted(cells):
        single_med = _median_or_none(per_seed.get((dataset, variant, _HOP_SINGLE)))
        multi_med = _median_or_none(per_seed.get((dataset, variant, _HOP_MULTI)))
        row = [
            dataset,
            variant,
            _fmt(single_med),
            _fmt(multi_med),
            _fmt(all_medians[(dataset, variant)]),
        ]
        if include_unknown:
            unknown_med = _median_or_none(per_seed.get((dataset, variant, _HOP_UNKNOWN)))
            row.append(_fmt(unknown_med))
        row.append(_fmt_delta(multi_med, single_med))
        lines.append("| " + " | ".join(row) + " |")

    out = "\n".join(lines) + "\n"
    if include_unknown:
        out += (
            f"<!-- {hop_counts[_HOP_UNKNOWN]} EM row(s) dropped to hop_type=unknown "
            "(record id missing from loader or label was 'unknown') -->\n"
        )
    return out


@register("F4")
def _f4_hop_split(store: Path, output: Path) -> None:
    """Grouped bar chart: three bars per (dataset, variant) cell — single, multi, all."""
    import matplotlib.pyplot as plt  # lazy: matplotlib is an optional [report] extra

    if not _any_phase2_rows(store):
        raise ValueError(
            "F4: store has no phase2_train_free rows. Run the Phase 2.1 train-free sweep first."
        )

    grouped, _hop_counts = _collect(store)
    per_seed = _per_seed_means(grouped)
    all_per_seed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (dataset, variant, seed, _hop), values in grouped.items():
        all_per_seed[(dataset, variant, seed)].extend(values)

    cells = sorted({(dataset, variant) for (dataset, variant, _hop) in per_seed})
    if not cells:
        raise ValueError(
            "F4: phase2_train_free rows present but none carry metric='em' — nothing to plot."
        )

    labels = [f"{d}\n{v}" for (d, v) in cells]
    hops = [_HOP_SINGLE, _HOP_MULTI, "all"]

    # Median + IQR error bars for each (cell, hop) bucket. Missing buckets get NaN so the bar
    # simply doesn't render (matplotlib skips NaN bars cleanly).
    import math

    heights: dict[str, list[float]] = {h: [] for h in hops}
    lo_err: dict[str, list[float]] = {h: [] for h in hops}
    hi_err: dict[str, list[float]] = {h: [] for h in hops}

    for (dataset, variant) in cells:
        for hop in hops:
            if hop == "all":
                seed_means = [
                    sum(vals) / len(vals)
                    for (d, v, _seed), vals in all_per_seed.items()
                    if d == dataset and v == variant and vals
                ]
            else:
                seed_means = per_seed.get((dataset, variant, hop), [])
            if not seed_means:
                heights[hop].append(math.nan)
                lo_err[hop].append(0.0)
                hi_err[hop].append(0.0)
                continue
            if len(seed_means) == 1:
                heights[hop].append(float(seed_means[0]))
                lo_err[hop].append(0.0)
                hi_err[hop].append(0.0)
                continue
            m, q1, q3 = median_with_iqr(seed_means)
            heights[hop].append(m)
            lo_err[hop].append(m - q1)
            hi_err[hop].append(q3 - m)

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(cells)), 4))
    n_cells = len(cells)
    bar_w = 0.25
    xs = list(range(n_cells))
    offsets = {_HOP_SINGLE: -bar_w, _HOP_MULTI: 0.0, "all": bar_w}
    for hop in hops:
        positions = [x + offsets[hop] for x in xs]
        ax.bar(
            positions,
            heights[hop],
            width=bar_w,
            label=hop,
            yerr=[lo_err[hop], hi_err[hop]],
            capsize=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("EM (median ± IQR across seeds)")
    ax.set_title("Phase 5.1 — quality by hop type")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
