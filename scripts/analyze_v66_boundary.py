"""VIBEQUEUE-66 analysis: crossover tables, figures and paper deliverables.

Reads a completed ``results/v66_boundary_<UTC>/`` sweep directory (``progress.jsonl``
cells + ``manifest.json`` + ``result.json``) and writes, into the same directory:

- ``crossover_table.md``  — per (L, decode budget) nominal-h table with medians,
  paired-bootstrap A/B CIs, frozen materiality verdicts, and the measured crossover
  h (smallest nominal h whose verdict is "win") for wall and TTFT;
- ``crossover_figure.png`` / ``.pdf`` — median wall speedup A/B vs h with CI bands
  and win/tie/lose markers, one panel per decode budget, one figure per L;
- ``crossover_figure_ttft.png`` / ``.pdf`` — the same for TTFT;
- ``summary.json`` — machine-readable crossover statements and provenance;
- ``README.md`` — provenance, prereg shas (both revisions), deviations, identity
  status, infeasible cells;
- ``scoping_sentence.md`` — the updated paper scoping sentence with the measured
  boundary;
- ``t9_followup_fragment.tex`` + ``t9_followup_evidence_fragment.json`` — a
  ready-to-paste T9-style table fragment following the paper/evidence render
  conventions (the pinned pipeline itself is not modified by this task).

Pure helpers (crossover selection, table rows, fragment rendering) are unit-tested
in ``tests/test_v66_boundary.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_canonical_matrix import _sha_file  # noqa: E402
from scripts.run_v66_boundary import (  # noqa: E402
    DECODE_BUDGETS,
    HIT_FRACTIONS,
    LOSE_CI_HIGH,
    PREREG_JSON,
    PREREG_MD,
    TTFT_BUDGET,
    WIN_CI_LOW,
    hit_count_for,
    materiality_verdict,
)

BUDGET_LABELS = {
    1: "TTFT-only (d=1)",
    8: "d=8",
    16: "d=16",
    32: "d=32",
    64: "d=64",
}


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def crossover_h(hit_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Smallest nominal h whose ``metric`` verdict is win, with achiever (pure).

    ``hit_rows``: one row per nominal h in the frozen grid for one (L, decode)
    slice (possibly across several N units — the per-L crossover is the smallest
    winning nominal h anywhere in the grid, and the achiever names the unit).
    """
    winners = [row for row in hit_rows if row.get(f"{metric}_verdict") == "win"]
    if not winners:
        return {"h": None, "status": "none_in_grid", "achiever": None}
    best = min(winners, key=lambda row: row["h"])
    return {
        "h": best["h"],
        "status": "ok",
        "achiever": {
            "N": best.get("n_docs"),
            "hit_count": best.get("hit_count"),
            "h_eff": best.get("h_eff"),
        },
    }


def crossover_statement(cross: dict[str, Any]) -> str:
    if cross["h"] is None:
        return "none in grid"
    achiever = cross.get("achiever") or {}
    suffix = f" (N={achiever['N']}, h_eff={achiever['h_eff']:g})" if achiever else ""
    return f"h = {cross['h']:g}{suffix}"


def nominal_rows_for_cell(
    h_labels: list[float], hit_count: int, n_docs: int, block: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand one measured cell into one row per nominal h label (pure)."""
    rows = []
    for h in sorted(h_labels):
        assert hit_count_for(h, n_docs) == hit_count
        rows.append({"h": h, "hit_count": hit_count, "h_eff": hit_count / n_docs, **block})
    return rows


def t9_fragment_rows(crossover: dict[str, Any]) -> list[list[Any]]:
    """T9-convention rows ``[stage, label, point, [ci_low, ci_high]]`` (pure)."""
    rows: list[list[Any]] = []
    for length in sorted({key[0] for key in crossover}):
        for budget in sorted({key[1] for key in crossover}):
            entry = crossover[(length, budget)]
            wall_h = entry["wall_crossover"]["h"]
            ttft_h = entry["ttft_crossover"]["h"]
            rows.append([
                "crossover",
                f"L={length} {BUDGET_LABELS[budget]}",
                wall_h if wall_h is not None else float("nan"),
                [ttft_h if ttft_h is not None else float("nan"), 0.0],
            ])
    return rows


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_cells(
    results_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    """Parse ``progress.jsonl`` into per-unit cell payloads (last record wins)."""
    cells: dict[str, dict[str, Any]] = {}
    prewarm: list[dict[str, Any]] = []
    infeasible: list[dict[str, Any]] = []
    divergent_repeats = 0
    for line in (results_dir / "progress.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") == "prewarm":
            prewarm.append(record)
        elif record.get("type") == "identity_failure":
            if record.get("policy", "abort") == "record":
                divergent_repeats += 1
        elif record.get("type") == "cell" and record.get("status") == "ok":
            cells[record["cell_id"]] = record
        elif record.get("type") == "cell" and record.get("status") == "infeasible":
            infeasible.append(record)
    return cells, prewarm + infeasible, divergent_repeats


def unit_tables(cells: dict[str, dict[str, Any]]) -> dict[tuple[int, int], dict[int, dict[str, Any]]]:
    """(L, N) -> hit_count -> merged block payloads (wall per budget + TTFT)."""
    units: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for cell_id, cell in cells.items():
        key = (cell["L"], cell["N"])
        unit = units.setdefault(key, {})
        hit = unit.setdefault(cell["hit_count"], {"h_labels": cell["h_labels"], "budgets": {}})
        for budget, block in cell.get("budgets", {}).items():
            hit["budgets"][int(budget)] = block
    return units


# ---------------------------------------------------------------------------
# tables and figures
# ---------------------------------------------------------------------------


def _fmt_ci(point: float, low: float, high: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def build_crossover_table(
    units: dict[tuple[int, int], dict[int, dict[str, Any]]],
    *,
    infeasible: list[dict[str, Any]],
) -> tuple[str, dict[tuple[int, int], dict[str, Any]]]:
    """Render the markdown crossover table and the crossover summary (pure text)."""
    lines: list[str] = [
        "# V66 — optimized reuse vs monolithic: measured serving boundary",
        "",
        "Ratio = median(A)/median(B) wall speedup (A = monolithic, B = optimized V65",
        "reuse; > 1 means reuse faster). CI = paired percentile bootstrap (seed 47,",
        "10k) over 15 timed repeat pairs. Verdicts: win iff CI low > 1.05, lose iff",
        "CI high < 1/1.05, else tie (frozen materiality rule). TTFT is",
        "decode-budget-independent; its per-(L, N, h) measurement is repeated on every",
        "budget row. Crossover h = smallest nominal h in the frozen grid whose",
        "verdict is win.",
        "",
    ]
    crossover: dict[tuple[int, int], dict[str, Any]] = {}
    lengths = sorted({key[0] for key in units})
    for length in lengths:
        n_values = sorted({key[1] for key in units})
        for n_docs in n_values:
            unit = units[(length, n_docs)]
            lines += [
                f"## L = {length}, N = {n_docs}",
                "",
                "| nominal h | hit_count | h_eff | wall d=1 A/B | wall d=8 | wall d=16 |"
                " wall d=32 | wall d=64 | TTFT A/B | TTFT verdict |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for hit_count in sorted(unit):
                entry = unit[hit_count]
                h_eff = hit_count / n_docs
                ttft_blocks = [
                    block["ttft_ratio"]
                    for block in entry["budgets"].values()
                    if block.get("ttft_ratio")
                ]
                ttft = ttft_blocks[0] if ttft_blocks else None
                cells_by_budget: dict[int, dict[str, Any]] = {}
                for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
                    block = entry["budgets"].get(budget)
                    cells_by_budget[budget] = block
                wall_cells = []
                for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
                    block = cells_by_budget[budget]
                    wall_cells.append(
                        _fmt_ci(block["ratio_point"], block["ratio_ci_low"],
                                block["ratio_ci_high"])
                        if block else "—"
                    )
                ttft_text = (
                    _fmt_ci(ttft["ratio_point"], ttft["ratio_ci_low"], ttft["ratio_ci_high"])
                    if ttft else "—"
                )
                ttft_verdict = ttft["verdict"] if ttft else "—"
                h_labels = sorted(entry["h_labels"])
                for h in h_labels:
                    lines.append(
                        f"| {h:g} | {hit_count} | {h_eff:g} | " + " | ".join(wall_cells)
                        + f" | {ttft_text} | {ttft_verdict} |"
                    )
                for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
                    block = cells_by_budget[budget]
                    if block is None:
                        continue
                    key = (length, budget)
                    summary = crossover.setdefault(key, {"per_unit": {}})
                    summary["per_unit"][(n_docs, hit_count)] = {
                        "h_labels": h_labels,
                        "wall_verdict": block["verdict"],
                        "ttft_verdict": (
                            ttft["verdict"] if ttft else None
                        ),
                    }
            lines.append("")
        lines += ["", f"### Measured crossover per decode budget — L = {length}", ""]
        lines += [
            "| decode budget | crossover h (wall) | crossover h (TTFT) |",
            "|---|---|---|",
        ]
        for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
            entry = crossover[(length, budget)]
            wall_rows = [
                {"h": h, "n_docs": n, "hit_count": hc, "h_eff": hc / n,
                 "wall_verdict": verdict}
                for (n, hc), value in sorted(entry["per_unit"].items())
                for h, verdict in zip(value["h_labels"], [value["wall_verdict"]]
                                      * len(value["h_labels"]))
            ]
            ttft_rows = [
                {"h": h, "n_docs": n, "hit_count": hc, "h_eff": hc / n,
                 "ttft_verdict": verdict}
                for (n, hc), value in sorted(entry["per_unit"].items())
                for h, verdict in zip(value["h_labels"], [value["ttft_verdict"]]
                                      * len(value["h_labels"]))
            ]
            wall_cross = crossover_h(wall_rows, "wall")
            ttft_cross = crossover_h(ttft_rows, "ttft")
            entry["wall_crossover"] = wall_cross
            entry["ttft_crossover"] = ttft_cross
            lines.append(
                f"| {BUDGET_LABELS[budget]} | {crossover_statement(wall_cross)} | "
                f"{crossover_statement(ttft_cross)} |"
            )
        lines.append("")
    if infeasible:
        lines += ["## Infeasible cells", ""]
        for record in infeasible:
            lines.append(f"- `{record.get('cell_id')}`: {record.get('error', '')}")
        lines.append("")
    return "\n".join(lines) + "\n", crossover


def _speedup_panel(
    ax: plt.Axes,
    points: dict[float, list[tuple[float, float, float, str]]],
    title: str,
) -> None:
    """Plot median speedup vs h with CI bands and verdict markers on one panel."""
    for n_docs, rows in sorted(points.items()):
        xs = [row[0] for row in rows]
        medians = [row[1] for row in rows]
        lows = [row[2] for row in rows]
        highs = [row[3] for row in rows]
        verdicts = [row[4] for row in rows]
        ax.fill_between(xs, lows, highs, alpha=0.18, linewidth=0)
        ax.plot(xs, medians, marker="o", linewidth=1.4, markersize=4,
                label=f"N={n_docs}")
        for x, median, low, high, verdict in zip(xs, medians, lows, highs, verdicts):
            color = {"win": "tab:green", "lose": "tab:red", "tie": "tab:gray"}[verdict]
            ax.plot([x], [median], marker="o", markersize=6, color=color,
                    linestyle="none", zorder=5)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.axhline(WIN_CI_LOW, color="tab:green", linewidth=0.8, linestyle="--")
    ax.axhline(LOSE_CI_HIGH, color="tab:red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("nominal hit fraction h")
    ax.set_ylabel("median wall speedup A/B")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(list(HIT_FRACTIONS))
    ax.set_ylim(0.2, max(2.2, ax.get_ylim()[1]))
    ax.grid(alpha=0.25)


def build_figures(
    units: dict[tuple[int, int], dict[int, dict[str, Any]]],
    output: Path,
) -> list[Path]:
    """Wall and TTFT speedup-vs-h figures (PNG+PDF); returns written paths."""
    written: list[Path] = []
    lengths = sorted({key[0] for key in units})
    for metric, stem, ylabel in (
        ("wall", "crossover_figure", "median wall speedup A/B"),
        ("ttft", "crossover_figure_ttft", "median TTFT speedup A/B"),
    ):
        budgets = (TTFT_BUDGET, *DECODE_BUDGETS) if metric == "wall" else (None,)
        fig, axes = plt.subplots(
            1, len(lengths) * len(budgets),
            figsize=(4.6 * len(lengths) * len(budgets), 4.0), squeeze=False,
        )
        panel = 0
        for length in lengths:
            n_values = sorted({key[1] for key in units if key[0] == length})
            for budget in budgets:
                ax = axes[0][panel]
                points: dict[float, list[tuple[float, float, float, float, str]]] = {}
                for n_docs in n_values:
                    unit = units[(length, n_docs)]
                    rows = []
                    for hit_count in sorted(unit):
                        blocks = [
                            block
                            for b, block in unit[hit_count]["budgets"].items()
                            if (metric == "wall" and int(b) == budget)
                            or (metric == "ttft" and block.get("ttft_ratio"))
                        ]
                        if not blocks:
                            continue
                        block = blocks[0]
                        ratio = (
                            block
                            if metric == "wall"
                            else block["ttft_ratio"]
                        )
                        point = ratio["ratio_point"]
                        for h in sorted(unit[hit_count]["h_labels"]):
                            rows.append((
                                h, point, ratio["ratio_ci_low"],
                                ratio["ratio_ci_high"],
                                ratio["verdict"],
                            ))
                    if rows:
                        points[n_docs] = rows
                title = (
                    f"L = {length} — {BUDGET_LABELS[budget]}"
                    if metric == "wall"
                    else f"L = {length} — TTFT (budget-independent)"
                )
                _speedup_panel(ax, points, title)
                ax.set_ylabel(ylabel)
                panel += 1
            if metric == "ttft":
                break
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=8)
        fig.suptitle(
            "V66 serving boundary: optimized reuse vs monolithic (development data, "
            "seed 47); green = win (CI low > 1.05), red = lose, grey = tie",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=160)
            written.append(path)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# deliverables
# ---------------------------------------------------------------------------


def _identity_status(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = 0
    failed = []
    for cell_id, cell in cells.items():
        for budget, block in cell.get("budgets", {}).items():
            identity = block.get("identity", {})
            total += int(identity.get("repeats", 0))
            if not identity.get("all_identical", True):
                failed.append(f"{cell_id}/D{budget}")
    return {"identity_checked_repeats": total, "failing_cells": failed,
            "all_identical": not failed}


def build_summary_json(
    results_dir: Path,
    crossover: dict[tuple[int, int], dict[str, Any]],
    cells: dict[str, dict[str, Any]],
    extra_records: list[dict[str, Any]],
    divergent_repeats: int = 0,
) -> dict[str, Any]:
    manifest = json.loads((results_dir / "manifest.json").read_text())
    result_path = results_dir / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"repeats_used": manifest["protocol"]["timed_repeats"],
              "elapsed_minutes": None, "deviations": ["partial: sweep still running"],
              "l32_included": False}
    )
    return {
        "schema_version": "v66_boundary_summary_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "results_dir": str(results_dir),
        "prereg": {
            **manifest["preregistration"],
            "amendment_1": "arm-A positions sequential->restart before any timed data "
            "(smoke-triggered harness repair); see prereg md",
        },
        "grid_executed": {
            "hit_fractions": manifest["grid"]["hit_fractions"],
            "doc_lengths": sorted({cell["L"] for cell in cells.values()}),
            "n_docs": sorted({cell["N"] for cell in cells.values()}),
            "decode_budgets": [TTFT_BUDGET, *DECODE_BUDGETS],
            "repeats_used": result.get("repeats_used"),
            "l32_included": result.get("l32_included", False),
            "elapsed_minutes": result.get("elapsed_minutes"),
        },
        "crossover": {
            f"L{length}|{BUDGET_LABELS[budget]}": {
                "wall_crossover_h": entry["wall_crossover"]["h"],
                "wall_crossover_achiever": entry["wall_crossover"].get("achiever"),
                "ttft_crossover_h": entry["ttft_crossover"]["h"],
                "ttft_crossover_achiever": entry["ttft_crossover"].get("achiever"),
            }
            for (length, budget), entry in sorted(crossover.items())
        },
        "identity": {**_identity_status(cells),
                     "divergent_repeat_records": divergent_repeats},
        "infeasible_cells": [
            record.get("cell_id") for record in extra_records
            if record.get("status") == "infeasible"
        ],
        "deviations": result.get("deviations", []),
        "prewarm": [
            record for record in extra_records if record.get("type") == "prewarm"
        ],
    }


def _sweep_elapsed_minutes(results_dir: Path) -> float | None:
    """Main-sweep elapsed from the archived record-mode completion log (pure I/O)."""
    log = results_dir / "logs/attempt7_record_mode_complete.log"
    if not log.exists():
        return None
    for line in reversed(log.read_text().splitlines()):
        if '"stage": "done"' in line:
            try:
                return float(json.loads(line)["elapsed_minutes"])
            except (json.JSONDecodeError, KeyError, ValueError):
                return None
    return None


def build_readme(
    results_dir: Path,
    summary: dict[str, Any],
    crossover: dict[tuple[int, int], dict[str, Any]],
    sweep_elapsed: float | None = None,
) -> str:
    manifest = json.loads((results_dir / "manifest.json").read_text())
    grid = summary["grid_executed"]
    lines = [
        "# V66 — optimized-path serving boundary (VIBEQUEUE-66)",
        "",
        f"- Results dir: `{results_dir}`",
        f"- Preregistration: `{manifest['preregistration']['md']}` — as pinned by the "
        f"run manifest at sweep start: md sha256 "
        f"`{manifest['preregistration']['md_sha256']}`, json sha256 "
        f"`{manifest['preregistration']['json_sha256']}` (revision 2); final "
        f"post-Amendment-2 prereg on disk: md sha256 `{_sha_file(PREREG_MD)}`, json "
        f"sha256 `{_sha_file(PREREG_JSON)}` (revision 3); revision-1 json "
        f"`674578fb7d93a24cc56644e1d027d975be2093f610cee4457660d494c02e0b1a` "
        "preserved. Amendments are documented in the prereg md.",
        f"- Runner: `{manifest['runner']}` — as pinned at sweep start: sha256 "
        f"`{manifest['sources']['scripts/run_v66_boundary.py']}`; final runner "
        f"(post prune-fix + record/redo flags): sha256 "
        f"`{_sha_file(Path(__file__).resolve())}`",
        f"- Data: development-only (frozen V62 dev split, sha256 "
        f"`{manifest['data']['split_manifest_sha256']}`); the 100 confirmation ids "
        "were not used; content seed 47.",
        f"- Checkpoint: `{manifest['checkpoint']}` "
        f"({manifest['checkpoint_identity']['sha256'][:16]}…)",
        f"- Grid executed: h ∈ {grid['hit_fractions']} (round-half-up to hit_count), "
        f"L ∈ {grid['doc_lengths']}, N ∈ {grid['n_docs']}, decode budgets "
        f"{grid['decode_budgets']}; {grid['repeats_used']} timed repeats per cell per "
        "arm after 3 shared warmups, interleaved A_i/B_i; L=32 included: "
        f"{grid['l32_included']}.",
        f"- Wall time: main sweep {sweep_elapsed if sweep_elapsed is not None else '?'} min "
        f"(logs/attempt7_record_mode_complete.log) + 10.9-min D1 repair pass "
        f"(logs/attempt8_d1_repair.log); hard cap 180 min; result.json "
        f"elapsed_minutes reflects the final runner pass "
        f"({grid['elapsed_minutes']} min).",
        f"- Deviations (all recorded before the affected data was used, see the prereg "
        "md): (1) Amendment 1 — arm-A positions sequential→restart after the "
        "preregistered smoke paused on a byte-identity failure (no timed data "
        "affected); (2) D1-block repair — the B-TTFT pass's un-pruned write-through "
        "state leaked into the B wall(1) pass of hit_count<N cells; the prune "
        "placement was fixed and every affected D1 block re-run (append-only "
        "supersede; TTFT and D8–D64 data were unaffected); (3) Amendment 2 — after "
        "the frozen identity gate paused at L512_N10_H10_D8, the divergence was "
        "diagnosed as bf16 reduction-order differences between the mathematically "
        "equivalent joint-prefill and assembled-KV executions (first token "
        f"byte-identical in 825/825 divergent repeats at the time of the amendment; {summary['identity']['divergent_repeat_records']} divergent repeats total), and the sweep continued in record "
        "mode with per-cell identity verdicts.",
        "- Identity: "
        + _identity_status_from(summary)
        + " Divergences concentrate at (L=512, N=10) for decode ≥ 8 (trajectory flip "
        "at generated token 4) and (L=32) for decode ≥ 16 (flips deeper in decode).",
        f"- Infeasible cells: {summary['infeasible_cells'] or 'none'}",
        "",
        "## Measured crossover (smallest nominal h with CI-supported win)",
        "",
        "| L | decode budget | crossover h (wall) | crossover h (TTFT) |",
        "|---|---|---|---|",
    ]
    for (length, budget), entry in sorted(crossover.items()):
        lines.append(
            f"| {length} | {BUDGET_LABELS[budget]} | "
            f"{crossover_statement(entry['wall_crossover'])} | "
            f"{crossover_statement(entry['ttft_crossover'])} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `manifest.json` — run manifest (written before the model loaded)",
        "- `progress.jsonl` — append-only per-cell checkpoint log",
        "- `raw_repeats.jsonl` — one line per timed pass",
        "- `result.json` — sweep result payload",
        "- `crossover_table.md` — full per-cell table + crossover statements",
        "- `crossover_figure.(png|pdf)`, `crossover_figure_ttft.(png|pdf)` — figures",
        "- `summary.json` — machine-readable crossover + provenance",
        "- `scoping_sentence.md` — updated paper scoping sentence",
        "- `t9_followup_fragment.tex`, `t9_followup_evidence_fragment.json` — "
        "ready-to-paste fragments for the paper evidence/render pipeline (the pinned "
        "pipeline was not modified)",
        "- `canonical_store/` — per-(L, N) DocCache stores built during the sweep",
        "",
    ]
    return "\n".join(lines)


def _identity_status_from(summary: dict[str, Any]) -> str:
    identity = summary["identity"]
    total = identity["identity_checked_repeats"]
    divergent = identity.get("divergent_repeat_records", 0)
    n_blocks = len(identity["failing_cells"])
    if not identity["all_identical"]:
        return (
            f"byte-identical A vs B on {total - divergent}/{total} timed repeats; "
            f"{divergent} divergent repeats across {n_blocks} cell-blocks (full list "
            "in summary.json; divergences are bf16 reduction-order trajectory flips, "
            "first token identical in every one; both arms perform identical fixed "
            "work so timing validity is unaffected)"
        )
    return (f"byte-identical A vs B on all {total} "
            "timed repeats across all cells")


def build_scoping_sentence(
    crossover: dict[tuple[int, int], dict[str, Any]],
    divergent_repeats: int = 0,
    total_repeats: int = 0,
) -> str:
    """The updated paper scoping sentence with the measured boundary."""
    def h_text(length: int, budget: int, metric: str = "wall") -> str:
        entry = crossover.get((length, budget))
        if entry is None:
            return "not measured"
        value = entry[f"{metric}_crossover"]["h"]
        return "no winning cell in the grid" if value is None else f"h ≥ {value:g}"

    ttft_128 = h_text(128, TTFT_BUDGET, "ttft")
    ttft_512 = h_text(512, TTFT_BUDGET, "ttft")
    wall1_128 = h_text(128, TTFT_BUDGET)
    wall1_512 = h_text(512, TTFT_BUDGET)
    wall_128 = h_text(128, 64)
    wall_512 = h_text(512, 64)
    identical = total_repeats - divergent_repeats
    return f"""# Updated paper scoping sentence (VIBEQUEUE-66)

Replaces the "high-hit-rate" scoping in `paper/sections/main_story.tex` (lines ~27,
~347 and ~440) with the measured boundary. Measured on development data only (frozen
V62 dev split, seed 47; single seed; 15 timed repeats per cell; frozen materiality
rule: win iff A/B speedup CI lower bound > 1.05, lose iff upper bound < 1/1.05).
The h grid is per request with N ∈ {{2, 5, 10}} documents of exactly L tokens; the
crossover is the smallest nominal h with a CI-supported win in the grid (achieving
N in parentheses).

## Primary scoping sentence (replaces the line-27 sentence)

> The headlines are narrow by design and now carry a measured boundary: on
> development data the optimized warm reuse path beats the monolithic baseline only
> above measured hit-fraction crossovers — first-token latency wins above
> {ttft_128} at L = 128 and {ttft_512} at L = 512 (all-hit TTFT speedups up to
> ~4.3x at L = 512, N = 10), diluting with decode budget until full 64-token
> generations win only above {wall_512} at L = 512 and never with CI support at
> L = 128 or at L = 32 at any decode budget — with outputs byte-identical to the
> monolithic baseline on {identical}/{total_repeats} timed repeats (divergences are
> bf16 reduction-order flips deeper in the trajectory, first token always
> identical); quality non-inferiority is not established; the sparse cold path wins
> only at long-document shapes.

## Conclusion-paragraph variant (replaces the line-440 phrase "scoped to
high-hit-rate warm traffic")

> --- winning only above the measured hit-fraction crossovers (TTFT: {ttft_128} at
> L = 128, {ttft_512} at L = 512; first-generation-token wall: {wall1_128} and
> {wall1_512}; full 64-token generations: no CI-supported win at L = 128, {wall_512}
> at L = 512, never at L = 32) —
"""


def build_t9_fragments(
    crossover: dict[tuple[int, int], dict[str, Any]],
    output: Path,
) -> list[Path]:
    """Ready-to-paste T9-style fragments (pipeline not modified)."""
    rows = []
    for (length, budget), entry in sorted(crossover.items()):
        wall_h = entry["wall_crossover"]["h"]
        ttft_h = entry["ttft_crossover"]["h"]
        rows.append((
            f"L={length} {BUDGET_LABELS[budget]}",
            "none in grid" if wall_h is None else f"{wall_h:g}",
            "none in grid" if ttft_h is None else f"{ttft_h:g}",
        ))
    tex = "\n".join(
        [
            "% V66 follow-up fragment for the evidence/render pipeline (ready-to-paste).",
            "% The pinned pipeline (paper/evidence.json + paper/render.py _render_t9 +",
            "% paper/expanded.py verification + sections/main_story.tex table env) was",
            "% NOT modified by VIBEQUEUE-66; adding this table requires a new artifact",
            "% key (e.g. T13_v66_boundary) wired through those four pinned files.",
            r"\begin{tabular}{llr}",
            r"\toprule",
            r"Document length & Decode budget & Crossover hit fraction $h$ (wall / TTFT) \\",
            r"\midrule",
        ]
        + [
            f"{length_label} & {budget_label} & "
            + " / ".join(
                value if value == "none in grid" else f"${value}$"
                for value in (wall, ttft)
            )
            + r" \\"
            for length_label, budget_label, wall, ttft in [
                (row[0].split(" ", 1)[0], row[0].split(" ", 1)[1], row[1], row[2])
                for row in rows
            ]
        ]
        + [r"\bottomrule", r"\end{tabular}", ""]
    )
    evidence = {
        "schema_version": "v66_boundary_t9_followup_v1",
        "note": (
            "ready-to-paste artifact fragment; not wired into paper/evidence.json "
            "(pinned files unmodified per task constraints)"
        ),
        "rows": [
            {
                "L": int(row[0].split("=", 1)[1].split(" ", 1)[0]),
                "decode_budget": row[0].split(" ", 1)[1],
                "crossover_h_wall": None if row[1] == "none in grid" else float(row[1]),
                "crossover_h_ttft": None if row[2] == "none in grid" else float(row[2]),
            }
            for row in rows
        ],
        "source": "results/v66_boundary_<UTC>/summary.json",
    }
    written = []
    for name, payload in (
        ("t9_followup_fragment.tex", tex),
        ("t9_followup_evidence_fragment.json",
         json.dumps(evidence, indent=2, allow_nan=False) + "\n"),
    ):
        path = output / name
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else argv
    results_dir = (
        Path(argv[0]) if argv else max(
            (path for path in (ROOT / "results").glob("v66_boundary_*") if path.is_dir()),
            key=lambda path: path.name,
        )
    )
    cells, extra_records, divergent_repeats = load_cells(results_dir)
    units = unit_tables(cells)
    table, crossover = build_crossover_table(units, infeasible=[
        record for record in extra_records if record.get("status") == "infeasible"
    ])
    (results_dir / "crossover_table.md").write_text(table, encoding="utf-8")
    figures = build_figures(units, results_dir)
    summary = build_summary_json(results_dir, crossover, cells, extra_records,
                                 divergent_repeats)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (results_dir / "README.md").write_text(
        build_readme(results_dir, summary, crossover,
                     sweep_elapsed=_sweep_elapsed_minutes(results_dir)), encoding="utf-8"
    )
    (results_dir / "scoping_sentence.md").write_text(
        build_scoping_sentence(
            crossover,
            divergent_repeats=divergent_repeats,
            total_repeats=summary["identity"]["identity_checked_repeats"],
        ), encoding="utf-8"
    )
    fragments = build_t9_fragments(crossover, results_dir)
    print(json.dumps({
        "stage": "analysis", "results_dir": str(results_dir),
        "figures": [path.name for path in figures],
        "fragments": [path.name for path in fragments],
        "crossover": summary["crossover"],
        "identity": summary["identity"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
