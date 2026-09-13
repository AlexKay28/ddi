"""VIBEQUEUE-71: trained-contract boundary analysis and V66 diff.

Consumes a V71 sweep directory (``progress.jsonl`` + ``manifest.json`` +
``result.json``) plus the frozen V66 summary, and writes into the V71
directory: ``crossover_table.md``, ``summary.json`` (schema
``v71_boundary_summary_v1``) and ``v66_v71_boundary_diff.md``.

Crossovers use the same pure helpers and the same frozen materiality rule as
V66: win iff the A/CP paired-ratio CI low > 1.05, lose iff CI high < 1/1.05.
Every verdict is runner-emitted; this script computes no thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_v66_boundary import (  # noqa: E402
    BUDGET_LABELS,
    crossover_h,
    crossover_statement,
    load_cells,
    unit_tables,
)
from scripts.run_canonical_matrix import _sha_file  # noqa: E402
from scripts.run_v66_boundary import DECODE_BUDGETS, TTFT_BUDGET  # noqa: E402

DEFAULT_V66_SUMMARY = (
    ROOT / "results/v66_boundary_20260910T132922Z/summary.json"
)


def _crossover_map(
    units: dict[tuple[int, int], dict[int, dict[str, Any]]],
) -> tuple[str, dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """Render the V71 crossover table; return (markdown, crossover, identity)."""
    lines: list[str] = [
        "# V71 — trained-contract reuse (CP) vs monolithic baseline (A): "
        "measured serving boundary",
        "",
        "Ratio = median(A)/median(CP) wall speedup on the frozen V66 grid; CP is the",
        "V70 k=5 trained-contract cached path. CI = paired percentile bootstrap",
        "(seed 47, 10k). Verdicts: win iff CI low > 1.05, lose iff CI high < 1/1.05,",
        "else tie (frozen materiality rule, identical to V66). TTFT is",
        "decode-budget-independent. Crossover h = smallest nominal h in the grid",
        "whose verdict is win.",
        "",
    ]
    crossover: dict[tuple[int, int], dict[str, Any]] = {}
    identity_ok = True
    identity_blocks = 0
    for length in sorted({key[0] for key in units}):
        for n_docs in sorted({key[1] for key in units}):
            unit = units[(length, n_docs)]
            lines += [
                f"## L = {length}, N = {n_docs}",
                "",
                "| nominal h | hit_count | h_eff | wall d=1 A/CP | wall d=8 | wall d=16"
                " | wall d=32 | wall d=64 | TTFT A/CP | TTFT verdict | CP=BP identity |"
                " BP/A wall (desc) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
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
                wall_cells = []
                for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
                    block = entry["budgets"].get(budget)
                    wall_cells.append(
                        f"{block['ratio_point']:.3f} "
                        f"[{block['ratio_ci_low']:.3f}, {block['ratio_ci_high']:.3f}]"
                        if block else "—"
                    )
                ttft_text = (
                    f"{ttft['ratio_point']:.3f} [{ttft['ratio_ci_low']:.3f}, "
                    f"{ttft['ratio_ci_high']:.3f}]" if ttft else "—"
                )
                ttft_verdict = ttft["verdict"] if ttft else "—"
                d1_block = entry["budgets"].get(TTFT_BUDGET)
                identity = "—"
                bp_desc = "—"
                if d1_block is not None:
                    identity_ok &= bool(d1_block["identity_cp_bp"]["all_identical"])
                    identity_blocks += 1
                    identity = "ok" if d1_block["identity_cp_bp"]["all_identical"] else "FAIL"
                    bp = d1_block.get("bp_wall_ratio_descriptive")
                    bp_desc = (
                        f"{bp['ratio_point']:.3f} [{bp['ratio_ci_low']:.3f}, "
                        f"{bp['ratio_ci_high']:.3f}]" if bp else "—"
                    )
                h_labels = sorted(entry["h_labels"])
                for h in h_labels:
                    lines.append(
                        f"| {h:g} | {hit_count} | {h_eff:g} | " + " | ".join(wall_cells)
                        + f" | {ttft_text} | {ttft_verdict} | {identity} | {bp_desc} |"
                    )
                for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
                    block = entry["budgets"].get(budget)
                    if block is None:
                        continue
                    summary = crossover.setdefault((length, budget), {"per_unit": {}})
                    summary["per_unit"][(n_docs, hit_count)] = {
                        "h_labels": h_labels,
                        "wall_verdict": block["verdict"],
                        "ttft_verdict": ttft["verdict"] if ttft else None,
                    }
            lines.append("")
        lines += ["", f"### Measured crossover per decode budget — L = {length}", ""]
        lines += [
            "| decode budget | crossover h (wall, A/CP) | crossover h (TTFT, A/CP) |",
            "|---|---|---|",
        ]
        for budget in (TTFT_BUDGET, *DECODE_BUDGETS):
            entry = crossover.get((length, budget))
            if entry is None:
                entry = crossover.setdefault((length, budget), {"per_unit": {}})
            wall_rows = [
                {"h": h, "n_docs": n, "hit_count": hc, "h_eff": hc / n,
                 "wall_verdict": verdict}
                for (n, hc), value in sorted(entry["per_unit"].items())
                for h, verdict in zip(
                    value["h_labels"],
                    [value["wall_verdict"]] * len(value["h_labels"]), strict=False)
            ]
            ttft_rows = [
                {"h": h, "n_docs": n, "hit_count": hc, "h_eff": hc / n,
                 "ttft_verdict": verdict}
                for (n, hc), value in sorted(entry["per_unit"].items())
                for h, verdict in zip(
                    value["h_labels"],
                    [value["ttft_verdict"]] * len(value["h_labels"]), strict=False)
            ]
            entry["wall_crossover"] = crossover_h(wall_rows, "wall")
            entry["ttft_crossover"] = crossover_h(ttft_rows, "ttft")
            lines.append(
                f"| {BUDGET_LABELS[budget]} | "
                f"{crossover_statement(entry['wall_crossover'])} | "
                f"{crossover_statement(entry['ttft_crossover'])} |"
            )
        lines.append("")
    identity = {"all_blocks_identical": identity_ok, "blocks_checked": identity_blocks}
    return "\n".join(lines) + "\n", crossover, identity


def _diff_markdown(
    v71_crossover: dict[tuple[int, int], dict[str, Any]],
    v66_summary: dict[str, Any],
) -> str:
    lines = [
        "# V66 → V71 boundary diff (untrained → trained contract)",
        "",
        "Same frozen grid, same materiality rule, same arm-A baseline.",
        "V66 B = untrained canonical reuse; V71 CP = trained k=5 reuse.",
        "",
        "| decode budget | L | V66 wall h | V71 wall h | shift | V66 TTFT h |"
        " V71 TTFT h | shift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (length, budget), entry in sorted(v71_crossover.items()):
        v66_key = f"L{length}|{BUDGET_LABELS[budget]}"
        v66_entry = v66_summary["crossover"].get(v66_key, {})
        wall_66 = v66_entry.get("wall_crossover_h")
        wall_71 = entry["wall_crossover"]["h"]
        ttft_66 = v66_entry.get("ttft_crossover_h")
        ttft_71 = entry["ttft_crossover"]["h"]

        def fmt(value: float | None) -> str:
            return "none" if value is None else f"{value:g}"

        def shift(old: float | None, new: float | None) -> str:
            if old is None and new is None:
                return "—"
            if old is None:
                return "appears"
            if new is None:
                return "lost"
            if new == old:
                return "0"
            return f"{new - old:+g}"

        lines.append(
            f"| {BUDGET_LABELS[budget]} | {length} | {fmt(wall_66)} | {fmt(wall_71)} "
            f"| {shift(wall_66, wall_71)} | {fmt(ttft_66)} | {fmt(ttft_71)} "
            f"| {shift(ttft_66, ttft_71)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--v66-summary", type=Path, default=DEFAULT_V66_SUMMARY)
    args = parser.parse_args()
    results_dir: Path = args.results_dir

    cells, extra_records, divergent_repeats = load_cells(results_dir)
    if not cells:
        raise SystemExit("no completed cells found; nothing to analyze")
    units = unit_tables(cells)
    table_md, crossover, identity = _crossover_map(units)
    (results_dir / "crossover_table.md").write_text(table_md, encoding="utf-8")

    v66_summary = json.loads(args.v66_summary.read_text())
    diff_md = _diff_markdown(crossover, v66_summary)
    (results_dir / "v66_v71_boundary_diff.md").write_text(diff_md, encoding="utf-8")

    result_path = results_dir / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"deviations": ["partial: sweep still running"], "repeats_used": None}
    )
    summary = {
        "schema_version": "v71_boundary_summary_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "results_dir": str(results_dir),
        "prereg": json.loads(
            (results_dir / "manifest.json").read_text()
        )["preregistration"],
        "checkpoints": json.loads(
            (results_dir / "manifest.json").read_text()
        )["checkpoints"],
        "crossover": {
            f"L{length}|{BUDGET_LABELS[budget]}": {
                "wall_crossover_h": entry["wall_crossover"]["h"],
                "wall_crossover_achiever": entry["wall_crossover"].get("achiever"),
                "ttft_crossover_h": entry["ttft_crossover"]["h"],
                "ttft_crossover_achiever": entry["ttft_crossover"].get("achiever"),
            }
            for (length, budget), entry in sorted(crossover.items())
        },
        "identity_cp_bp": {**identity,
                           "divergent_repeat_records": divergent_repeats},
        "infeasible_cells": [
            record.get("cell_id") for record in extra_records
            if record.get("status") == "infeasible"
        ],
        "deviations": result.get("deviations", []),
        "repeats_used": result.get("repeats_used"),
        "elapsed_minutes": result.get("elapsed_minutes"),
        "l32_included": result.get("l32_included", False),
        "v66_summary_source": {
            "path": str(args.v66_summary),
            "sha256": _sha_file(args.v66_summary),
        },
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "analyzed", "results_dir": str(results_dir),
        "cells": len(cells),
        "identity_all_ok": identity["all_blocks_identical"],
        "crossovers": summary["crossover"],
    }))


if __name__ == "__main__":
    main()
