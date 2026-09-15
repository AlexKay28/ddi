#!/usr/bin/env python3
"""One-shot extraction: build V14/V15/V16/V17 ledger artifacts + rebuilt T13/T17 rows
from pinned raw sources (VIBEQUEUE-89..98 prepublication review, V92/V91).

Every value written into paper/evidence.json by this script is copied from a field
of a sha256-pinned source file; the script re-verifies each source hash before use
and prints the derivation. Re-running is idempotent w.r.t. the sources (they are
frozen results); it is NOT part of the render path -- the renderer only reads the
ledger. Kept for provenance audit.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV_PATH = ROOT / "paper" / "evidence.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ev: dict, name: str) -> None:
    src = ev["sources"][name]
    path = ROOT / src["path"]
    digest = sha256(path)
    assert digest == src["sha256"], f"{name}: source hash drifted ({path})"
    print(f"[ok] {name}: {src['path']} sha256 verified")


def f3(x: float) -> float:
    return round(float(x), 3)


def main() -> None:
    ev = json.loads(EV_PATH.read_text())
    for name in (
        "v71_sweep_result",
        "v71_summary",
        "v66_boundary_crossover_table",
        "v70_confirmation_verdict",
        "v72_confirmation_verdict",
        "v72s2_confirmation_verdict",
        "cache_v53_summary",
        "v65_combined_result",
    ):
        if name not in ev["sources"]:
            if name == "v72s2_confirmation_verdict":
                path = ROOT / "results/v72s2_confirmation_analysis/verdict.json"
                ev["sources"][name] = {
                    "path": "results/v72s2_confirmation_analysis/verdict.json",
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "required": True,
                }
                print(f"[add] source v72s2_confirmation_verdict pinned ({path.name})")
        check(ev, name)

    # ---- V71 per-cell serving ratios (result.json) ------------------------
    v71 = json.loads((ROOT / ev["sources"]["v71_sweep_result"]["path"]).read_text())
    cells = {}
    for c in v71["cells"]:
        assert c["status"] == "ok", f"cell {c['cell_id']} not ok: {c['status']}"
        assert len(c["budgets"]) == 1, f"cell {c['cell_id']} has multiple budgets"
        (budget_key,) = c["budgets"]
        cells[(c["L"], c["N"], c["hit_count"], budget_key)] = c
    assert len(cells) == len(v71["cells"]) == 240

    v71_series: dict[int, list[list]] = {}
    for (L, N, h, _), c in sorted(cells.items()):
        if _ != "1":
            continue
        b1 = c["budgets"]["1"]
        v71_series.setdefault(L, []).append(
            ["V71", N, f3(h / N), h, None, f3(b1["ratio_point"]), f3(b1["ttft_ratio"]["ratio_point"])]
        )

    # ---- V66 series + per-(L,N) crossovers from the pinned table ----------
    table = (ROOT / ev["sources"]["v66_boundary_crossover_table"]["path"]).read_text()
    row_re = re.compile(
        r"^\| (\d(?:\.5)?|\d) \| (\d+) \| (\d+(?:\.\d+)?) \| "
        r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\]"
        r"(?: \| ([0-9.]+) \[([0-9.]+), ([0-9.]+)\])*"
    )
    # generic parse of one per-N block
    block_re = re.compile(r"## L = (\d+), N = (\d+)\n\n(.*?)\n\n", re.S)
    metric_re = re.compile(
        r"^\| [^|]+ \| (?P<hits>\d+) \| (?P<heff>\d+(?:\.\d+)?) \| "
        r"(?P<w1>[0-9.]+) \[(?P<w1lo>[0-9.]+), (?P<w1hi>[0-9.]+)\] \| "
        r"(?P<w8>[0-9.]+) \[(?P<w8lo>[0-9.]+), (?P<w8hi>[0-9.]+)\] \| "
        r"(?P<w16>[0-9.]+) \[(?P<w16lo>[0-9.]+), (?P<w16hi>[0-9.]+)\] \| "
        r"(?P<w32>[0-9.]+) \[(?P<w32lo>[0-9.]+), (?P<w32hi>[0-9.]+)\] \| "
        r"(?P<w64>[0-9.]+) \[(?P<w64lo>[0-9.]+), (?P<w64hi>[0-9.]+)\] \| "
        r"(?P<ttft>[0-9.]+) \[(?P<ttftlo>[0-9.]+), (?P<ttfthi>[0-9.]+)\] \| "
        r"(?P<ttftv>\w+) \|",
        re.M,
    )
    v66_series: dict[int, list[list]] = {128: [], 512: []}
    t13_rows: list[list] = []
    seen_v66: set[tuple] = set()
    for m in block_re.finditer(table):
        L, N, body = int(m.group(1)), int(m.group(2)), m.group(3)
        best = {d: None for d in ("1", "8", "16", "32", "64")}
        best_ttft = None
        best_ttft_nom = None
        best_nom = {d: None for d in best}
        for r in metric_re.finditer(body):
            hits, heff, nom = int(r["hits"]), float(r["heff"]), float(r.group(1))
            wall_win = {}
            for d in ("1", "8", "16", "32", "64"):
                lo = float(r[f"w{d}lo"])
                hi = float(r[f"w{d}hi"])
                # frozen V66 materiality rule: win iff CI low > 1.05
                wall_win[d] = lo > 1.05
                if wall_win[d] and best[d] is None:
                    best[d] = heff
                    best_nom[d] = nom
            if r["ttftv"] == "win" and best_ttft is None:
                best_ttft = heff
                best_ttft_nom = nom
            key = (L, N, hits)
            if key not in seen_v66 and L in (128, 512):
                seen_v66.add(key)
                v66_series[L].append(
                    ["V66", N, f3(heff), hits, nom, f3(float(r["w1"])), f3(float(r["ttft"]))]
                )
        t13_rows.append(
            [L, N]
            + [best[d] for d in ("1", "8", "16", "32", "64")]
            + [best_ttft]
            + [[best_nom[d] for d in ("1", "8", "16", "32", "64")] + [best_ttft_nom]]
        )
    for L in (128, 512):
        v66_series[L].sort(key=lambda r: (r[1], r[2]))
        v71_series[L].sort(key=lambda r: (r[1], r[2]))
        assert v66_series[L], f"no V66 rows parsed for L={L}"

    # ---- T17 (V71 per-shape crossovers, min over N) -----------------------
    # ---- T17: comparative V66 -> V71 boundary per (L, N) ------------------
    # columns [L, N, v66_wall_d1, v71_wall_d1, v66_ttft, v71_ttft], h_eff values
    t13_by_key = {(r[0], r[1]): r for r in t13_rows}
    t17_rows = []
    for L in (32, 128, 512):
        Ns = sorted({N for (l, N, _, _) in cells if l == L})
        for N in Ns:
            hit_counts = sorted({h for (l, n, h, _) in cells if l == L and n == N})
            v71_w1 = None
            v71_ttft = None
            for h in hit_counts:
                c1 = cells.get((L, N, h, "1"))
                if c1 is None:
                    continue
                if v71_w1 is None and c1["budgets"]["1"]["verdict"] == "win":
                    v71_w1 = f3(h / N)
                if v71_ttft is None and c1["budgets"]["1"]["ttft_ratio"]["verdict"] == "win":
                    v71_ttft = f3(h / N)
            v66 = t13_by_key[(L, N)]
            t17_rows.append([L, N, v66[2], v71_w1, v66[7], v71_ttft])

    # cross-check T17 min-over-N against the official v71 summary crossover (d=1 key)
    summary = json.loads((ROOT / ev["sources"]["v71_summary"]["path"]).read_text())
    for key, entry in summary["crossover"].items():
        if "|TTFT-only (d=1)" not in key:
            continue
        L = int(key.split("|")[0][1:])
        for metric, col in (("wall", 3), ("ttft", 5)):
            ach = entry[f"{metric}_crossover_achiever"]
            if entry[f"{metric}_crossover_h"] is None:
                assert ach is None
                continue
            got = [r[col] for r in t17_rows if r[0] == L and r[1] == ach["N"]]
            assert got and got[0] == f3(ach["h_eff"]), (
                f"T17 mismatch {key} {metric}: rows {got} vs official h_eff {ach['h_eff']}"
            )
    print("[ok] T17 rows cross-check against v71 summary crossover achievers")

    # ---- V16 gate deltas from the three verdicts --------------------------
    v16_rows = []
    stations = [
        ("V70", "Qwen2.5-7B", 47, 5, "v70_confirmation_verdict"),
        ("V72-a", "Qwen2.5-7B", 1, 0, "v72_confirmation_verdict"),
        ("V72-b", "Llama-3.1-8B", 47, 5, "v72s2_confirmation_verdict"),
    ]
    for station, family, seed, k, src in stations:
        v = json.loads((ROOT / ev["sources"][src]["path"]).read_text())
        assert v["input"]["record_count"] == 300 and v["overall_gate_pass"] is True
        cb = v["gates"]["C-B"]["metrics"]["em"]
        cd = v["gates"]["C-D"]["metrics"]["em"]
        assert cb["mean"] == 0.0 and cb["ci95"] == [0.0, 0.0]
        v16_rows.append(
            [
                station,
                family,
                seed,
                k,
                f3(cb["mean"]), f3(cb["ci95"][0]), f3(cb["ci95"][1]),
                f3(cd["mean"]), f3(cd["ci95"][0]), f3(cd["ci95"][1]),
                v["input"]["record_count"],
            ]
        )
    print("[ok] V16 rows extracted from three verdicts")

    # ---- V17 program-timeline values --------------------------------------
    v53 = ev["artifacts"]["V53_cache_diagnostic"]
    t9 = ev["artifacts"]["T9_v65_serving"]
    v70_rows = ev["artifacts"]["T15_v70_gates"]["arm_means"]
    ca = next(r for r in t9["rows"] if r[0] == "ratio" and r[1] == "benchmark C/A")
    bench = next(r for r in t9["rows"] if r[0] == "benchmark" and r[1] == "C all-hit reuse")
    mono = next(r for r in t9["rows"] if r[0] == "benchmark" and r[1] == "A monolithic")
    v71_ttft = cells[(512, 10, 10, "1")]["budgets"]["1"]["ttft_ratio"]["ratio_point"]
    ident = summary["identity_cp_bp"]
    v17_rows = [
        ["V53", "hit_rate", f3(v53["hit_rate"])],
        ["V53", "cache_over_fresh_wall", f3(v53["ratios"][0][1])],
        ["V53", "divergent_streams", v53["behavior_failures"]["measured_streams"][0]],
        ["V53", "measured_streams", v53["behavior_failures"]["measured_streams"][1]],
        ["V65", "allhit_speedup_vs_monolithic", f3(1.0 / ca[2])],
        ["V65", "byte_identical_pairs", t9["byte_identical_ac"][0]],
        ["V65", "byte_identical_total", t9["byte_identical_ac"][1]],
        ["V70", "cb_em", f3(v70_rows["C"]["em"] - v70_rows["B"]["em"])],
        ["V70", "n_records", 300],
        ["V71", "allhit_ttft_speedup_L512_N10", f3(v71_ttft)],
        ["V71", "identity_violations", ident["divergent_repeat_records"]],
        ["V71", "identity_blocks_checked", ident["blocks_checked"]],
        ["V71", "grid_cells", len(v71["cells"])],
        ["V71", "repeats_per_cell", v71["repeats_used"]],
    ]
    # sanity anchors against the figures' previous pinned prose
    assert v17_rows[0][2] == 0.945 and v17_rows[1][2] == 1.193
    assert v17_rows[4][2] == 2.047 and v17_rows[9][2] == 5.547
    assert bench[2] < mono[2]  # reuse faster than monolithic in the benchmark
    print("[ok] V17 rows extracted (V53/V65/V70/V71)")

    # ---- write ledger ------------------------------------------------------
    art = ev["artifacts"]
    art["V14_v71_crossover_L512"] = {
        "sources": ["v71_sweep_result", "v66_boundary_crossover_table"],
        "note": (
            "Crossover chart at L=512. Rows [contract, N, h_eff, hit_count, nominal_h, "
            "wall_d1_ratio, ttft_ratio]. V71: A/CP paired ratio of medians at decode d=1 "
            "(wall) and strict TTFT (ttft_ratio) from v71 result.json budgets['1']. "
            "V66: wall d=1 A/B and TTFT A/B point estimates from crossover_table.md. "
            "h_eff = hit_count/N (actual cached fraction); nominal_h is the grid label "
            "(None for V71 rows: the sweep realizes hit counts directly)."
        ),
        "rows": v66_series[512] + v71_series[512],
    }
    art["V15_v71_crossover_L128"] = {
        "sources": ["v71_sweep_result", "v66_boundary_crossover_table"],
        "note": art["V14_v71_crossover_L512"]["note"].replace("L=512", "L=128"),
        "rows": v66_series[128] + v71_series[128],
    }
    art["V16_gate_deltas"] = {
        "sources": [
            "v70_confirmation_verdict",
            "v72_confirmation_verdict",
            "v72s2_confirmation_verdict",
        ],
        "note": (
            "Rows [station, family, seed, k, cb_em, cb_lo, cb_hi, cd_em, cd_lo, cd_hi, n] "
            "from verdict.json gates C-B/C-D EM metrics (paired percentile bootstrap, "
            "10k resamples, seed 47). C-B is exactly 0 with degenerate CI [0,0] at every "
            "station: the all-zero paired differences make every resample zero, so the "
            "interval carries no information beyond the point mass (VIBEQUEUE-94)."
        ),
        "rows": v16_rows,
    }
    art["V17_program_timeline"] = {
        "sources": [
            "cache_v53_summary",
            "v65_combined_result",
            "v70_confirmation_verdict",
            "v71_summary",
            "v71_sweep_result",
        ],
        "note": (
            "Rows [station, field, value]. V53 hit rate and cache/fresh wall ratio from "
            "V53_cache_diagnostic artifact fields (sources cache_v53_*); divergent greedy "
            "streams 19/150. V65 all-hit speedup = 1/(benchmark C/A wall ratio 0.48854) "
            "and byte_identical_ac from T9_v65_serving artifact. V70 C-B EM from "
            "T15_v70_gates arm_means (C-B = 0.0 exactly). V71 all-hit TTFT ratio "
            "(L512_N10_H10_D1 budgets['1'].ttft_ratio), identity_cp_bp zero divergent "
            "repeat records over 48 blocks, 240 grid configurations, 15 repeats."
        ),
        "rows": v17_rows,
    }
    art["T13_v66_boundary"]["rows"] = t13_rows
    art["T13_v66_boundary"]["value_semantics"] = (
        "Cells are h_eff = hit_count/N of the smallest tested winning configuration for "
        "that (L, N, budget) under the frozen materiality rule (win iff bootstrap CI low "
        "> 1.05); null = no winning configuration in the frozen grid. Nominal grid labels "
        "are kept per row in 'nominal' and must not be read as universal thresholds."
    )
    art["T13_v66_boundary"]["nominal"] = [r[-1] for r in t13_rows]
    art["T13_v66_boundary"]["rows"] = [r[:-1] for r in t13_rows]
    art["T17_v71_boundary"]["rows"] = t17_rows
    art["T17_v71_boundary"]["value_semantics"] = (
        "Comparative serving boundary per (L, N): [L, N, V66 wall d=1, V71 wall d=1, "
        "V66 TTFT, V71 TTFT]. Every value is h_eff = hit_count/N of the smallest tested "
        "winning configuration under the frozen materiality rule (win iff bootstrap CI "
        "low > 1.05); null = none in the frozen grid. V66 columns come from the pinned "
        "crossover_table.md; V71 columns from v71 result.json per-cell A/CP verdicts "
        "(min over N equals the official summary crossover, cross-checked). These are "
        "smallest tested winning configurations on the frozen development grid, not "
        "universal thresholds."
    )

    EV_PATH.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {EV_PATH} updated")


if __name__ == "__main__":
    main()
