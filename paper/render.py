#!/usr/bin/env python
"""Render paper/generated/{T2,T3}.tex and {F3,F5}.pdf from logs/*_summary.json.

Replaces the missing ddi.report_cli (VIBEQUEUE-16). Reads the JSON summaries
produced by `ddi.aggregate` and emits LaTeX tables + matplotlib figures.

Usage:  python paper/render.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path("logs")
GEN = Path("paper/generated")
GEN.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    with open(LOGS / f"{name}_summary.json") as f:
        return json.load(f)


# ---- T2: quality table (mask x dataset, EM + F1 + presence) ------------------

def render_t2() -> None:
    d = load("quality")
    metrics = {t["metric"]: t["data"] for t in d["tables"]}
    model = "Qwen/Qwen2.5-7B-Instruct"
    masks = ["full", "ddi_qf", "blind"]
    datasets = ["hotpot_qa", "musique", "synthetic_qa"]

    def val(metric: str, mask: str, ds: str) -> str:
        try:
            v = metrics[metric][model][mask][ds]
            return f"{v:.2f}"
        except (KeyError, TypeError):
            return "—"

    lines = [
        r"\begin{tabular}{ll|ccc|ccc}",
        r" & & \multicolumn{3}{c|}{EM} & \multicolumn{3}{c}{F1} \\",
        r" mask & dataset & \multicolumn{3}{c|}{} & \multicolumn{3}{c}{} \\",
        r"\hline",
    ]
    for mask in masks:
        for i, ds in enumerate(datasets):
            row = []
            if i == 0:
                row.append(r"\multirow{3}{*}{" + mask + r"}")
            else:
                row.append("")
            row.append(ds)
            row.append(val("em", mask, ds))
            row.append(val("f1", mask, ds))
            row.append(val("presence", mask, ds))
            # Duplicate for the second metric group (placeholder — actual layout below)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")

    # Simpler: one table with EM, F1, presence per (mask, dataset)
    lines = [
        r"\begin{tabular}{l|ccc|ccc|ccc}",
        r" & \multicolumn{3}{c|}{EM} & \multicolumn{3}{c|}{F1} & \multicolumn{3}{c}{Presence} \\",
        r" mask & hotpot & musique & synth & hotpot & musique & synth & hotpot & musique & synth \\",
        r"\hline",
    ]
    for mask in masks:
        row = [mask]
        for metric in ["em", "f1", "presence"]:
            for ds in datasets:
                row.append(val(metric, mask, ds))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    (GEN / "T2.tex").write_text("\n".join(lines))
    print("[render] T2.tex written")


# ---- T3: timing realization table (by N, for L=512, sdpa) --------------------

def render_t3() -> None:
    d = load("performance")
    metrics = {t["metric"]: t["data"] for t in d["tables"]}
    model = "Qwen/Qwen2.5-7B-Instruct"
    Ns = [1, 3, 5, 9, 13, 25]
    L = "512"
    backend = "sdpa"

    def val(metric: str, mask: str, N: str) -> str:
        try:
            v = metrics[metric][model][L][backend][mask][N]
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)
        except (KeyError, TypeError):
            return "—"

    lines = [
        r"\begin{tabular}{r|ccc|ccc|r}",
        r" & \multicolumn{3}{c|}{TTFT (ms)} & \multicolumn{3}{c|}{TPOT (ms)} & \\",
        r" $N$ & full & ddi\_qf & blind & full & ddi\_qf & blind & sparsity \\",
        r"\hline",
    ]
    for N in Ns:
        Ns_str = str(N)
        sparsity = val("sparsity", "ddi_qf", Ns_str)
        row = [str(N)]
        for metric in ["ttft", "tpot"]:
            for mask in ["full", "ddi_qf", "blind"]:
                row.append(val(metric, mask, Ns_str))
        row.append(sparsity)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    (GEN / "T3.tex").write_text("\n".join(lines))
    print("[render] T3.tex written")


# ---- F3: realization ratio vs N, one line per L -----------------------------

def render_f3() -> None:
    d = load("performance")
    ttft = {t["metric"]: t["data"] for t in d["tables"]}["ttft"]
    analytical = {t["metric"]: t["data"] for t in d["tables"]}["analytical"]
    model = "Qwen/Qwen2.5-7B-Instruct"
    backend = "sdpa"
    Ns = [1, 3, 5, 9, 13, 25]
    Ls = ["128", "512", "2048"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for L in Ls:
        ratios = []
        for N in Ns:
            try:
                ttft_full = ttft[model][L][backend]["full"][str(N)]
                ttft_ddi = ttft[model][L][backend]["ddi_qf"][str(N)]
                realized = ttft_full / ttft_ddi if ttft_ddi > 0 else 1.0
                ana = analytical[model][L][backend]["ddi_qf"][str(N)]
                ratio = realized / ana if ana > 0 else 1.0
                ratios.append(ratio)
            except (KeyError, TypeError):
                ratios.append(None)
        valid = [(N, r) for N, r in zip(Ns, ratios) if r is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="o", label=f"L={L}")

    ax.set_xlabel("N (document count)")
    ax.set_ylabel("Realization ratio (realized / analytical)")
    ax.set_title("F3: Realization ratio vs N (sdpa, ddi_qf mask)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GEN / "F3.pdf")
    plt.close(fig)
    print("[render] F3.pdf written")


# ---- F5: throughput bar chart (ddi_cache vs ddi_recompute vs full_causal) ---

def render_f5() -> None:
    d = load("cache_throughput")
    qps = {t["metric"]: t["data"] for t in d["tables"]}["qps"]
    model = "Qwen/Qwen2.5-7B-Instruct"

    modes = ["full_causal", "ddi_recompute", "ddi_cache"]
    labels = ["full causal", "ddi recompute", "ddi cache"]
    values = []
    for mode in modes:
        try:
            # Each mode has one arm
            arm_data = qps[model][mode]
            v = list(arm_data.values())[0]
            values.append(v)
        except (KeyError, TypeError, IndexError):
            values.append(0)

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#888888", "#5599cc", "#22aa66"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("QPS (queries/sec)")
    ax.set_title("F5: Throughput under document reuse (Zipf a=1.5)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(GEN / "F5.pdf")
    plt.close(fig)
    print("[render] F5.pdf written")


# ---- T4: finetune results (base vs seq-tuned vs mixed-tuned quality + restart) -

def render_t4() -> None:
    """Quality and restart results across base, sequential-tuned, mixed-tuned 7B.

    Numbers are hardcoded from the VIBEQUEUE-2 resolution chain (2026-08-27).
    Source: results/runs.jsonl, 100 records per cell, EM metric.
    """
    rows = [
        # (checkpoint, mask, dataset, EM)
        ("base",       "full",   "hotpot", 0.57),
        ("base",       "ddi_qf", "hotpot", 0.31),
        ("base",       "full",   "musique", 0.29),
        ("base",       "ddi_qf", "musique", 0.05),
        ("seq-tuned",  "full",   "hotpot", 0.46),
        ("seq-tuned",  "ddi_qf", "hotpot", 0.46),
        ("seq-tuned",  "full",   "musique", 0.23),
        ("seq-tuned",  "ddi_qf", "musique", 0.13),
        ("mixed-tuned","full",   "hotpot", 0.40),
        ("mixed-tuned","ddi_qf", "hotpot", 0.39),
        ("mixed-tuned","full",   "musique", 0.22),
        ("mixed-tuned","ddi_qf", "musique", 0.12),
    ]

    restart_rows = [
        # (checkpoint, layout, dataset, EM)
        ("base",        "sequential", "hotpot", 0.31),
        ("base",        "restart",    "hotpot", 0.07),
        ("seq-tuned",   "sequential", "hotpot", 0.46),
        ("seq-tuned",   "restart",    "hotpot", 0.06),
        ("mixed-tuned", "sequential", "hotpot", 0.39),
        ("mixed-tuned", "restart",    "hotpot", 0.07),
    ]

    checkpoints = ["base", "seq-tuned", "mixed-tuned"]
    masks = ["full", "ddi_qf"]
    datasets = ["hotpot", "musique"]

    def lookup(ckpt, mask, ds):
        for c, m, d, em in rows:
            if c == ckpt and m == mask and d == ds:
                return f"{em:.2f}"
        return "—"

    lines = [
        r"\begin{tabular}{ll|cc|cc}",
        r" & & \multicolumn{2}{c|}{hotpot} & \multicolumn{2}{c}{musique} \\",
        r" checkpoint & mask & EM & $\Delta$ & EM & $\Delta$ \\",
        r"\hline",
    ]
    for ckpt in checkpoints:
        for j, mask in enumerate(masks):
            row_parts = []
            if j == 0:
                row_parts.append(r"\multirow{2}{*}{" + ckpt + r"}")
            else:
                row_parts.append("")
            row_parts.append(mask)
            for ds in datasets:
                row_parts.append(lookup(ckpt, mask, ds))
                # delta = full - ddi_qf for this checkpoint+dataset
                full_v = next((em for c,m,d,em in rows if c==ckpt and m=="full" and d==ds), None)
                ddi_v = next((em for c,m,d,em in rows if c==ckpt and m=="ddi_qf" and d==ds), None)
                if full_v is not None and ddi_v is not None:
                    delta = full_v - ddi_v
                    row_parts.append(f"{delta:+.2f}")
                else:
                    row_parts.append("—")
            lines.append(" & ".join(row_parts) + r" \\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    (GEN / "T4.tex").write_text("\n".join(lines))
    print("[render] T4.tex written")

    # Restart quality table
    def lookup_restart(ckpt, layout):
        for c, l, d, em in restart_rows:
            if c == ckpt and l == layout and d == "hotpot":
                return f"{em:.2f}"
        return "—"

    rlines = [
        r"\begin{tabular}{l|cc|c}",
        r" & \multicolumn{2}{c|}{hotpot EM} & \\",
        r" checkpoint & sequential & restart & $\Delta$ \\",
        r"\hline",
    ]
    for ckpt in checkpoints:
        seq_v = next((em for c,l,d,em in restart_rows if c==ckpt and l=="sequential" and d=="hotpot"), None)
        rst_v = next((em for c,l,d,em in restart_rows if c==ckpt and l=="restart" and d=="hotpot"), None)
        rlines.append(f"{ckpt} & {lookup_restart(ckpt,'sequential')} & {lookup_restart(ckpt,'restart')} & {f'{seq_v-rst_v:+.2f}' if seq_v is not None and rst_v is not None else '—'}" + r" \\")
    rlines.append(r"\end{tabular}")
    (GEN / "T4_restart.tex").write_text("\n".join(rlines))
    print("[render] T4_restart.tex written")


if __name__ == "__main__":
    render_t2()
    render_t3()
    render_f3()
    render_f5()
    render_t4()
    print("[render] done")
