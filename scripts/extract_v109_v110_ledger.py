#!/usr/bin/env python3
"""Extract V109/V110/V101 results into the paper evidence ledger (one-shot).

Sources (sha-verified before use):
- results/v109_v110_20260915T211622Z/summary.json  (VIBEQUEUE-109/110 arms + paired CIs)
- results/v101_sdpa_dispatch_20260915T212816Z/summary.json (VIBEQUEUE-111 dispatch)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV_PATH = ROOT / "paper" / "evidence.json"

SOURCES = {
    "v109_v110_summary": "results/v109_v110_20260915T211622Z/summary.json",
    "v101_sdpa_summary": "results/v101_sdpa_dispatch_20260915T212816Z/summary.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f4(x: float) -> float:
    return round(float(x), 4)


def main() -> None:
    ev = json.loads(EV_PATH.read_text())
    loaded = {}
    for name, rel in SOURCES.items():
        path = ROOT / rel
        digest = sha256(path)
        if name in ev["sources"]:
            assert ev["sources"][name]["sha256"] == digest, f"{name} drifted"
        else:
            ev["sources"][name] = {
                "path": rel,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "required": True,
            }
        loaded[name] = json.loads(path.read_text())
        print(f"[ok] {name} pinned {digest[:16]}")

    # ---- T18: arms table ---------------------------------------------------
    ref = loaded["v109_v110_summary"]
    pinned_c = {"em": 0.4233, "f1": 0.5561}
    rows = []
    for arm in ("base-std", "k5-std", "k5-iso-seq", "k5-d-replica"):
        means = ref["arms"][arm]
        pair = ref["paired_vs_pinned_c"][arm]
        rows.append(
            [
                arm,
                f4(means["em"]),
                f4(means["f1"]),
                f4(pair["em"]["c_minus_arm_mean"]),
                f4(pair["em"]["ci95"][0]),
                f4(pair["em"]["ci95"][1]),
            ]
        )
    art = ev["artifacts"]
    art["T18_v109_v110_arms"] = {
        "sources": ["v109_v110_summary"],
        "note": (
            "Rows [arm, EM, F1, C-EM minus arm, CI low, CI high], n=300 frozen "
            "confirmation300 records, greedy 64, bf16 eager, torch 2.13.0+cu126, "
            "A100 80GB. base-std = original hub Qwen2.5-7B-Instruct, plain stream, "
            "sequential positions, full-causal (V109). k5-* arms run the trained "
            "results/v70_arms_valid_seed47/checkpoint_k5 (aggregate pin in the "
            "source). k5-d-replica is the pinned V70 D construction (replication "
            "anchor; drifted from pinned D EM 0.1833 to 0.1967 across runtime "
            "stacks - deviation recorded in experiments/v110_preregistration.md "
            "Amendment 1). Paired bootstrap 10k, seed 47 vs the pinned V70 C arm."
        ),
        "rows": rows,
    }

    # ---- T19: SDPA dispatch ------------------------------------------------
    dsp = loaded["v101_sdpa_summary"]
    drows = []
    for case in dsp["cases"]:
        kernels = case["kernels_us"]
        kernel = next(iter(kernels), "unknown")
        drows.append([case["label"], kernel, round(case["wall_s_per_call"], 3)])
    art["T19_sdpa_dispatch"] = {
        "sources": ["v101_sdpa_summary"],
        "note": (
            "Rows [mask case, dispatched kernel family, wall s per forward]. Shape "
            "N=25, L=512, Q=64 (total 12,809 tokens; 95.86% causal-pair sparsity), "
            "bf16, torch 2.13.0+cu126, A100 80GB, 3 warmup + 10 timed forwards "
            "under torch.profiler (CUDA activity). The 2D causal path dispatches "
            "SDPA flash; both 4D additive masks dispatch the SDPA memory-efficient "
            "cutlass kernel - block-sparsity is invisible to it either way."
        ),
        "rows": drows,
        "pair_sparsity": dsp["pair_sparsity"],
    }

    EV_PATH.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {EV_PATH} updated")


if __name__ == "__main__":
    main()
