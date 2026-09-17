#!/usr/bin/env python3
"""Extract V112 prefix-cache-redo results into the paper evidence ledger (one-shot)."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV_PATH = ROOT / "paper" / "evidence.json"

SUMMARY = sorted(glob.glob(str(ROOT / "results/v112_prefix_cache_redo_*/summary.json")))[-1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ev = json.loads(EV_PATH.read_text())
    path = Path(SUMMARY)
    digest = sha256(path)
    ev["sources"]["v112_prefix_summary"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "required": True,
    }
    summary = json.loads(path.read_text())
    assert summary["gates"]["G1_first_request_identical"] is True
    assert summary["gates"]["G2_hit_forwards_query_only"] is True
    assert summary["gates"]["G4_prefix_fixed_under_half_mono"] is True
    assert summary["gates"]["G5a_prefix_reordered_no_win"] is True
    assert summary["gates"]["G5b_ddi_reordered_total_under_mono_total"] is True

    rows = []
    for arm in ("monolithic", "prefix-fixed", "prefix-reordered", "ddi-fixed", "ddi-reordered"):
        v = summary["arms"][arm]
        rows.append(
            [
                arm,
                round(v["median_prefill_ms"], 1) if v["median_prefill_ms"] is not None else None,
                round(v["median_ttft_ms"], 1) if v["median_ttft_ms"] is not None else None,
                round(v["median_total_ms"], 1),
            ]
        )
    ev["artifacts"]["T20_prefix_cache_redo"] = {
        "sources": ["v112_prefix_summary"],
        "note": (
            "Rows [arm, median prefill ms, median TTFT ms, median total ms]. Genuine "
            "exact-prefix KV cache (VIBEQUEUE-114): hits load stored document-prefix "
            "K/V tensors and prefill only the query tail (instrumented: only query "
            "tokens forwarded on hits, G2); first miss is byte-identical to "
            "monolithic (G1). 20 dev requests (5x500-token docs, fixed doc order) x "
            "3 repeats, greedy 16, bf16 eager, untrained restart checkpoint (V66 "
            "pin). Prefix keys are ORDERED document tuples, so per-request shuffles "
            "miss every time while the DDI canonical cache (order-independent "
            "content keys) keeps its win (G5). Hit outputs match monolithic on "
            "90% of requests (bf16 reduction-order divergence across shapes; "
            "preregistered as measured-not-gated) - DDI's canonical exact identity "
            "remains the contrast."
        ),
        "rows": rows,
        "gates": summary["gates"],
        "prefix_cache": summary["prefix_cache"],
    }
    EV_PATH.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    print(f"[ok] v112_prefix_summary pinned {digest[:16]} -> T20_prefix_cache_redo")


if __name__ == "__main__":
    main()
