"""CLI over `ddi.delta` (issue #12, Phase 1.4).

    ddi-delta --store results/runs.jsonl \\
              --model Qwen/Qwen2.5-7B \\
              --dataset nq \\
              --metric em \\
              [--k 2.0] [--record]

Prints the computed Delta. `--record` also writes the per-seed means and the derived Δ
back into the store so downstream reports can pull it via `ddi.report`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from ddi.delta import compute_delta, format_delta, record_delta
from ddi.provenance import capture
from ddi.store import DEFAULT_STORE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi-delta")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument(
        "--record",
        action="store_true",
        help="write per-seed means and Δ back into the store for auditability",
    )
    args = parser.parse_args(argv)

    delta = compute_delta(
        store=args.store,
        model=args.model,
        dataset=args.dataset,
        metric=args.metric,
        k=args.k,
    )
    print(format_delta(delta))

    if args.record:
        # Config hash is derived from the CLI arguments so re-running with the same
        # inputs yields the same identity in the store. Seed is fixed to 0 for the
        # aggregate row (individual per-seed means carry their own seed in the run_id).
        payload = (
            f"delta|{args.model}|{args.dataset}|{args.metric}|k={args.k}"
        ).encode("utf-8")
        config_hash = hashlib.sha256(payload).hexdigest()[:16]
        prov = capture(
            config_hash=config_hash,
            seed=0,
            model_revision=args.model,
            dataset_revision=args.dataset,
        )
        record_delta(store=args.store, delta=delta, provenance=prov)
        print(f"recorded to {args.store}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
