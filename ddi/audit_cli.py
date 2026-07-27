"""ddi-audit CLI (issue #10, Phase 1.2).

Two subcommands wrap `ddi.hop_labels` for humans:

    ddi-audit sample --dataset X --split val --seed 0 --n 100 --out audit_X.jsonl
    ddi-audit score  --dataset X --split val --seed 0 --human audit_X.jsonl

`sample` writes a JSONL of records to review (one per line). `score` re-samples the same records,
applies any registered heuristic to produce auto labels, and reports agreement with the reviewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ddi.hop_labels import (
    Hop,
    agreement_rate,
    apply_heuristic,
    load_audit,
    sample_for_audit,
)

# Truncate context strings written into the audit file so a 100-record sample is scannable in a
# text editor. The auditor rarely needs more than a few hundred chars to decide single vs. multi.
_CONTEXT_CHARS = 400


def _last_user_turn(dialog: list[dict]) -> str:
    for msg in reversed(dialog):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _truncate(text: str, limit: int = _CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _sample(args: argparse.Namespace) -> int:
    records = sample_for_audit(args.dataset, args.split, args.seed, args.n)
    out_path = Path(args.out)
    with out_path.open("w") as f:
        for rec in records:
            row = {
                "id": rec["id"],
                "dialog": _last_user_turn(rec.get("dialog", [])),
                "info_contexts": [_truncate(c) for c in rec.get("info_contexts", [])],
                "possible_answers": rec.get("possible_answers", []),
                "hop_type": None,
            }
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 0


def _score(args: argparse.Namespace) -> int:
    sampled = sample_for_audit(args.dataset, args.split, args.seed, args.n)
    auto: dict[str, Hop] = {
        rec["id"]: rec["hop_type"]  # type: ignore[misc]
        for rec in apply_heuristic(args.dataset, sampled)
    }
    human = load_audit(Path(args.human))

    n_sampled = len(sampled)
    n_reviewed = len(human)
    n_skipped = n_sampled - n_reviewed

    rate = agreement_rate(auto, human)
    n_hits = round(rate * n_reviewed)

    print(
        f"dataset: {args.dataset}, split: {args.split}, seed: {args.seed}, "
        f"n_reviewed: {n_reviewed}"
    )
    print(f"auto vs human agreement: {rate * 100:.1f}% ({n_hits} / {n_reviewed})")
    if n_skipped:
        print(f"skipped (unfilled human labels): {n_skipped}")

    disagreements = [
        (rid, auto.get(rid, "<missing>"), label)
        for rid, label in human.items()
        if auto.get(rid) != label
    ]
    if disagreements:
        print("disagreements (id  auto  human):")
        for rid, a, h in disagreements:
            print(f"  {rid}  {a}  {h}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi-audit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample_p = sub.add_parser("sample", help="sample records for hand-audit")
    sample_p.add_argument("--dataset", required=True)
    sample_p.add_argument("--split", required=True)
    sample_p.add_argument("--seed", type=int, required=True)
    sample_p.add_argument("--n", type=int, required=True)
    sample_p.add_argument("--out", required=True)
    sample_p.set_defaults(func=_sample)

    score_p = sub.add_parser("score", help="score auto-vs-human agreement on a filled audit file")
    score_p.add_argument("--dataset", required=True)
    score_p.add_argument("--split", required=True)
    score_p.add_argument("--seed", type=int, required=True)
    score_p.add_argument("--n", type=int, required=True)
    score_p.add_argument("--human", required=True)
    score_p.set_defaults(func=_score)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
