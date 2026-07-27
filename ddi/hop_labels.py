"""Per-example hop-type labels and hand-audit workflow (issue #10, Phase 1.2).

Phase 5 (§5, `paper/RESEARCH_PLAN.md`) splits every result by single-doc vs. multi-doc, so every
record must carry a trustworthy `hop_type`. Native labels win when a dataset provides them
(HotpotQA's bridge/comparison, MoNaCo's all-multi construction); otherwise a per-dataset
heuristic labeler fills the gap. The audit workflow measures how much to trust either source:
sample ~100 records, hand-label them, compare against the automatic label — the acceptance
criterion in issue #1 §1.2.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Literal

from ddi.data import load
from ddi.data.loaders import Split

Hop = Literal["single", "multi", "unknown"]
Heuristic = Callable[[dict], Hop]

_HEURISTICS: dict[str, Heuristic] = {}


def hop_type(record: dict) -> Hop:
    """Return the record's `hop_type` field; default 'unknown' if absent."""
    value = record.get("hop_type", "unknown")
    if value not in {"single", "multi", "unknown"}:
        return "unknown"
    return value  # type: ignore[return-value]


def register_heuristic(dataset: str) -> Callable[[Heuristic], Heuristic]:
    """Decorator to register a heuristic labeler for records that arrive with hop_type='unknown'."""

    def decorator(fn: Heuristic) -> Heuristic:
        if dataset in _HEURISTICS:
            raise ValueError(f"heuristic for dataset {dataset!r} is already registered")
        _HEURISTICS[dataset] = fn
        return fn

    return decorator


def apply_heuristic(dataset: str, records: Iterable[dict]) -> Iterator[dict]:
    """Yield shallow copies with `hop_type` filled in by the dataset's heuristic when unknown.

    Non-destructive so callers can pipe a loader directly without polluting the source records.
    """
    heuristic = _HEURISTICS.get(dataset)
    for rec in records:
        out = dict(rec)
        if heuristic is not None and hop_type(out) == "unknown":
            out["hop_type"] = heuristic(out)
        yield out


def sample_for_audit(dataset: str, split: Split, seed: int, n: int) -> list[dict]:
    """Deterministically sample `n` records from `ddi.data.load(dataset, split, seed)`.

    Sorted by id after sampling so the audit file diffs cleanly across runs. If `n` exceeds the
    available population, returns everything available (no error — the caller usually just wants
    "as many as you have, up to n").
    """
    if n <= 0:
        return []
    records = list(load(dataset, split, seed))
    if n >= len(records):
        chosen = list(records)
    else:
        # random.sample would work but we need control over the RNG seed independently of
        # loader-internal shuffles. Use a stable per-(dataset, split, seed) key.
        import random as _random

        rng = _random.Random(f"audit::{dataset}::{split}::{seed}")
        chosen = rng.sample(records, n)
    chosen.sort(key=lambda r: r["id"])
    return chosen


def load_audit(path: Path) -> dict[str, Hop]:
    """Read a completed audit JSONL. Skip entries whose hop_type is null or missing."""
    labels: dict[str, Hop] = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("hop_type")
            if label is None:
                continue
            if label not in {"single", "multi", "unknown"}:
                raise ValueError(
                    f"audit row for id={row.get('id')!r} has invalid hop_type={label!r}"
                )
            labels[row["id"]] = label  # type: ignore[assignment]
    return labels


def agreement_rate(auto: dict[str, Hop], human: dict[str, Hop]) -> float:
    """Fraction of ids in `human` where auto.get(id) == human[id]. ValueError if human is empty."""
    if not human:
        raise ValueError("human labels are empty; cannot compute agreement rate")
    hits = sum(1 for rid, label in human.items() if auto.get(rid) == label)
    return hits / len(human)
