"""Append-only per-example quality outputs."""

from __future__ import annotations

import dataclasses
import fcntl
import json
from pathlib import Path

from ddi.provenance import Provenance


def append(
    run_id: str,
    record_id: str,
    prediction: str,
    possible_answers: list[str],
    metrics: dict[str, float],
    provenance: Provenance,
    store: Path,
) -> None:
    """Append one complete quality result, guarded for parallel workers."""
    store.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "record_id": record_id,
        "prediction": prediction,
        "possible_answers": possible_answers,
        "metrics": metrics,
        "provenance": dataclasses.asdict(provenance),
    }
    with open(store, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        f.flush()
