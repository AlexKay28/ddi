"""Append-only results store (issue #7, Phase 0.6).

One JSONL row per `(run_id, metric, value)`. Committed. Ground rule 2. No number appears in a
generated table or figure that is not in this store (ground rule 3).
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
from pathlib import Path

from ddi.provenance import Provenance

DEFAULT_STORE = Path("results/runs.jsonl")

# POSIX-only: fcntl.flock guards concurrent appends by parallel sweep workers.
_ALLOWED_VALUE_TYPES = (bool, int, float, str)


def append(
    run_id: str,
    metric: str,
    value: float | int | bool | str,
    provenance: Provenance,
    store: Path = DEFAULT_STORE,
) -> None:
    if not isinstance(value, _ALLOWED_VALUE_TYPES):
        raise TypeError(
            f"value for metric={metric!r} must be int/float/bool/str, got {type(value).__name__}"
        )

    store.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "metric": metric,
        "value": value,
        "provenance": dataclasses.asdict(provenance),
    }
    line = json.dumps(row) + "\n"

    with open(store, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        f.flush()
