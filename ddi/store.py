"""Append-only results store (issue #7, Phase 0.6).

One JSONL row per `(run_id, metric, value)`. Committed. Ground rule 2. No number appears in a
generated table or figure that is not in this store (ground rule 3).
"""

from __future__ import annotations

from pathlib import Path

from ddi.provenance import Provenance

DEFAULT_STORE = Path("results/runs.jsonl")


def append(
    run_id: str,
    metric: str,
    value: float,
    provenance: Provenance,
    store: Path = DEFAULT_STORE,
) -> None:
    raise NotImplementedError("issue #7")
