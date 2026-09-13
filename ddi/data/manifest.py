"""Split-hash manifest (issue #9, Phase 1.1).

Enforces the frozen-splits discipline: first load writes `results/splits/<dataset>.json`;
subsequent loads validate against it. A mismatch is a bug (ground rule 1: `dataset_revision` on
every metric row would otherwise be a lie), so we raise loudly.

`dataset` is an opaque manifest key, not necessarily the bare loader name. Loaders whose split
partition is itself a function of `seed` (synthetic_qa, hotpot_qa, musique — ground rule 5
requires ≥3 seeds, and their train/val/test membership legitimately differs per seed) key
their manifest as `f"{name}__seed{seed}"` so each seed freezes
its own split instead of colliding with seed 0's (issue #40).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

Split = Literal["train", "val", "test"]

DEFAULT_ROOT = Path("results/splits")


def _resolve(root: Path | None) -> Path:
    # Late-bind so tests can monkeypatch DEFAULT_ROOT.
    return DEFAULT_ROOT if root is None else root


def read(dataset: str, root: Path | None = None) -> dict | None:
    path = _resolve(root) / f"{dataset}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write(
    dataset: str,
    splits: dict[Split, tuple[int, str]],
    root: Path | None = None,
) -> None:
    resolved = _resolve(root)
    resolved.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "splits": {name: {"n": n, "sha256": sha} for name, (n, sha) in splits.items()},
    }
    path = resolved / f"{dataset}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_or_write(
    dataset: str,
    splits: dict[Split, tuple[int, str]],
    root: Path | None = None,
) -> None:
    existing = read(dataset, root=root)
    if existing is None:
        write(dataset, splits, root=root)
        return

    stored = existing.get("splits", {})
    for name, (n, sha) in splits.items():
        entry = stored.get(name)
        if entry is None:
            raise RuntimeError(
                f"{dataset}/{name}: expected split absent from manifest; got n={n} sha={sha}"
            )
        if entry["n"] != n or entry["sha256"] != sha:
            raise RuntimeError(
                f"{dataset}/{name}: expected n={entry['n']} sha={entry['sha256']}, "
                f"got n={n} sha={sha}"
            )


def select_or_write(
    dataset: str,
    ids: list[str],
    limit: int | None,
    root: Path,
    *,
    allow_create: bool = True,
) -> list[str]:
    """Freeze a sorted held-out ID selection, then require those IDs on later runs.

    Unlike the split manifest, this stores the IDs themselves.  A quality run can
    therefore reproduce its held-out examples even when a loader changes yield order. Set
    ``allow_create=False`` to verify an existing manifest without creating one.
    """
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: held-out source IDs must be unique")
    if limit is not None and limit < 1:
        raise ValueError(f"{dataset}: held-out limit must be >= 1, got {limit}")

    path = root / f"{dataset}.json"
    available = set(ids)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload.get("ids")
        if payload.get("dataset") != dataset or not isinstance(selected, list):
            raise RuntimeError(f"{path}: malformed held-out manifest")
        if limit != payload.get("limit"):
            raise RuntimeError(
                f"{dataset}: held-out limit changed from {payload.get('limit')} to {limit}"
            )
        missing = sorted(set(selected) - available)
        if missing:
            raise RuntimeError(f"{dataset}: held-out IDs absent from loader: {missing}")
        return selected

    if not allow_create:
        raise FileNotFoundError(f"{path}: held-out manifest is required in verify-only mode")
    selected = sorted(ids) if limit is None else sorted(ids)[:limit]
    root.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "ids": selected, "limit": limit}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selected
