"""Persistent record cache for dataset loaders.

Every ddi/data loader normally re-fetches its records from the HuggingFace hub each
time ``load(name, split, seed)`` is called. On an offline serverless container
(``HF_HUB_OFFLINE=1``) that first fetch fails hard; even on a networked host it wastes
minutes and bandwidth on every run.

This module persists the normalized records emitted by each loader to
``results/splits/<name>__seed<seed>__<split>_records.jsonl`` immediately after the
first HF-hub fetch. Subsequent invocations (from any container, including offline
ones) stream those files back from disk without touching the network. The manifest
JSON produced by :func:`ddi.data.manifest.validate_or_write` is separate — it stores
count+hash for auditing, not the records themselves — and both artefacts can be
shipped together from a networked build node to an air-gapped runner.

Two-tier "cached" check (file exists + non-zero size) catches a partial-write
interruption cleanly: the loader falls back to the HF path and re-persists rather
than yielding a truncated split.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ddi.data import manifest
from ddi.data.loaders import Split


def _cache_root() -> Path:
    """Late-bind against :data:`ddi.data.manifest.DEFAULT_ROOT`.

    Sharing the manifest's root has two useful effects: (a) both artefacts —
    audit-hash JSON and record cache — sit in the same directory and get shipped
    together to the offline runner; (b) any test that monkeypatches
    ``manifest.DEFAULT_ROOT`` (there are several) redirects the cache too, so
    tests stay hermetic without a second fixture.
    """
    return manifest.DEFAULT_ROOT


def records_path(name: str, seed: int, split: Split) -> Path:
    """Filesystem location of the records file for one (loader, seed, split)."""
    return _cache_root() / f"{name}__seed{seed}__{split}_records.jsonl"


def has_cached(name: str, seed: int, splits: Iterable[Split]) -> bool:
    """True iff every listed split has a non-empty records file on disk.

    An empty (0-byte) file is treated as *not cached* so a crashed writer that
    only created the file before dying does not fool the loader into yielding
    zero records forever.
    """
    for s in splits:
        p = records_path(name, seed, s)
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def write_split(name: str, seed: int, split: Split, records: list[dict]) -> None:
    """Serialise ``records`` to the cache file for one (loader, seed, split).

    One JSON object per line; the parent directory is created on demand so the
    caller doesn't have to know about ``results/splits/``.
    """
    path = records_path(name, seed, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_split(name: str, seed: int, split: Split) -> Iterator[dict]:
    """Stream previously-cached records back one at a time.

    Blank lines are tolerated (a trailing newline in the file is harmless) but
    every non-blank line must parse as JSON — a truncated write surfaces as a
    ``json.JSONDecodeError`` rather than silently yielding a partial split.
    """
    with records_path(name, seed, split).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
