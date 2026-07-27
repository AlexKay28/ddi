"""Report generation from the store only (issue #8, Phase 0.7).

Renders every table and figure in the paper from `results/runs.jsonl`. Ground rule 3: a number
absent from the store cannot appear in output. Regenerating a table twice from identical store
contents must be byte-identical (no timestamps, stable sort keys, file-order iteration).

This module provides only the registry, primitives, and dispatch. Concrete renderers for T2, T3,
and F2-F6 register themselves from later-phase modules on import, as their input data lands.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from ddi.store import DEFAULT_STORE

# Table renderers take (store) -> str (markdown). Figure renderers take (store, output) -> None.
TableRenderer = Callable[[Path], str]
FigureRenderer = Callable[[Path, Path], None]
Renderer = TableRenderer | FigureRenderer

_RENDERERS: dict[str, Renderer] = {}


def register(name: str) -> Callable[[Renderer], Renderer]:
    """Decorator: register a renderer under a name like `T2` or `F3`.

    Names starting with `T` are tables (return markdown), names starting with `F` are figures
    (write to an output path). Anything else is accepted for tests/other artifacts.
    """

    def wrap(fn: Renderer) -> Renderer:
        if name in _RENDERERS:
            raise ValueError(f"renderer already registered for name={name!r}")
        _RENDERERS[name] = fn
        return fn

    return wrap


def list_reports() -> list[str]:
    return sorted(_RENDERERS)


def render(name: str, store: Path = DEFAULT_STORE, output: Path | None = None) -> str | None:
    if name not in _RENDERERS:
        raise KeyError(f"unknown report {name!r} — known: {list_reports()}")
    fn = _RENDERERS[name]
    if _is_figure(name):
        if output is None:
            raise ValueError(f"figure renderer {name!r} requires an output path")
        fn(store, output)  # type: ignore[call-arg]
        return None
    return fn(store)  # type: ignore[call-arg]


def _is_figure(name: str) -> bool:
    return name.startswith("F")


# ---- Store reader --------------------------------------------------------------------------


def iter_rows(store: Path = DEFAULT_STORE) -> Iterator[dict]:
    """Yield rows from the JSONL store in file order.

    The store is append-only, so file order is deterministic and is the canonical order for any
    downstream renderer that needs stability.
    """
    with open(store, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_rows(
    rows: Iterable[dict],
    *,
    metric: str | None = None,
    run_id: str | None = None,
    provenance: dict | None = None,
) -> Iterator[dict]:
    """Filter rows by metric, run_id, and/or a subset match on the nested provenance dict."""
    for row in rows:
        if metric is not None and row.get("metric") != metric:
            continue
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if provenance is not None:
            prov = row.get("provenance") or {}
            if not all(prov.get(k) == v for k, v in provenance.items()):
                continue
        yield row


# ---- Utility helpers -----------------------------------------------------------------------


def median_with_iqr(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (median, Q1, Q3) using linear interpolation between order statistics."""
    if not values:
        raise ValueError("median_with_iqr requires at least one value")
    xs = sorted(values)
    return float(median(xs)), _quantile(xs, 0.25), _quantile(xs, 0.75)


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    n = len(sorted_xs)
    if n == 1:
        return float(sorted_xs[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * frac)


def group_by(rows: Iterable[dict], keys: Sequence[str]) -> dict[tuple, list[dict]]:
    """Group rows by the tuple of `row[k]` for `k in keys`.

    Dotted paths (e.g. `"provenance.git_sha"`) traverse nested dicts.
    """
    out: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(_lookup(row, k) for k in keys)
        out.setdefault(key, []).append(row)
    return out


def _lookup(row: dict, path: str) -> Any:
    node: Any = row
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node
