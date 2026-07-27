"""Loader registry (issue #9, Phase 1.1).

Registers dataset loaders by name and dispatches `load()` to them. Every registered loader yields
records of the common shape documented in `ddi/data/__init__.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

Split = Literal["train", "val", "test"]
Loader = Callable[[Split, int], Iterator[dict]]

_REGISTRY: dict[str, Loader] = {}


def register_loader(name: str) -> Callable[[Loader], Loader]:
    """Register a loader under `name`; duplicate registration raises ValueError."""

    def decorator(fn: Loader) -> Loader:
        if name in _REGISTRY:
            raise ValueError(f"loader {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def load(name: str, split: Split, seed: int) -> Iterator[dict]:
    """Dispatch to a registered loader; unknown name raises KeyError."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; registered: {list_datasets()}")
    return _REGISTRY[name](split, seed)


def list_datasets() -> list[str]:
    """Sorted registered dataset names."""
    return sorted(_REGISTRY)
