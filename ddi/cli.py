"""CLI entry point (issue #2, Phase 0.1).

    ddi-run --config <file.yaml> --seed <n>

Loads the YAML, computes its content hash (part of run identity per ground rule 1), seeds every
RNG we know about, and dispatches to the runner named by the config's `task` key. Runners are
registered by later phases.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

Runner = Callable[[dict[str, Any], str, int], None]

_RUNNERS: dict[str, Runner] = {}


def register_runner(name: str) -> Callable[[Runner], Runner]:
    """Decorator: register a runner under a `task:` value used in configs."""

    def wrap(fn: Runner) -> Runner:
        if name in _RUNNERS:
            raise ValueError(f"runner already registered for task={name!r}")
        _RUNNERS[name] = fn
        return fn

    return wrap


# Side-effect import: pulls in every runner module so their `@register_runner` calls
# populate `_RUNNERS` before `dispatch()` is invoked. Must be below `register_runner`
# because the runner modules import that name from here.
import ddi.runners  # noqa: F401, E402


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config_hash = hashlib.sha256(raw).hexdigest()[:16]
    config = yaml.safe_load(raw) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config at {path} is not a mapping")
    return config, config_hash


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dispatch(config: dict[str, Any], config_hash: str, seed: int) -> None:
    task = config.get("task")
    if not task:
        raise ValueError("config missing required key: task")
    runner = _RUNNERS.get(task)
    if runner is None:
        raise NotImplementedError(
            f"no runner registered for task={task!r} — known: {sorted(_RUNNERS)}"
        )
    runner(config, config_hash, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ddi-run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    config, config_hash = load_config(args.config)
    set_seed(args.seed)
    dispatch(config, config_hash, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
