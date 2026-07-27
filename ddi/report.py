"""Report generation from the store only (issue #8, Phase 0.7).

Renders every table and figure from `results/runs.jsonl`. Ground rule 3: a number absent from the
store cannot appear in output. Regenerating a table twice must be byte-identical.
"""

from __future__ import annotations

from pathlib import Path

from ddi.store import DEFAULT_STORE


def render_table(name: str, store: Path = DEFAULT_STORE) -> str:
    raise NotImplementedError("issue #8")


def render_figure(name: str, output: Path, store: Path = DEFAULT_STORE) -> None:
    raise NotImplementedError("issue #8")
