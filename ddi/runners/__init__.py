"""Runner registry side-effect imports (issue #12, Phase 1.4).

Importing this package registers every runner in `ddi/runners/` with `ddi.cli._RUNNERS` via
`@register_runner`. `ddi.cli` imports this package at module top so `dispatch()` can find a
runner by name from a config's `task:` key.
"""

from __future__ import annotations

from ddi.runners import phase1_baseline as _phase1_baseline  # noqa: F401
from ddi.runners import phase2_finetune as _phase2_finetune  # noqa: F401
from ddi.runners import phase2_train_free as _phase2_train_free  # noqa: F401
from ddi.runners import phase4_cache_correctness as _phase4_cache_correctness  # noqa: F401
from ddi.runners import phase4_restart_quality as _phase4_restart_quality  # noqa: F401

__all__: list[str] = []
