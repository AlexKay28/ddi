"""Dataset framework (issue #9, Phase 1.1).

Frozen inputs on which every quality claim is measured. Ground rule 1: every metric row carries a
`dataset_revision`; ground rule 3: tables regenerate from `results/runs.jsonl`. Loaders yield a
common record shape and their split contents are enforced identical run-to-run by the split-hash
manifest at `results/splits/<name>.json`.
"""

from __future__ import annotations

from ddi.data.loaders import list_datasets, load, register_loader

# Import loader modules for their register_loader side-effects. Order matters: loaders.py must
# import first so the registry decorator exists. The ``composition`` and ``ddi_stress`` loaders
# were dropped from scope in #89 alongside their runners / renderers; ``kilt_nq`` and ``nq``
# were deleted in VIBEQUEUE-16 because neither could produce a usable measurement — see the
# 2026-08-08 CHANGELOG entry for both post-mortems, and
# ``ddi.aggregate._QUALITY_DROPPED_DATASETS`` for the rows they left behind.
from ddi.data import hotpot_qa as _hotpot_qa  # noqa: F401, E402
from ddi.data import musique as _musique  # noqa: F401, E402
from ddi.data import synthetic as _synthetic  # noqa: F401, E402

__all__ = ["list_datasets", "load", "register_loader"]
