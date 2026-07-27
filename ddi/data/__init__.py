"""Dataset framework (issue #9, Phase 1.1).

Frozen inputs on which every quality claim is measured. Ground rule 1: every metric row carries a
`dataset_revision`; ground rule 3: tables regenerate from `results/runs.jsonl`. Loaders yield a
common record shape and their split contents are enforced identical run-to-run by the split-hash
manifest at `results/splits/<name>.json`.
"""

from __future__ import annotations

from ddi.data.loaders import list_datasets, load, register_loader

# Import loader modules for their register_loader side-effects. Order matters: loaders.py must
# import first so the registry decorator exists.
from ddi.data import composition as _composition  # noqa: F401, E402
from ddi.data import musique as _musique  # noqa: F401, E402
from ddi.data import nq as _nq  # noqa: F401, E402
from ddi.data import synthetic as _synthetic  # noqa: F401, E402

__all__ = ["list_datasets", "load", "register_loader"]
