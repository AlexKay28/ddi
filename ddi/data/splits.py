"""Deterministic dataset splits and split hashes (issue #9, Phase 1.1).

`hash_split` sorts IDs before hashing so the yield order of a loader cannot silently invalidate a
committed manifest — only the *set* of IDs in a split defines its identity.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Literal

Split = Literal["train", "val", "test"]


def split_ids(
    all_ids: Sequence[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[Split, list[str]]:
    """Deterministic permutation via `random.Random(seed)`, partitioned into train/val/test."""
    if not 0.0 <= val_frac <= 1.0 or not 0.0 <= test_frac <= 1.0:
        raise ValueError("val_frac and test_frac must be in [0, 1]")
    if val_frac + test_frac > 1.0:
        raise ValueError(f"val_frac + test_frac = {val_frac + test_frac} > 1.0")

    ids = list(all_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test

    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def hash_split(ids: Sequence[str]) -> str:
    """sha256 over sorted IDs; 16 hex chars is plenty for split-identity."""
    joined = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]
