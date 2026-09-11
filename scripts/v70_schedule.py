"""Dependency-free, process-stable common schedule for V70 arms."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RecordSchedule:
    encoding: Literal["monolithic", "independent"]
    shuffle_key: str
    perturb: bool


def stable_schedule(
    record_id: str,
    step: int,
    seed: int,
    *,
    independent_rate: float = 0.15,
) -> RecordSchedule:
    """Return an arm-independent SHA-256 schedule for one record and step."""
    if not record_id or step < 0 or not 0.0 <= independent_rate <= 1.0:
        raise ValueError("invalid V70 schedule arguments")
    digest = hashlib.sha256(f"v70\0{seed}\0{step}\0{record_id}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return RecordSchedule(
        encoding="independent" if unit < independent_rate else "monolithic",
        shuffle_key=digest.hex(),
        perturb=unit < independent_rate,
    )
