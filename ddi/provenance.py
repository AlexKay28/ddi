"""Run identity capture (issue #7, Phase 0.6).

Ground rule 1: every metric row carries git SHA (with dirty flag), config hash, seed, model
revision, dataset revision, GPU model, driver/CUDA/torch versions, and UTC timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    git_sha: str
    dirty: bool
    config_hash: str
    seed: int
    model_revision: str
    dataset_revision: str
    gpu_model: str
    driver_version: str
    cuda_version: str
    torch_version: str
    timestamp_utc: str


def capture(
    config_hash: str,
    seed: int,
    model_revision: str,
    dataset_revision: str,
) -> Provenance:
    raise NotImplementedError("issue #7")
