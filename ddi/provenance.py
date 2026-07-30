"""Run identity capture (issue #7, Phase 0.6).

Ground rule 1: every metric row carries git SHA (with dirty flag), config hash, seed, model
revision, dataset revision, GPU model, driver/CUDA/torch versions, and UTC timestamp.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import torch


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


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _git_or(sentinel: str) -> tuple[str, bool]:
    """Return `(git_sha, dirty)`, or `(sentinel, False)` if we're outside a git repo.

    Serverless containers ship the code as plain files with no `.git` dir; the earlier
    behaviour (unconditional `git rev-parse` with `check=True`) crashed every runner cell
    on such hosts, and per-cell isolation silently swallowed the failure — the aggregator
    then reported "no rows in store" without any hint that git was the culprit. The
    sidecar helper in `ddi/reports/_provenance.py::_git_sha` uses the same fallback.
    """
    if shutil.which("git") is None:
        return sentinel, False
    try:
        git_sha = _run(["git", "rev-parse", "HEAD"]).strip()
        dirty = bool(_run(["git", "status", "--porcelain"]).strip())
    except (subprocess.CalledProcessError, OSError):
        return sentinel, False
    return git_sha or sentinel, dirty


def capture(
    config_hash: str,
    seed: int,
    model_revision: str,
    dataset_revision: str,
) -> Provenance:
    git_sha, dirty = _git_or("not-a-git-repo")

    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        # nvidia-smi may be absent in CI containers with torch+CUDA installed.
        if shutil.which("nvidia-smi") is None:
            driver_version = "unknown"
        else:
            driver_version = _run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ).strip()
    else:
        gpu_model = "cpu"
        driver_version = "unknown"

    return Provenance(
        git_sha=git_sha,
        dirty=dirty,
        config_hash=config_hash,
        seed=seed,
        model_revision=model_revision,
        dataset_revision=dataset_revision,
        gpu_model=gpu_model,
        driver_version=driver_version,
        cuda_version=torch.version.cuda or "cpu",
        torch_version=torch.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
