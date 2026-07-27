"""Tests for the Phase 5.2 Parallel Context Windows baseline runner (issue #27).

End-to-end runner behaviour needs a real model; these unit tests cover the small,
model-free parts: registration, the `run_id_prefix` shape (so the paper's PCW column
keys cleanly), and config validation on the two required keys.
"""

from __future__ import annotations

import pytest


def test_runner_is_registered() -> None:
    """Side-effect import of the runners package must register `phase5_pcw`."""
    import ddi.runners  # noqa: F401 — force side-effect registration
    from ddi.cli import _RUNNERS

    assert "phase5_pcw" in _RUNNERS


def test_run_id_shape() -> None:
    """`_make_run_id_prefix` produces `phase5_pcw:{model}:{dataset}:seed=...:{sha8}`.

    No `variant_tag` segment — PCW is one fixed setup, not a family of variants.
    """
    from ddi.runners.phase5_pcw import _make_run_id_prefix

    prefix = _make_run_id_prefix(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset="nq",
        seed=0,
        git_sha="abcdef1234567890",
    )
    assert prefix == "phase5_pcw:Qwen/Qwen2.5-7B-Instruct:nq:seed=0:abcdef12"


def test_run_id_shape_uses_first_eight_sha_chars() -> None:
    """Only the first 8 characters of git_sha go into the prefix — mirrors other phases."""
    from ddi.runners.phase5_pcw import _make_run_id_prefix

    prefix = _make_run_id_prefix(
        model_name="m", dataset="d", seed=42, git_sha="0123456789abcdef"
    )
    assert prefix == "phase5_pcw:m:d:seed=42:01234567"


def test_config_missing_model_raises() -> None:
    """Missing `model` must fail before any model / dataset side effects."""
    from ddi.runners.phase5_pcw import _run

    with pytest.raises((KeyError, ValueError), match="model"):
        _run({"dataset": "nq"}, "cfg", 0)


def test_config_missing_dataset_raises() -> None:
    """Missing `dataset` must fail before any model / dataset side effects."""
    from ddi.runners.phase5_pcw import _run

    with pytest.raises((KeyError, ValueError), match="dataset"):
        _run({"model": "irrelevant"}, "cfg", 0)
