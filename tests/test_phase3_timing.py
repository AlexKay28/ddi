"""Tests for the Phase 3 timing sweep runner (issues #18, #19, #20).

The runner itself needs a real HF model to end-to-end exercise, so those paths stay behind
`DDI_INTEGRATION=1` in the timing harness tests. Here we cover the pure logic that carries
the paper's guarantees:

- runner registration under `task: phase3_timing`
- analytical speedup matches RESEARCH_PLAN §7 (N=13, L=512, Q=64 -> 10.59x)
- `full` variant collapses to speedup = 1.0 exactly (correctness gate at N=1 or full)
- sparsity matches the §7 table (0.906 at N=13, L=512, Q=64)
- FA2-on-CPU raises before any model download starts
"""

from __future__ import annotations

import pytest
import torch


def test_runner_is_registered() -> None:
    """Side-effect import of the runners package must register `phase3_timing`."""
    import ddi.runners  # noqa: F401 — force side-effect registration
    from ddi.cli import _RUNNERS

    assert "phase3_timing" in _RUNNERS


# ---------------------------------------------------------------------------
# Analytical speedup — the RESEARCH_PLAN §7 targets that F3 plots against.
# ---------------------------------------------------------------------------


def test_analytical_speedup_full() -> None:
    """`full` at any shape is exactly 1.0 — the correctness gate."""
    from ddi.runners.phase3_timing import _analytical_speedup

    assert _analytical_speedup("full", N=1, L=512, Q=64) == 1.0
    assert _analytical_speedup("full", N=13, L=512, Q=64) == 1.0
    assert _analytical_speedup("full", N=25, L=2048, Q=64) == 1.0


def test_analytical_speedup_ddi_qf_N13() -> None:
    """N=13, L=512, Q=64 -> 10.59x per RESEARCH_PLAN §7 table (within 1%)."""
    from ddi.runners.phase3_timing import _analytical_speedup

    got = _analytical_speedup("ddi_qf", N=13, L=512, Q=64)
    assert got == pytest.approx(10.59, rel=0.01)


def test_analytical_speedup_ddi_qf_N1_is_one() -> None:
    """N=1 collapse: single doc has no cross-doc leakage to isolate, so speedup -> 1."""
    from ddi.runners.phase3_timing import _analytical_speedup

    got = _analytical_speedup("ddi_qf", N=1, L=512, Q=64)
    assert got == pytest.approx(1.0, rel=0.01)


# ---------------------------------------------------------------------------
# Sparsity — matches the §7 table too (relative to causal-pair count).
# ---------------------------------------------------------------------------


def test_sparsity_full_is_zero() -> None:
    """A full-causal allow-matrix removes nothing from the causal budget."""
    from ddi.masks import build_mask
    from ddi.runners.phase3_timing import _sparsity

    allow = build_mask([512] * 13, query_len=64, answer_len=0, variant="full")
    assert _sparsity(allow) == pytest.approx(0.0, abs=1e-9)


def test_sparsity_matches_target_ddi_qf_N13() -> None:
    """RESEARCH_PLAN §7 table: 0.906 sparsity at N=13, L=512, Q=64 for ddi_qf."""
    from ddi.masks import build_mask
    from ddi.runners.phase3_timing import _sparsity

    allow = build_mask([512] * 13, query_len=64, answer_len=0, variant="ddi_qf")
    assert _sparsity(allow) == pytest.approx(0.906, abs=0.01)


# ---------------------------------------------------------------------------
# FA2 on CPU — must fail BEFORE load_model would try to download anything.
# ---------------------------------------------------------------------------


def test_fa2_on_cpu_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`backend: fa2` on a CPU-only worker must raise RuntimeError before load_model runs."""
    from ddi.runners import phase3_timing

    # Force the CPU path even on machines that happen to have CUDA visible in CI.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # If load_model were reached, it would try to download the 7B checkpoint — the test would
    # hang or fail with a network error. A clear RuntimeError mentioning CUDA proves the
    # short-circuit fires first.
    config = {
        "task": "phase3_timing",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "mask": {"variant": "ddi_qf", "bridge_k": None},
        "N": 13,
        "L": 512,
        "Q": 64,
        "backend": "fa2",
    }
    with pytest.raises(RuntimeError, match="CUDA"):
        phase3_timing._run(config, "cfg-hash", seed=0)


# ---------------------------------------------------------------------------
# Config validation — minimal surface, mirrors phase2_train_free's shape.
# ---------------------------------------------------------------------------


def test_config_missing_mask_key_raises() -> None:
    from ddi.runners.phase3_timing import _run

    with pytest.raises(ValueError, match="missing required key: mask"):
        _run({"model": "irrelevant", "N": 1, "L": 1, "Q": 1}, "cfg", 0)


def test_config_missing_variant_raises() -> None:
    from ddi.runners.phase3_timing import _run

    with pytest.raises(ValueError, match="missing required key: variant"):
        _run(
            {"model": "irrelevant", "mask": {}, "N": 1, "L": 1, "Q": 1},
            "cfg",
            0,
        )
