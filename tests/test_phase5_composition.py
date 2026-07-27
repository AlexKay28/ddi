"""Tests for the Phase 5.3 composition-probe runner (issue #28).

End-to-end runner behaviour needs a real model and is gated behind `DDI_INTEGRATION=1`.
The unit tests here cover the small pieces that own the interesting invariants: the runner
is registered, config validation fires on the usual bad inputs, and the run_id encodes both
probe knobs (`distance`, `pair_position`) unambiguously.
"""

from __future__ import annotations

import pytest


def test_runner_is_registered() -> None:
    """Side-effect import of the runners package must register `phase5_composition`."""
    import ddi.runners  # noqa: F401 — force side-effect registration
    from ddi.cli import _RUNNERS

    assert "phase5_composition" in _RUNNERS


def test_config_missing_mask_key_raises() -> None:
    from ddi.runners.phase5_composition import _run

    with pytest.raises(ValueError, match="missing required key: mask"):
        _run({"dataset": "composition", "model": "irrelevant"}, "cfg", 0)


def test_config_missing_variant_raises() -> None:
    from ddi.runners.phase5_composition import _run

    with pytest.raises(ValueError, match="missing required key: variant"):
        _run(
            {"dataset": "composition", "model": "irrelevant", "mask": {}},
            "cfg",
            0,
        )


def test_config_bridge_variant_without_bridge_k_raises() -> None:
    from ddi.runners.phase5_composition import _run

    with pytest.raises(ValueError, match="requires mask.bridge_k"):
        _run(
            {
                "dataset": "composition",
                "model": "irrelevant",
                "mask": {"variant": "ddi_bridge_k"},
            },
            "cfg",
            0,
        )


def test_config_bridge_k_without_bridge_variant_raises() -> None:
    from ddi.runners.phase5_composition import _run

    with pytest.raises(ValueError, match="only valid for variant='ddi_bridge_k'"):
        _run(
            {
                "dataset": "composition",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf", "bridge_k": 4},
            },
            "cfg",
            0,
        )


def test_config_unknown_layout_raises() -> None:
    from ddi.runners.phase5_composition import _run

    with pytest.raises(ValueError, match="unknown layout"):
        _run(
            {
                "dataset": "composition",
                "model": "irrelevant",
                "mask": {"variant": "ddi_qf"},
                "layout": "bogus",
            },
            "cfg",
            0,
        )


def test_run_id_encodes_probe_knobs() -> None:
    """The runner's run_id must expose distance and pair_position so T6/F7 can slice."""
    from ddi.runners.phase5_composition import _build_run_id

    rid = _build_run_id(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset="composition",
        variant_tag="ddi_qf",
        layout="restart",
        distance=3,
        pair_position="middle",
        seed=0,
        git_sha_short="abcd1234",
        record_id="comp_0_000042",
    )
    assert ":d3:" in rid
    assert ":pmiddle:" in rid
    assert rid.startswith("phase5_composition:")
    assert rid.endswith(":comp_0_000042")
    # Order-preserving: distance sits before pair_position in the run_id.
    assert rid.index(":d3:") < rid.index(":pmiddle:")
