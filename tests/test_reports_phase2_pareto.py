"""Tests for ddi/reports/phase2_pareto.py — F6 renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render


def _write_row(store: Path, run_id: str, metric: str, value: float, seed: int = 0) -> None:
    with store.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "value": value,
                    "provenance": {"seed": seed},
                }
            )
            + "\n"
        )


def _bridge_em(store: Path, k: int, seed: int, em: float) -> None:
    rid = f"phase2_train_free:Qwen:nq:ddi_bridge_k_k{k}:seed={seed}:sha:ex_{seed}"
    _write_row(store, rid, "em", em, seed)


def _full_em(store: Path, seed: int, em: float) -> None:
    rid = f"phase2_train_free:Qwen:nq:full:seed={seed}:sha:ex_{seed}"
    _write_row(store, rid, "em", em, seed)


def _bridge_ftt(store: Path, k: int, seed: int, ftt_ms: float) -> None:
    rid = (
        f"phase3_timing:Qwen:N=13:L=512:Q=64:ddi_bridge_k_k{k}:sdpa:"
        f"seed={seed}:abcd1234"
    )
    _write_row(store, rid, "ftt_median_ms", ftt_ms, seed)


def _full_ftt(store: Path, seed: int, ftt_ms: float) -> None:
    rid = f"phase3_timing:Qwen:N=13:L=512:Q=64:full:sdpa:seed={seed}:abcd1234"
    _write_row(store, rid, "ftt_median_ms", ftt_ms, seed)


def test_f6_registered():
    assert "F6" in list_reports()


def test_f6_raises_when_no_pairs(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    store = tmp_path / "runs.jsonl"
    store.touch()
    # bridge EM without any timing rows — no pairs possible.
    for seed in (0, 1):
        _bridge_em(store, k=4, seed=seed, em=0.6)
    out = tmp_path / "F6.pdf"
    with pytest.raises(ValueError, match="F6: no .bridge, timing. pairs"):
        render("F6", store=store, output=out)


def test_f6_raises_when_only_timing_pairs_missing_em(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    store = tmp_path / "runs.jsonl"
    # Timing rows present including full baseline, but no bridge EM to pair with.
    _full_ftt(store, seed=0, ftt_ms=720.0)
    _bridge_ftt(store, k=4, seed=0, ftt_ms=100.0)
    out = tmp_path / "F6.pdf"
    with pytest.raises(ValueError, match="F6: no .bridge, timing. pairs"):
        render("F6", store=store, output=out)


def test_f6_renders_pdf(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    # 3 Ks with both EM (Phase 2.3) and FTT (Phase 3.2) arms.
    for seed in (0, 1, 2):
        _full_em(store, seed=seed, em=0.70)
        _full_ftt(store, seed=seed, ftt_ms=720.0)
        for k in (1, 4, 16):
            _bridge_em(store, k=k, seed=seed, em=0.60 + k * 0.005)
            _bridge_ftt(store, k=k, seed=seed, ftt_ms=200.0 - k * 5.0)

    out = tmp_path / "F6.pdf"
    render("F6", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f6_renders_with_delta_band(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for seed in (0, 1, 2):
        _full_em(store, seed=seed, em=0.70)
        _full_ftt(store, seed=seed, ftt_ms=720.0)
        for k in (1, 4):
            _bridge_em(store, k=k, seed=seed, em=0.65)
            _bridge_ftt(store, k=k, seed=seed, ftt_ms=200.0)
    # Δ from ddi.delta writes.
    _write_row(
        store,
        "phase1_delta:Qwen:nq:metric=em:k=2",
        "delta_em",
        0.02,
    )

    out = tmp_path / "F6_delta.pdf"
    render("F6", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0
