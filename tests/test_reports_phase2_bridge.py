"""Tests for ddi/reports/phase2_bridge.py — F2 renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render
from ddi.reports.phase2_bridge import _extract_bridge_k


def _write_row(store: Path, run_id: str, metric: str, value: float) -> None:
    with store.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "value": value,
                    "provenance": {"seed": 0},
                }
            )
            + "\n"
        )


def test_f2_is_registered():
    assert "F2" in list_reports()


def test_extract_bridge_k_from_valid_tag():
    rid = "phase2_train_free:Qwen/Qwen2.5-7B:nq:ddi_bridge_k_k4:seed=0:abcd1234:ex_0"
    assert _extract_bridge_k(rid) == 4


def test_extract_bridge_k_returns_none_for_non_bridge():
    rid = "phase2_train_free:Qwen/Qwen2.5-7B:nq:ddi_qf:seed=0:abcd1234:ex_0"
    assert _extract_bridge_k(rid) is None
    rid = "phase1_baseline:Qwen/Qwen2.5-7B:nq:seed=0:abcd1234:ex_0"
    assert _extract_bridge_k(rid) is None


def test_f2_raises_when_store_has_no_bridge_rows(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    out = tmp_path / "F2.pdf"
    with pytest.raises(ValueError, match="phase2_train_free rows with variant=ddi_bridge_k"):
        render("F2", store=store, output=out)


def test_f2_renders_to_pdf(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for k in (1, 2, 4, 8):
        for seed in (0, 1, 2):
            rid = f"phase2_train_free:Qwen:nq:ddi_bridge_k_k{k}:seed={seed}:sha:ex_{seed}"
            _write_row(store, rid, "em", 0.5 + k * 0.01 + seed * 0.001)

    out = tmp_path / "F2.pdf"
    render("F2", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f2_renders_em_only_metric_present(tmp_path: Path):
    """EM alone is enough to render — F1/presence lines are added only when present."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for k in (1, 26):
        for seed in (0, 1, 2):
            rid = f"phase2_train_free:Qwen:composition:ddi_bridge_k_k{k}:seed={seed}:sha:ex_{seed}"
            _write_row(store, rid, "em", 0.0)

    out = tmp_path / "F2.pdf"
    render("F2", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f2_plots_em_f1_and_presence_together(tmp_path: Path):
    """The #36 shape: EM is flat/degenerate but F1 and presence carry real separation."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    values = {
        1: {"em": 0.0, "f1": 0.05, "presence": 0.82},
        26: {"em": 0.0, "f1": 0.40, "presence": 0.97},
    }
    for k, metrics in values.items():
        for seed in (0, 1, 2):
            rid = f"phase2_train_free:Qwen:composition:ddi_bridge_k_k{k}:seed={seed}:sha:ex_{seed}"
            for metric, value in metrics.items():
                _write_row(store, rid, metric, value)

    out = tmp_path / "F2.pdf"
    render("F2", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0
