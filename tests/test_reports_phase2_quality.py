"""Tests for ddi/reports/phase2_quality.py — T2 renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render


def _write(store: Path, run_id: str, metric: str, value: float, seed: int = 0) -> None:
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


def test_t2_is_registered():
    assert "T2" in list_reports()


def test_t2_raises_when_no_phase2_rows(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase2_train_free rows"):
        render("T2", store=store)


def test_t2_renders_rows_for_each_variant(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for seed in (0, 1, 2):
        for record in range(3):
            _write(store, f"phase2_train_free:Qwen:nq:full:seed={seed}:sha:r{record}", "em", 0.5 + 0.01 * seed, seed)
            _write(store, f"phase2_train_free:Qwen:nq:ddi_qf:seed={seed}:sha:r{record}", "em", 0.6 + 0.01 * seed, seed)
    out = render("T2", store=store)
    assert "Qwen" in out
    assert "full" in out
    assert "ddi_qf" in out
    # seeds column should show 3
    assert "| 3 |" in out


def test_t2_shows_na_when_metric_missing(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Only EM rows for one variant — F1/presence/loss should render n/a
    _write(store, "phase2_train_free:Qwen:nq:full:seed=0:sha:r0", "em", 0.5, 0)
    _write(store, "phase2_train_free:Qwen:nq:full:seed=1:sha:r0", "em", 0.5, 1)
    _write(store, "phase2_train_free:Qwen:nq:full:seed=2:sha:r0", "em", 0.5, 2)
    out = render("T2", store=store)
    assert "n/a" in out
