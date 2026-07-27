"""Tests for ddi/reports/phase4_throughput.py — F5 renderer (issue #25)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render
from ddi.reports.phase4_throughput import _parse_run_id


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


def test_f5_registered() -> None:
    assert "F5" in list_reports()


def test_parse_run_id_extracts_mode() -> None:
    rid = "phase4_throughput:Qwen:ddi_cache:restart:reuse=zipf(a=1.5):seed=0:abcd1234"
    parsed = _parse_run_id(rid)
    assert parsed == ("ddi_cache", "reuse=zipf(a=1.5)")


def test_parse_run_id_rejects_wrong_prefix() -> None:
    assert _parse_run_id("phase3_timing:foo:bar:baz") is None


def test_parse_run_id_rejects_unknown_mode() -> None:
    rid = "phase4_throughput:Qwen:bogus:restart:reuse=zipf(a=1.5):seed=0:abcd1234"
    assert _parse_run_id(rid) is None


def test_f5_raises_when_no_throughput_rows(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    out = tmp_path / "F5.pdf"
    with pytest.raises(ValueError, match="phase4_throughput rows"):
        render("F5", store=store, output=out)


def test_f5_raises_when_qps_rows_have_no_hit_rate(tmp_path: Path) -> None:
    """qps rows alone are not enough — the join needs both."""
    store = tmp_path / "runs.jsonl"
    _write_row(
        store,
        "phase4_throughput:Qwen:ddi_cache:restart:reuse=zipf(a=1.5):seed=0:sha",
        "qps",
        7.5,
    )
    out = tmp_path / "F5.pdf"
    with pytest.raises(ValueError, match="phase4_throughput rows"):
        render("F5", store=store, output=out)


def test_f5_renders_pdf(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    # Seed one run per mode with a qps + hit_rate pair. ddi_cache gets multiple runs at
    # varying hit rates so the curve has more than one point.
    for hit_rate, qps in [(0.2, 3.0), (0.5, 6.5), (0.85, 18.0)]:
        rid = (
            f"phase4_throughput:Qwen:ddi_cache:restart:reuse=zipf(a=1.5):"
            f"seed={int(hit_rate * 100)}:sha"
        )
        _write_row(store, rid, "qps", qps)
        _write_row(store, rid, "hit_rate", hit_rate)

    rid_recompute = "phase4_throughput:Qwen:ddi_recompute:restart:reuse=uniform:seed=0:sha"
    _write_row(store, rid_recompute, "qps", 2.5)
    _write_row(store, rid_recompute, "hit_rate", 0.0)

    rid_full = "phase4_throughput:Qwen:full_causal:restart:reuse=uniform:seed=0:sha"
    _write_row(store, rid_full, "qps", 1.2)
    _write_row(store, rid_full, "hit_rate", 0.0)

    out = tmp_path / "F5.pdf"
    render("F5", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f5_ignores_per_request_rows(tmp_path: Path) -> None:
    """Per-request `per_request_latency_ms` rows must not confuse the qps join."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    base = "phase4_throughput:Qwen:ddi_cache:restart:reuse=zipf(a=1.5):seed=0:sha"
    _write_row(store, base, "qps", 5.0)
    _write_row(store, base, "hit_rate", 0.4)
    # Add per-request rows — these should be ignored by the F5 join (they carry no qps
    # of their own and their run_id has an extra :reqN suffix).
    for i in range(5):
        _write_row(store, f"{base}:req{i}", "per_request_latency_ms", 200.0)

    out = tmp_path / "F5.pdf"
    render("F5", store=store, output=out)
    assert out.exists()
