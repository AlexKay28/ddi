"""Tests for ddi/reports/phase5_composition.py — T6 (table) and F7 (curve). Issue #28."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render
from ddi.reports.phase5_composition import _parse_run_id


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


def _rid(*, variant: str, layout: str, distance: int, pair_position: str, rec_id: str) -> str:
    return (
        f"phase5_composition:Qwen:composition:{variant}:{layout}:"
        f"d{distance}:p{pair_position}:seed=0:abcd1234:{rec_id}"
    )


def test_t6_and_f7_registered() -> None:
    assert "T6" in list_reports()
    assert "F7" in list_reports()


def test_parse_run_id_extracts_knobs() -> None:
    rid = _rid(variant="ddi_qf", layout="restart", distance=2, pair_position="middle", rec_id="comp_0_000001")
    parsed = _parse_run_id(rid)
    assert parsed is not None
    assert parsed["variant_tag"] == "ddi_qf"
    assert parsed["layout"] == "restart"
    assert parsed["distance"] == 2
    assert parsed["pair_position"] == "middle"


def test_parse_run_id_rejects_wrong_prefix() -> None:
    assert _parse_run_id("phase2_train_free:x:y:ddi_qf:seed=0:sha:ex_0") is None


def test_t6_raises_on_empty_store(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase5_composition"):
        render("T6", store=store)


def test_t6_groups_by_distance(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    # Three distances × two records each, one (variant, layout) combo.
    for distance in (0, 2, 4):
        for j in range(2):
            rid = _rid(
                variant="ddi_qf",
                layout="restart",
                distance=distance,
                pair_position="middle",
                rec_id=f"comp_0_{distance:03d}{j}",
            )
            _write_row(store, rid, "em", 1.0 - distance * 0.1 + j * 0.05)

    table = render("T6", store=store)
    assert isinstance(table, str)
    assert "<!-- provenance -->" in table
    # Header + separator + one row per distance = 3 data rows.
    data_lines = [
        ln for ln in table.strip().splitlines()
        if ln.startswith("| ") and "distance" not in ln and not ln.startswith("|---")
    ]
    assert len(data_lines) == 3
    # Each distance value appears as its own row.
    assert any("| 0 |" in ln for ln in data_lines)
    assert any("| 2 |" in ln for ln in data_lines)
    assert any("| 4 |" in ln for ln in data_lines)


def test_t6_separates_variants(tmp_path: Path) -> None:
    """(variant, layout, distance) is the grouping key — different variants stay in different rows."""
    store = tmp_path / "runs.jsonl"
    for variant in ("full", "ddi_qf"):
        rid = _rid(
            variant=variant, layout="restart", distance=1, pair_position="middle", rec_id=f"comp_0_{variant}"
        )
        _write_row(store, rid, "em", 0.7)

    table = render("T6", store=store)
    data_lines = [
        ln for ln in table.strip().splitlines()
        if ln.startswith("| ") and "distance" not in ln and not ln.startswith("|---")
    ]
    assert len(data_lines) == 2


def test_f7_raises_on_empty_store(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    out = tmp_path / "F7.pdf"
    with pytest.raises(ValueError, match="phase5_composition"):
        render("F7", store=store, output=out)


def test_f7_renders_pdf(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for variant in ("full", "ddi_qf"):
        for distance in (0, 1, 2, 3, 4):
            for j in range(3):
                rid = _rid(
                    variant=variant,
                    layout="restart",
                    distance=distance,
                    pair_position="middle",
                    rec_id=f"comp_0_{variant}_{distance}_{j}",
                )
                # ddi_qf degrades with distance; full stays high.
                em = 0.95 if variant == "full" else max(0.0, 0.9 - distance * 0.2)
                _write_row(store, rid, "em", em + j * 0.01)

    out = tmp_path / "F7.pdf"
    render("F7", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_t6_renders_em_only_metric_present(tmp_path: Path) -> None:
    """EM alone is enough to render — F1/presence columns are added only when present."""
    store = tmp_path / "runs.jsonl"
    rid = _rid(variant="ddi_qf", layout="restart", distance=1, pair_position="middle", rec_id="comp_0_0")
    _write_row(store, rid, "em", 0.0)

    table = render("T6", store=store)
    assert "EM (mean)" in table
    assert "F1 (mean)" not in table
    assert "presence (mean)" not in table


def test_t6_adds_f1_and_presence_columns_when_present(tmp_path: Path) -> None:
    """The #37 shape: EM is flat/degenerate on composition but F1/presence carry separation."""
    store = tmp_path / "runs.jsonl"
    values = {
        ("full", 0): {"em": 0.0, "f1": 0.60, "presence": 0.95},
        ("ddi_qf", 4): {"em": 0.0, "f1": 0.10, "presence": 0.50},
    }
    for (variant, distance), metrics in values.items():
        rid = _rid(
            variant=variant, layout="restart", distance=distance, pair_position="middle",
            rec_id=f"comp_0_{variant}",
        )
        for metric, value in metrics.items():
            _write_row(store, rid, metric, value)

    table = render("T6", store=store)
    assert "F1 (mean)" in table
    assert "presence (mean)" in table
    lines = table.strip().splitlines()
    full_row = next(ln for ln in lines if ln.startswith("| full |"))
    ddi_qf_row = next(ln for ln in lines if ln.startswith("| ddi_qf |"))
    assert "0.6000" in full_row
    assert "0.1000" in ddi_qf_row


def test_f7_plots_em_f1_and_presence_together(tmp_path: Path) -> None:
    """The #37 shape: three metric panels, one per present metric."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for variant in ("full", "ddi_qf"):
        for layout in ("sequential", "restart"):
            for distance in (0, 4):
                for j in range(2):
                    rid = _rid(
                        variant=variant, layout=layout, distance=distance,
                        pair_position="middle", rec_id=f"comp_{j}_{variant}_{layout}_{distance}",
                    )
                    for metric, base in (("em", 0.0), ("f1", 0.3), ("presence", 0.7)):
                        _write_row(store, rid, metric, base + j * 0.01)

    out = tmp_path / "F7.pdf"
    render("F7", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0
