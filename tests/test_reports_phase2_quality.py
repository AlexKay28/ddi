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


# ---- Per-metric renderers (T2_em / T2_f1 / T2_presence / T2_loss, #73) ---------------------


def test_t2_per_metric_renderers_registered():
    reports = list_reports()
    for name in ("T2_em", "T2_f1", "T2_presence", "T2_loss"):
        assert name in reports, f"{name} not registered"


def test_t2_em_fixed_column_order(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Write one em row per dataset column so every cell is populated.
    for ds in (
        "composition",
        "synthetic_qa",
        "hotpot_qa",
        "musique",
        "kilt_nq",
        "ddi_stress__fact_chain",
        "ddi_stress__fact_aggregation",
        "ddi_stress__contradiction",
        "ddi_stress__coreference",
        "ddi_stress__needle",
    ):
        _write(store, f"phase2_train_free:Qwen:{ds}:full:seed=0:sha:r0", "em", 0.5)

    out = render("T2_em", store=store)
    # Composition must appear before every other column, and ds/needle must be last.
    header_line = next(line for line in out.splitlines() if line.startswith("| mask |"))
    cols = [c.strip() for c in header_line.strip().strip("|").split("|")]
    # cols[0] == "mask", then dataset columns in fixed order.
    assert cols == [
        "mask",
        "composition",
        "synthetic_qa",
        "hotpot_qa",
        "musique",
        "kilt_nq",
        "ds/fact_chain",
        "ds/fact_agg",
        "ds/contra",
        "ds/coref",
        "ds/needle",
    ]


def test_t2_em_excludes_nq(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _write(store, "phase2_train_free:Qwen:nq:full:seed=0:sha:r0", "em", 0.9)
    _write(store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "em", 0.3)
    out = render("T2_em", store=store)
    header = next(line for line in out.splitlines() if line.startswith("| mask |"))
    assert " nq " not in header
    assert "composition" in header


def test_t2_em_maps_ddi_stress_family_to_ds_column(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _write(
        store,
        "phase2_train_free:Qwen:ddi_stress__fact_chain:full:seed=0:sha:r0",
        "em",
        0.75,
    )
    _write(
        store,
        "phase2_train_free:Qwen:ddi_stress__fact_aggregation:full:seed=0:sha:r0",
        "em",
        0.25,
    )
    out = render("T2_em", store=store)
    # ds/fact_chain cell should show 0.7500 in the full row.
    full_row = next(
        line for line in out.splitlines() if line.startswith("| full |")
    )
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    # cells[0] == "full"; column order is fixed, so ds/fact_chain is index 6 and
    # ds/fact_agg is index 7.
    assert cells[6] == "0.7500"
    assert cells[7] == "0.2500"


def test_t2_em_missing_cells_render_as_em_dash(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Only fill composition; everything else must be em-dash, not n/a.
    _write(
        store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "em", 0.5
    )
    out = render("T2_em", store=store)
    assert "—" in out
    assert "n/a" not in out


def test_t2_em_bare_ddi_stress_row_dropped(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Older EXP-01 row with no __family suffix — must not land in a stress column.
    _write(
        store, "phase2_train_free:Qwen:ddi_stress:full:seed=0:sha:r0", "em", 0.9
    )
    _write(
        store,
        "phase2_train_free:Qwen:ddi_stress__needle:full:seed=0:sha:r0",
        "em",
        0.2,
    )
    out = render("T2_em", store=store)
    full_row = next(
        line for line in out.splitlines() if line.startswith("| full |")
    )
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    # ds/needle should reflect only the family row (0.2000), not the mixed value.
    assert cells[10] == "0.2000"


def test_t2_em_one_model_one_table(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _write(
        store,
        "phase2_train_free:ModelA:composition:full:seed=0:sha:r0",
        "em",
        0.5,
    )
    out = render("T2_em", store=store)
    # exactly one "###" section for one model.
    assert out.count("### ModelA") == 1


def test_t2_em_multiple_models_multiple_tables(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    _write(
        store,
        "phase2_train_free:ModelA:composition:full:seed=0:sha:r0",
        "em",
        0.5,
    )
    _write(
        store,
        "phase2_train_free:ModelB:composition:ddi_qf:seed=0:sha:r0",
        "em",
        0.4,
    )
    out = render("T2_em", store=store)
    assert "### ModelA" in out
    assert "### ModelB" in out


def test_t2_f1_f1_uses_f1_metric(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # em should NOT bleed into f1 table.
    _write(
        store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "em", 0.9
    )
    _write(
        store, "phase2_train_free:Qwen:composition:full:seed=0:sha:r0", "f1", 0.4
    )
    out = render("T2_f1", store=store)
    full_row = next(
        line for line in out.splitlines() if line.startswith("| full |")
    )
    cells = [c.strip() for c in full_row.strip().strip("|").split("|")]
    assert cells[1] == "0.4000"


def test_t2_per_metric_raises_when_no_phase2_rows(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase2_train_free"):
        render("T2_em", store=store)
