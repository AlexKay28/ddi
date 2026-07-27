"""Tests for ddi/reports/phase5_hop_split.py — T5 table and F4 figure renderers (issue #26)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from ddi.data import load, manifest, register_loader
from ddi.data.loaders import _REGISTRY
from ddi.report import list_reports, render


@pytest.fixture(autouse=True)
def _manifest_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the split manifest so real loaders can compute + write hashes hermetically."""
    root = tmp_path / "splits"
    monkeypatch.setattr(manifest, "DEFAULT_ROOT", root)
    return root


@pytest.fixture
def isolate_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot the loader registry so tests can `register_loader` freely."""
    monkeypatch.setattr("ddi.data.loaders._REGISTRY", dict(_REGISTRY))


def _write_row(
    store: Path, run_id: str, metric: str, value: float, seed: int = 0
) -> None:
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


def _rid(
    *,
    model: str = "Qwen/Qwen2.5-7B",
    dataset: str,
    variant: str,
    seed: int,
    sha: str = "abcd1234",
    record_id: str,
) -> str:
    return (
        f"phase2_train_free:{model}:{dataset}:{variant}:seed={seed}:{sha}:{record_id}"
    )


def test_t5_and_f4_registered() -> None:
    reports = list_reports()
    assert "T5" in reports
    assert "F4" in reports


def test_t5_raises_when_no_phase2_rows(tmp_path: Path) -> None:
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase2_train_free"):
        render("T5", store=store)


def test_t5_uses_synthetic_qa_hop_types(tmp_path: Path) -> None:
    """synthetic_qa records are all single-hop, so the multi column must render as n/a."""
    store = tmp_path / "runs.jsonl"
    records = list(load("synthetic_qa", "val", 0))
    assert records, "synthetic_qa val split should be non-empty"
    for rec in records[:5]:
        _write_row(
            store,
            _rid(dataset="synthetic_qa", variant="ddi_qf", seed=0, record_id=rec["id"]),
            "em",
            0.5,
            seed=0,
        )
        _write_row(
            store,
            _rid(dataset="synthetic_qa", variant="full", seed=0, record_id=rec["id"]),
            "em",
            0.8,
            seed=0,
        )

    out = render("T5", store=store)
    assert isinstance(out, str)
    assert "synthetic_qa" in out
    assert "ddi_qf" in out and "full" in out
    # synthetic_qa is all single-hop → multi column must be n/a.
    # Find the ddi_qf row and inspect its columns.
    ddi_line = [ln for ln in out.splitlines() if "ddi_qf" in ln][0]
    cols = [c.strip() for c in ddi_line.split("|")]
    # cols: ['', 'dataset', 'variant', 'EM (single)', 'EM (multi)', 'EM (all)', 'Δ EM (multi − single)', '']
    assert cols[3] == "0.5000"  # EM (single)
    assert cols[4] == "n/a"  # EM (multi)
    assert cols[5] == "0.5000"  # EM (all)
    assert cols[6] == "n/a"  # Δ can't be computed


def test_t5_multi_and_single_split(
    tmp_path: Path, isolate_loaders: None
) -> None:
    """A hand-built loader with mixed hop_types must split cleanly in the table."""
    fake_records = [
        {"id": "r_single_0", "hop_type": "single"},
        {"id": "r_single_1", "hop_type": "single"},
        {"id": "r_multi_0", "hop_type": "multi"},
        {"id": "r_multi_1", "hop_type": "multi"},
    ]

    @register_loader("mixed_hop_ds")
    def _loader(split: str, seed: int) -> Iterator[dict]:
        for rec in fake_records:
            yield dict(rec)

    store = tmp_path / "runs.jsonl"
    # ddi_qf: single=0.9, multi=0.2 (large degradation on multi as the paper predicts).
    # full  : single=0.9, multi=0.85 (near-parity across hops).
    scores = {
        ("ddi_qf", "r_single_0"): 0.9,
        ("ddi_qf", "r_single_1"): 0.9,
        ("ddi_qf", "r_multi_0"): 0.2,
        ("ddi_qf", "r_multi_1"): 0.2,
        ("full", "r_single_0"): 0.9,
        ("full", "r_single_1"): 0.9,
        ("full", "r_multi_0"): 0.85,
        ("full", "r_multi_1"): 0.85,
    }
    for (variant, rid_val), val in scores.items():
        _write_row(
            store,
            _rid(dataset="mixed_hop_ds", variant=variant, seed=0, record_id=rid_val),
            "em",
            val,
            seed=0,
        )

    out = render("T5", store=store)
    assert "mixed_hop_ds" in out
    ddi_line = [ln for ln in out.splitlines() if "ddi_qf" in ln][0]
    ddi_cols = [c.strip() for c in ddi_line.split("|")]
    assert ddi_cols[3] == "0.9000"  # single
    assert ddi_cols[4] == "0.2000"  # multi
    assert ddi_cols[5] == "0.5500"  # all (mean of four values = 0.55)
    assert ddi_cols[6] == "-0.7000"  # multi − single degradation

    full_line = [ln for ln in out.splitlines() if "| full |" in ln][0]
    full_cols = [c.strip() for c in full_line.split("|")]
    assert full_cols[3] == "0.9000"
    assert full_cols[4] == "0.8500"
    assert full_cols[6] == "-0.0500"  # small degradation, well within Δ typically


def test_hop_type_unknown_column_present_when_any_row_is_unknown(
    tmp_path: Path,
) -> None:
    """When the loader isn't registered, every record falls to unknown → the column appears."""
    store = tmp_path / "runs.jsonl"
    _write_row(
        store,
        _rid(dataset="ghost_ds", variant="ddi_qf", seed=0, record_id="rec_a"),
        "em",
        0.4,
        seed=0,
    )
    _write_row(
        store,
        _rid(dataset="ghost_ds", variant="full", seed=0, record_id="rec_a"),
        "em",
        0.7,
        seed=0,
    )

    out = render("T5", store=store)
    assert "EM (unknown)" in out
    # And the inline comment should note how many rows dropped.
    assert "dropped to hop_type=unknown" in out


def test_f4_renders_pdf(tmp_path: Path, isolate_loaders: None) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    fake_records = [
        {"id": "r_s_0", "hop_type": "single"},
        {"id": "r_s_1", "hop_type": "single"},
        {"id": "r_m_0", "hop_type": "multi"},
    ]

    @register_loader("f4_mixed_ds")
    def _loader(split: str, seed: int) -> Iterator[dict]:
        for rec in fake_records:
            yield dict(rec)

    store = tmp_path / "runs.jsonl"
    for seed in (0, 1):
        for variant, base in (("full", 0.8), ("ddi_qf", 0.5)):
            for rec in fake_records:
                bump = 0.05 if rec["hop_type"] == "single" else -0.1
                _write_row(
                    store,
                    _rid(
                        dataset="f4_mixed_ds",
                        variant=variant,
                        seed=seed,
                        record_id=rec["id"],
                    ),
                    "em",
                    base + bump + 0.01 * seed,
                    seed=seed,
                )

    out = tmp_path / "F4.pdf"
    render("F4", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f4_raises_when_no_rows(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    out = tmp_path / "F4.pdf"
    with pytest.raises(ValueError, match="phase2_train_free"):
        render("F4", store=store, output=out)
