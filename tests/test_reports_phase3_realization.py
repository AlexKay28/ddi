"""Tests for ddi/reports/phase3_realization.py — T3 table and F3 figure renderers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddi.report import list_reports, render


def _write_row(store: Path, run_id: str, metric: str, value: float | str, seed: int = 0) -> None:
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


def _seed_cell(
    store: Path,
    *,
    n: int,
    l: int,
    q: int,
    variant_tag: str,
    ftt_ms: float,
    analytical: float | None = None,
    sparsity: float | None = None,
    backend: str = "sdpa",
    seed: int = 0,
    model: str = "Qwen/Qwen2.5-7B",
) -> None:
    rid = (
        f"phase3_timing:{model}:N={n}:L={l}:Q={q}:{variant_tag}:{backend}:"
        f"seed={seed}:abcd1234"
    )
    _write_row(store, rid, "ftt_median_ms", ftt_ms, seed)
    if analytical is not None:
        _write_row(store, rid, "analytical_speedup", analytical, seed)
    if sparsity is not None:
        _write_row(store, rid, "mask_sparsity", sparsity, seed)


def test_t3_registered_and_f3_registered():
    reports = list_reports()
    assert "T3" in reports
    assert "F3" in reports


def test_t3_empty_store_raises_when_no_phase3_rows(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    store.touch()
    _write_row(store, "phase1_baseline:x:y:seed=0:sha:ex_0", "em", 0.5)
    with pytest.raises(ValueError, match="phase3_timing"):
        render("T3", store=store)


def test_t3_renders_markdown_with_expected_columns(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Two Ns; each has full + ddi_qf ftt across two seeds. Realized = full / ddi_qf.
    for seed in (0, 1):
        _seed_cell(
            store, n=1, l=512, q=64, variant_tag="full", ftt_ms=100.0, seed=seed
        )
        _seed_cell(
            store,
            n=1,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=100.0,
            analytical=1.0,
            sparsity=0.0,
            seed=seed,
        )
        _seed_cell(
            store, n=13, l=512, q=64, variant_tag="full", ftt_ms=720.0, seed=seed
        )
        _seed_cell(
            store,
            n=13,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=600.0,
            analytical=10.59,
            sparsity=0.906,
            seed=seed,
        )

    out = render("T3", store=store)
    assert isinstance(out, str)
    # Header shape from RESEARCH_PLAN §7.
    assert (
        "| N | analytical | realized (median) | realization ratio | sparsity |" in out
    )
    # N=1 row: analytical=1.00x, realized=1.00x, ratio=1.000
    assert "| 1 | 1.00x | 1.00x | 1.000 | 0.000 |" in out
    # N=13 row: analytical=10.59x, realized=720/600=1.20x, ratio=1.20/10.59=0.113
    assert "| 13 | 10.59x | 1.20x | 0.113 | 0.906 |" in out


def test_t3_renders_na_for_missing_arm(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    # Only a `full` arm at N=5 — no ddi_qf to pair with.
    _seed_cell(store, n=5, l=512, q=64, variant_tag="full", ftt_ms=200.0)
    # analytical alone (no ftt) still creates a row for N=9.
    _seed_cell(
        store,
        n=9,
        l=512,
        q=64,
        variant_tag="ddi_qf",
        ftt_ms=100.0,
        analytical=7.39,
        sparsity=0.865,
    )
    out = render("T3", store=store)
    # N=5: realized missing => n/a for realized and ratio, sparsity also n/a
    assert "| 5 |" in out
    assert "n/a" in out


def test_t3_is_idempotent(tmp_path: Path):
    store = tmp_path / "runs.jsonl"
    for seed in (0, 1, 2):
        _seed_cell(
            store, n=3, l=512, q=64, variant_tag="full", ftt_ms=100.0 + seed, seed=seed
        )
        _seed_cell(
            store,
            n=3,
            l=512,
            q=64,
            variant_tag="ddi_qf",
            ftt_ms=50.0 + seed,
            analytical=2.59,
            sparsity=0.614,
            seed=seed,
        )
    a = render("T3", store=store)
    b = render("T3", store=store)
    assert a == b


def test_f3_renders_pdf(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    store = tmp_path / "runs.jsonl"
    for l in (128, 512):
        for n in (1, 5, 13):
            for seed in (0, 1):
                _seed_cell(
                    store,
                    n=n,
                    l=l,
                    q=64,
                    variant_tag="full",
                    ftt_ms=100.0 * n,
                    seed=seed,
                )
                _seed_cell(
                    store,
                    n=n,
                    l=l,
                    q=64,
                    variant_tag="ddi_qf",
                    ftt_ms=100.0 * n / 2.0,
                    analytical=float(n),
                    sparsity=0.5,
                    seed=seed,
                )

    out = tmp_path / "F3.pdf"
    render("F3", store=store, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f3_empty_store_raises(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    store = tmp_path / "runs.jsonl"
    store.touch()
    out = tmp_path / "F3.pdf"
    with pytest.raises(ValueError, match="phase3_timing"):
        render("F3", store=store, output=out)
